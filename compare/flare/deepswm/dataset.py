"""Daily, gap-free HMI windows for the DeepSWM adaptation.

The class deliberately inherits the project's labelled ``FlareDataset`` so
that labels, split boundaries, normalization and HMI file discovery stay on
the existing SolarCHIP data path.  Only the sampling unit changes from one day
to a strictly consecutive daily window.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from data.dataset.SolarDataset import (
    enhance_funciton,
    image_preprocess,
)
from data.utils import read_pt_image as read_image
from downstream.flare.data.dataset import (
    DEFAULT_CLASS_GROUPS,
    DEFAULT_LABEL_PATH,
    FlareDataset,
)


class DeepSWMSequenceDataset(FlareDataset):
    """Return gap-free daily HMI sequences with the last day's flare label.

    A target day ``D`` is retained only when every day in
    ``[D-window_length+1, ..., D]`` occurs in the *same parent dataset
    instance*.  Because a parent instance represents exactly one configured
    split, this prevents both gap filling and train/validation leakage.

    The parent image preprocessor concatenates the frames as channels before
    sampling horizontal flip, vertical flip and rotation.  Consequently each
    spatial transform is identical for all frames in a window.
    """

    def __init__(
        self,
        modal_list: Sequence[str] | None = None,
        log1p_scale: float = 1,
        load_imgs: bool = False,
        torch_augment_type: Sequence[float] | None = None,
        time_interval: Sequence[int] | None = None,
        time_step: int = 1,
        enhance_type: Sequence[str] | None = None,
        label_path=DEFAULT_LABEL_PATH,
        label_summary_path=None,
        verify_label_summary: bool = True,
        expected_event_time_column: str = "start_time",
        return_date_id: bool = False,
        class_groups: Sequence[str] | None = DEFAULT_CLASS_GROUPS,
        window_length: int = 1,
    ) -> None:
        if int(window_length) < 1:
            raise ValueError("window_length must be at least one day")
        if int(window_length) > 1 and int(time_step) != 1:
            raise ValueError(
                "DeepSWMSequenceDataset requires time_step=1 for multi-day "
                "windows; otherwise consecutive calendar days are excluded"
            )

        resolved_modal_list = ["hmi"] if modal_list is None else list(modal_list)
        if resolved_modal_list != ["hmi"]:
            raise ValueError(
                "DeepSWM HMI-only adaptation requires modal_list=['hmi']; "
                f"received {resolved_modal_list!r}"
            )

        super().__init__(
            modal_list=resolved_modal_list,
            log1p_scale=log1p_scale,
            load_imgs=load_imgs,
            torch_augment_type=torch_augment_type,
            time_interval=time_interval,
            time_step=time_step,
            enhance_type=enhance_type,
            label_path=label_path,
            label_summary_path=label_summary_path,
            verify_label_summary=verify_label_summary,
            expected_event_time_column=expected_event_time_column,
            return_date_id=return_date_id,
            class_groups=class_groups,
        )

        self.window_length = int(window_length)
        split_date_ids = {int(value) for value in self.exist_idx}
        target_date_ids = [
            end_date_id
            for end_date_id in sorted(split_date_ids)
            if all(
                (end_date_id - offset) in split_date_ids
                for offset in range(self.window_length)
            )
        ]
        self.exist_idx = np.asarray(target_date_ids, dtype=np.int64)
        self.window_date_ids = tuple(
            tuple(range(end - self.window_length + 1, end + 1))
            for end in target_date_ids
        )

        counts = Counter(
            self.grouped_labels_by_date_id[date_id] for date_id in target_date_ids
        )
        self.class_counts = {
            class_id: counts.get(class_id, 0)
            for class_id in range(self.num_classes)
        }
        self.num_dropped_for_windows = len(split_date_ids) - len(target_date_ids)
        print(
            "DeepSWM daily windows: "
            f"T={self.window_length}, retained={len(self.exist_idx)}, "
            f"dropped={self.num_dropped_for_windows}, "
            f"class_counts={self.class_counts}"
        )

    def _load_window(self, window_date_ids: tuple[int, ...]) -> torch.Tensor:
        raw_frames = []
        hmi_dataset = self.dataset[0]
        for date_id in window_date_ids:
            path, exists = hmi_dataset[date_id]
            if not bool(exists):
                # This should be unreachable because __init__ filters windows,
                # but keeping the check makes filesystem/index corruption loud.
                raise RuntimeError(
                    f"HMI availability changed after window construction: {date_id}"
                )
            frame = read_image(path)
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 2:
                raise ValueError(
                    "DeepSWM expects each stored HMI magnetogram to be 2-D "
                    f"(or singleton-channel 3-D); date_id={date_id} has "
                    f"shape={tuple(frame.shape)}"
                )
            raw_frames.append(frame)

        image_size, p_flip, p_rotate = self.torch_augment_type
        sequence = image_preprocess(
            raw_frames,
            image_size=image_size,
            p_flip=p_flip,
            p_rotate=p_rotate,
        )
        # Repeating the modality name gives one HMI normalization statistic per
        # time step while preserving the parent preprocessing implementation.
        sequence = enhance_funciton(
            sequence,
            modal_list=["hmi"] * self.window_length,
            enhance_type=self.enhance_type,
            log1p_scale=self.log1p_scale,
        )
        expected_shape = (
            self.window_length,
            1,
            int(image_size),
            int(image_size),
        )
        if not torch.is_tensor(sequence) or tuple(sequence.shape) != expected_shape:
            raise RuntimeError(
                "DeepSWM sequence preprocessing violated its [T,1,H,W] "
                f"contract: expected {expected_shape}, got "
                f"{getattr(sequence, 'shape', None)}"
            )
        return sequence

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        position = int(position)
        window = self.window_date_ids[position]
        target_date_id = window[-1]
        sample: dict[str, torch.Tensor] = {"hmi": self._load_window(window)}

        if getattr(self, "_omit_targets_for_statistics", False):
            return sample

        sample["label"] = torch.tensor(
            self.grouped_labels_by_date_id[target_date_id], dtype=torch.long
        )
        if self.return_date_id:
            sample["date_id"] = torch.tensor(target_date_id, dtype=torch.long)
        return sample


__all__ = ["DeepSWMSequenceDataset"]
