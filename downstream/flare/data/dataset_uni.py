"""Deterministic, temporally uniform train/validation flare dataset splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .class_groups import DEFAULT_CLASS_GROUPS
from .dataset import (
    DEFAULT_LABEL_PATH,
    FlareDataset,
    load_daily_label_table,
)


_HELD_OUT_SPLITS = {"validation", "test"}


def normalize_split(split: str) -> str:
    """Normalize the public split name while keeping ``test`` auditable."""

    if not isinstance(split, str):
        raise TypeError("split must be a string")
    normalized = split.strip().lower()
    if normalized == "val":
        normalized = "validation"
    if normalized not in {"train", *_HELD_OUT_SPLITS}:
        raise ValueError(
            "split must be one of 'train', 'validation', 'val', or 'test'; "
            f"got {split!r}"
        )
    return normalized


def validate_validation_ratio(validation_ratio: float) -> float:
    """Return a finite ratio strictly between zero and one."""

    if isinstance(validation_ratio, (bool, np.bool_)):
        raise TypeError("validation_ratio must be a real number, not a boolean")
    try:
        ratio = float(validation_ratio)
    except (TypeError, ValueError) as error:
        raise TypeError("validation_ratio must be a real number") from error
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(
            "validation_ratio must be finite and strictly between 0 and 1; "
            f"got {validation_ratio!r}"
        )
    return ratio


def evenly_spaced_validation_positions(
    total_size: int,
    validation_ratio: float = 0.2,
) -> np.ndarray:
    """Choose one deterministic sample near the center of each temporal bin.

    The returned positions never use a random-number generator.  The requested
    validation count is rounded to the nearest integer and clamped so that both
    train and validation remain non-empty.
    """

    if isinstance(total_size, (bool, np.bool_)) or not isinstance(
        total_size, (int, np.integer)
    ):
        raise TypeError("total_size must be an integer")
    total_size = int(total_size)
    if total_size < 2:
        raise ValueError(
            "A uniform train/validation split requires at least two samples; "
            f"got {total_size}"
        )
    ratio = validate_validation_ratio(validation_ratio)
    validation_size = int(np.floor(total_size * ratio + 0.5))
    validation_size = min(max(validation_size, 1), total_size - 1)

    # Integer arithmetic selects the center of each of validation_size equal
    # temporal bins. Since total_size / validation_size > 1, positions are
    # strictly increasing without a deduplication or seed-dependent tie break.
    bin_numbers = np.arange(validation_size, dtype=np.int64)
    return (((2 * bin_numbers + 1) * total_size) // (2 * validation_size)).astype(
        np.int64, copy=False
    )


class FlareDatasetUni(FlareDataset):
    """Split the complete HMI/label intersection at uniform temporal positions.

    ``validation`` and ``test`` select the same held-out partition so that the
    training config recorded in a run can be reconstructed directly by
    :mod:`downstream.flare.test`.  ``train`` selects its exact complement.
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
        label_path: str | Path = DEFAULT_LABEL_PATH,
        label_summary_path: str | Path | None = None,
        verify_label_summary: bool = True,
        expected_event_time_column: str = "start_time",
        return_date_id: bool = False,
        class_groups: Sequence[str] | None = DEFAULT_CLASS_GROUPS,
        split: str = "train",
        validation_ratio: float = 0.2,
    ) -> None:
        resolved_split = normalize_split(split)
        resolved_ratio = validate_validation_ratio(validation_ratio)
        if isinstance(time_step, (bool, np.bool_)) or not isinstance(
            time_step, (int, np.integer)
        ):
            raise TypeError("time_step must be a positive integer")
        resolved_time_step = int(time_step)
        if resolved_time_step < 1:
            raise ValueError("time_step must be a positive integer")

        # With no explicit interval, use the complete date coverage of the
        # label table. The parent then intersects it with the available HMI
        # indices (and any explicitly requested additional modalities).
        if time_interval is None:
            available_labels = load_daily_label_table(label_path)
            resolved_interval = [
                min(available_labels),
                max(available_labels) + 1,
            ]
        else:
            if len(time_interval) != 2:
                raise ValueError("time_interval must contain exactly [start, end]")
            resolved_interval = [int(time_interval[0]), int(time_interval[1])]
            if resolved_interval[0] >= resolved_interval[1]:
                raise ValueError("time_interval must satisfy start < end")

        # This downstream task is HMI-only by default. Passing modal_list keeps
        # the parent's normal multimodal-intersection behavior available.
        resolved_modal_list = ["hmi"] if modal_list is None else modal_list
        super().__init__(
            modal_list=resolved_modal_list,
            log1p_scale=log1p_scale,
            load_imgs=load_imgs,
            torch_augment_type=torch_augment_type,
            time_interval=resolved_interval,
            time_step=resolved_time_step,
            enhance_type=enhance_type,
            label_path=label_path,
            label_summary_path=label_summary_path,
            verify_label_summary=verify_label_summary,
            expected_event_time_column=expected_event_time_column,
            return_date_id=return_date_id,
            class_groups=class_groups,
        )

        full_date_ids = np.asarray(self.exist_idx)
        validation_positions = evenly_spaced_validation_positions(
            len(full_date_ids), resolved_ratio
        )
        validation_mask = np.zeros(len(full_date_ids), dtype=bool)
        validation_mask[validation_positions] = True
        train_mask = ~validation_mask

        validation_date_ids = full_date_ids[validation_mask]
        train_date_ids = full_date_ids[train_mask]
        selected_date_ids = (
            train_date_ids if resolved_split == "train" else validation_date_ids
        )

        # Preserve the parent's ndarray/list convention for downstream callers.
        if isinstance(self.exist_idx, np.ndarray):
            self.exist_idx = np.asarray(selected_date_ids, dtype=self.exist_idx.dtype)
        else:
            self.exist_idx = [int(value) for value in selected_date_ids]

        self.split = resolved_split
        self.partition = "train" if resolved_split == "train" else "validation"
        self.validation_ratio = resolved_ratio
        self.effective_validation_ratio = len(validation_date_ids) / len(full_date_ids)
        self.time_interval = tuple(resolved_interval)
        self.time_step = resolved_time_step
        self.all_exist_idx = tuple(int(value) for value in full_date_ids)
        self.train_date_ids = tuple(int(value) for value in train_date_ids)
        self.validation_date_ids = tuple(int(value) for value in validation_date_ids)
        self.validation_positions = tuple(int(value) for value in validation_positions)
        self.num_total_samples = len(full_date_ids)
        self.num_train_samples = len(train_date_ids)
        self.num_validation_samples = len(validation_date_ids)

        # FlareDataset computed these counts for the complete pool. Consumers
        # use them for class weighting, so refresh them for the actual split.
        selected_ids = [int(value) for value in selected_date_ids]
        raw_counts = Counter(self.labels_by_date_id[value] for value in selected_ids)
        self.raw_class_counts = {label: raw_counts.get(label, 0) for label in range(6)}
        grouped_counts = Counter(
            self.grouped_labels_by_date_id[value] for value in selected_ids
        )
        self.class_counts = {
            label: grouped_counts.get(label, 0) for label in range(self.num_classes)
        }
        print(
            "Uniform flare split "
            f"{resolved_split!r}: total={self.num_total_samples}, "
            f"train={self.num_train_samples}, "
            f"validation={self.num_validation_samples}, "
            f"selected counts={self.class_counts}"
        )


# Compatibility spelling for configs that use the requested ``_uni`` suffix
# on both the module and class name. New configs should prefer FlareDatasetUni.
FlareDataset_uni = FlareDatasetUni


__all__ = [
    "FlareDatasetUni",
    "FlareDataset_uni",
    "evenly_spaced_validation_positions",
]
