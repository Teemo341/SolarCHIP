"""Deterministic ratio subsets of the uniform flare training partition."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .class_groups import DEFAULT_CLASS_GROUPS
from .dataset import DEFAULT_LABEL_PATH
from .dataset_uni import FlareDatasetUni, normalize_split


def validate_train_ratio(train_ratio: float) -> float:
    """Return a finite training-data ratio in the interval ``(0, 1]``."""

    if isinstance(train_ratio, (bool, np.bool_)):
        raise TypeError("train_ratio must be a real number, not a boolean")
    try:
        ratio = float(train_ratio)
    except (TypeError, ValueError) as error:
        raise TypeError("train_ratio must be a real number") from error
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(
            "train_ratio must be finite and in the interval (0, 1]; "
            f"got {train_ratio!r}"
        )
    return ratio


def evenly_spaced_train_positions(
    total_size: int,
    train_ratio: float,
) -> np.ndarray:
    """Choose deterministic temporal-bin centers from an ordered train split.

    The requested sample count is rounded to the nearest integer and clamped to
    at least one sample.  A ratio of one returns every position unchanged.
    """

    if isinstance(total_size, (bool, np.bool_)) or not isinstance(
        total_size, (int, np.integer)
    ):
        raise TypeError("total_size must be an integer")
    total_size = int(total_size)
    if total_size < 1:
        raise ValueError(
            "A training subset requires at least one sample; "
            f"got {total_size}"
        )

    ratio = validate_train_ratio(train_ratio)
    subset_size = int(np.floor(total_size * ratio + 0.5))
    subset_size = min(max(subset_size, 1), total_size)

    # Select the center of each equal temporal bin using integer arithmetic.
    # total_size / subset_size >= 1, so positions are strictly increasing.
    bin_numbers = np.arange(subset_size, dtype=np.int64)
    return (((2 * bin_numbers + 1) * total_size) // (2 * subset_size)).astype(
        np.int64, copy=False
    )


class FlareDatasetSplit(FlareDatasetUni):
    """Use an evenly spaced fraction of ``FlareDatasetUni``'s train split.

    This dataset intentionally accepts only ``split='train'``.  Validation must
    continue to use :class:`FlareDatasetUni`, which prevents ``train_ratio``
    from changing the held-out samples by accident.  The inherited
    ``train_date_ids`` and ``num_train_samples`` continue to describe the full
    train/validation partition; ``sampled_train_date_ids`` and
    ``num_sampled_train_samples`` describe this dataset's smaller train subset.
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
        train_ratio: float = 1.0,
    ) -> None:
        resolved_split = normalize_split(split)
        if resolved_split != "train":
            raise ValueError(
                "FlareDatasetSplit only supports split='train'; use "
                "FlareDatasetUni for validation or test data"
            )
        resolved_train_ratio = validate_train_ratio(train_ratio)

        super().__init__(
            modal_list=modal_list,
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
            split=resolved_split,
            validation_ratio=validation_ratio,
        )

        full_train_date_ids = np.asarray(self.exist_idx)
        subset_positions = evenly_spaced_train_positions(
            len(full_train_date_ids), resolved_train_ratio
        )
        sampled_train_date_ids = full_train_date_ids[subset_positions]

        if isinstance(self.exist_idx, np.ndarray):
            self.exist_idx = np.asarray(
                sampled_train_date_ids, dtype=self.exist_idx.dtype
            )
        else:
            self.exist_idx = [int(value) for value in sampled_train_date_ids]

        self.train_ratio = resolved_train_ratio
        self.full_train_date_ids = tuple(int(value) for value in full_train_date_ids)
        self.train_subset_positions = tuple(int(value) for value in subset_positions)
        self.sampled_train_date_ids = tuple(
            int(value) for value in sampled_train_date_ids
        )
        self.num_full_train_samples = self.num_train_samples
        self.num_sampled_train_samples = len(sampled_train_date_ids)
        self.effective_train_ratio = (
            self.num_sampled_train_samples / self.num_full_train_samples
        )

        selected_ids = [int(value) for value in sampled_train_date_ids]
        raw_counts = Counter(self.labels_by_date_id[value] for value in selected_ids)
        self.raw_class_counts = {label: raw_counts.get(label, 0) for label in range(6)}
        grouped_counts = Counter(
            self.grouped_labels_by_date_id[value] for value in selected_ids
        )
        self.class_counts = {
            label: grouped_counts.get(label, 0) for label in range(self.num_classes)
        }
        print(
            "Uniform flare training subset: "
            f"requested_ratio={self.train_ratio:g}, "
            f"full_train={self.num_full_train_samples}, "
            f"selected={self.num_sampled_train_samples}, "
            f"effective_ratio={self.effective_train_ratio:.6f}, "
            f"selected counts={self.class_counts}"
        )


# Compatibility spelling matching the existing dataset_uni module convention.
FlareDataset_split = FlareDatasetSplit


__all__ = [
    "FlareDatasetSplit",
    "FlareDataset_split",
    "evenly_spaced_train_positions",
    "validate_train_ratio",
]
