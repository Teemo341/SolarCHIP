"""SolarCHIP multimodal dataset with a daily GOES flare-class label."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from data.dataset.SolarDataset import multimodal_dataset

from .class_groups import (
    DEFAULT_CLASS_GROUPS,
    build_raw_label_to_group,
    normalize_class_groups,
)


DATASET_EPOCH = date(2010, 5, 1)
DEFAULT_LABEL_PATH = Path(__file__).resolve().parent / "flare_daily_labels.csv"
DEFAULT_MODAL_LIST = (
    "hmi",
    "0094",
    "0131",
    "0171",
    "0193",
    "0211",
    "0304",
    "0335",
    "1600",
    "1700",
    "4500",
)


def load_daily_label_table(label_path: str | Path) -> dict[int, int]:
    """Load and validate the auditable ``date_id -> class label`` table."""

    path = Path(label_path).expanduser()
    labels: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date_id", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label file {path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                date_id = int(row["date_id"])
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid date_id/label at {path}:{row_number}: {row}"
                ) from error
            if date_id in labels:
                raise ValueError(f"Duplicate date_id {date_id} at {path}:{row_number}")
            if label not in range(6):
                raise ValueError(
                    f"Label {label} at {path}:{row_number} is outside 0..5"
                )

            csv_date = (row.get("date") or "").strip()
            if csv_date:
                expected_date = DATASET_EPOCH + timedelta(days=date_id)
                try:
                    parsed_date = date.fromisoformat(csv_date)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid date {csv_date!r} at {path}:{row_number}"
                    ) from error
                if parsed_date != expected_date:
                    raise ValueError(
                        f"date/date_id mismatch at {path}:{row_number}: "
                        f"{parsed_date} != {expected_date}"
                    )
            labels[date_id] = label

    if not labels:
        raise ValueError(f"Label file {path} contains no labels")
    return labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_label_summary(
    label_path: Path,
    summary_path: Path,
    expected_event_time_column: str,
) -> dict:
    """Verify that a label CSV still matches its provenance sidecar."""

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read label summary {summary_path}: {error}"
        ) from error

    actual_hash = sha256_file(label_path)
    if summary.get("output_sha256") != actual_hash:
        raise ValueError(
            f"Label CSV hash does not match {summary_path}: "
            f"{actual_hash} != {summary.get('output_sha256')}"
        )
    if summary.get("dataset_epoch") != DATASET_EPOCH.isoformat():
        raise ValueError(
            f"Label summary dataset_epoch must be {DATASET_EPOCH.isoformat()}, "
            f"got {summary.get('dataset_epoch')!r}"
        )
    if summary.get("event_time_column") != expected_event_time_column:
        raise ValueError(
            "Label day policy mismatch: expected "
            f"{expected_event_time_column!r}, got "
            f"{summary.get('event_time_column')!r} in {summary_path}"
        )
    return summary


class FlareDataset(multimodal_dataset):
    """Extend ``multimodal_dataset`` with a scalar ``torch.long`` label.

    The parent returns ``{modal_name: image_tensor}``.  This subclass preserves
    that contract and adds ``sample['label']``.  Date alignment uses the parent
    dataset's global day ID ``self.exist_idx[position]``, never the compressed
    positional index.
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
    ) -> None:
        if load_imgs:
            raise ValueError(
                "FlareDataset requires load_imgs=False because the parent's "
                "preload branch returns a stacked tensor instead of a modality dict"
            )

        resolved_modal_list = list(
            DEFAULT_MODAL_LIST if modal_list is None else modal_list
        )
        resolved_augment = list(
            (224, 0.5, 90) if torch_augment_type is None else torch_augment_type
        )
        resolved_interval = list((0, 5400) if time_interval is None else time_interval)
        resolved_enhance = list(("log1p",) if enhance_type is None else enhance_type)

        super().__init__(
            modal_list=resolved_modal_list,
            log1p_scale=log1p_scale,
            load_imgs=False,
            torch_augment_type=resolved_augment,
            time_interval=resolved_interval,
            time_step=time_step,
            enhance_type=resolved_enhance,
        )

        self.class_groups = normalize_class_groups(class_groups)
        self.num_classes = len(self.class_groups)
        self.raw_label_to_group = build_raw_label_to_group(self.class_groups)

        self.label_path = str(Path(label_path).expanduser())
        # Keep the auditable CSV labels in their original 0..5 meaning. Only
        # __getitem__ exposes the grouped training target.
        self.labels_by_date_id = load_daily_label_table(self.label_path)
        self.raw_labels_by_date_id = self.labels_by_date_id
        self.grouped_labels_by_date_id = {
            date_id: self.raw_label_to_group[raw_label]
            for date_id, raw_label in self.labels_by_date_id.items()
        }
        resolved_summary_path = (
            Path(self.label_path).with_suffix(".summary.json")
            if label_summary_path is None
            else Path(label_summary_path).expanduser()
        )
        self.label_summary_path = str(resolved_summary_path)
        self.label_summary = (
            load_and_verify_label_summary(
                label_path=Path(self.label_path),
                summary_path=resolved_summary_path,
                expected_event_time_column=expected_event_time_column,
            )
            if verify_label_summary
            else None
        )
        self.return_date_id = return_date_id

        selected_date_ids = [int(value) for value in self.exist_idx]
        self.num_selected_before_label_filter = len(selected_date_ids)
        missing_date_ids = [
            date_id
            for date_id in selected_date_ids
            if date_id not in self.labels_by_date_id
        ]
        self.missing_label_date_ids = tuple(missing_date_ids)
        self.num_dropped_for_missing_labels = len(missing_date_ids)
        if missing_date_ids:
            preview = ", ".join(str(value) for value in missing_date_ids[:10])
            print(
                f"Dropped {len(missing_date_ids)} selected SolarCHIP dates "
                f"without flare labels; first date IDs: {preview}"
            )
            selected_date_ids = [
                date_id
                for date_id in selected_date_ids
                if date_id in self.labels_by_date_id
            ]
            if isinstance(self.exist_idx, np.ndarray):
                self.exist_idx = np.asarray(
                    selected_date_ids, dtype=self.exist_idx.dtype
                )
            else:
                self.exist_idx = selected_date_ids
        if not selected_date_ids:
            raise ValueError(
                "No selected SolarCHIP dates have flare labels after filtering "
                f"with {self.label_path}"
            )

        raw_counts = Counter(
            self.labels_by_date_id[value] for value in selected_date_ids
        )
        self.raw_class_counts = {label: raw_counts.get(label, 0) for label in range(6)}
        counts = Counter(
            self.grouped_labels_by_date_id[value] for value in selected_date_ids
        )
        self.class_counts = {
            label: counts.get(label, 0) for label in range(self.num_classes)
        }
        print(
            f"Flare class groups {list(self.class_groups)}; "
            f"selected-sample counts: {self.class_counts}"
        )

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(position)
        if not isinstance(sample, dict):
            raise TypeError(
                "Expected multimodal_dataset to return a modality dict; "
                f"received {type(sample).__name__}"
            )
        if getattr(self, "_omit_targets_for_statistics", False):
            return sample
        date_id = int(self.exist_idx[position])
        sample["label"] = torch.tensor(
            self.grouped_labels_by_date_id[date_id], dtype=torch.long
        )
        if self.return_date_id:
            sample["date_id"] = torch.tensor(date_id, dtype=torch.long)
        return sample

    def compute_modal_statistics(self, stats_path: str | None = None) -> dict:
        """Reuse parent statistics while excluding label/date_id target keys."""

        previous_value = getattr(self, "_omit_targets_for_statistics", False)
        self._omit_targets_for_statistics = True
        try:
            return super().compute_modal_statistics(stats_path=stats_path)
        finally:
            self._omit_targets_for_statistics = previous_value
