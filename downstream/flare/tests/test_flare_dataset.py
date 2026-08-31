"""Tests for grouped flare labels exposed by FlareDataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from data.dataset.SolarDataset import multimodal_dataset
from downstream.flare.data.class_groups import (
    DEFAULT_CLASS_GROUPS,
    build_raw_label_to_group,
    normalize_class_groups,
)
from downstream.flare.data.dataset import FlareDataset


class ClassGroupTests(unittest.TestCase):
    def test_default_and_custom_group_mappings(self) -> None:
        self.assertEqual(
            normalize_class_groups(),
            DEFAULT_CLASS_GROUPS,
        )
        self.assertEqual(
            build_raw_label_to_group(),
            {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3},
        )
        self.assertEqual(
            normalize_class_groups(["BA0", "C", "M", "X"]),
            DEFAULT_CLASS_GROUPS,
        )
        self.assertEqual(
            build_raw_label_to_group(["0", "AB", "CM", "X"]),
            {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
        )

    def test_invalid_groups_are_rejected_at_construction(self) -> None:
        invalid_values = [
            "0ABCMX",
            ["0AB", "C", "M"],
            ["0AB", "BC", "M", "X"],
            ["0AB", "C", "M", "Y", "X"],
            ["", "0AB", "C", "M", "X"],
            ["0ABCMX"],
            {"0AB", "C", "M", "X"},
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                normalize_class_groups(value)


class FlareDatasetGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.label_path = Path(self.temporary_directory.name) / "labels.csv"
        self.label_path.write_text(
            "date,date_id,label\n"
            "2010-05-01,0,0\n"
            "2010-05-02,1,1\n"
            "2010-05-03,2,2\n"
            "2010-05-04,3,3\n"
            "2010-05-05,4,4\n"
            "2010-05-06,5,5\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _fake_parent_init(dataset, **kwargs) -> None:
        del kwargs
        dataset.exist_idx = list(range(6))
        dataset.modal_list = ["hmi"]

    @staticmethod
    def _fake_parent_getitem(dataset, position: int) -> dict[str, torch.Tensor]:
        del dataset, position
        return {"hmi": torch.zeros(1, 4, 4)}

    def test_dataset_preserves_raw_table_and_returns_grouped_targets(self) -> None:
        with (
            patch.object(multimodal_dataset, "__init__", self._fake_parent_init),
            patch.object(
                multimodal_dataset,
                "__getitem__",
                self._fake_parent_getitem,
            ),
        ):
            dataset = FlareDataset(
                modal_list=["hmi"],
                label_path=self.label_path,
                verify_label_summary=False,
            )
            labels = next(iter(DataLoader(dataset, batch_size=6)))["label"]

        self.assertEqual(dataset.class_groups, DEFAULT_CLASS_GROUPS)
        self.assertEqual(dataset.num_classes, 4)
        self.assertEqual(
            [dataset.labels_by_date_id[index] for index in range(6)],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [dataset.grouped_labels_by_date_id[index] for index in range(6)],
            [0, 0, 0, 1, 2, 3],
        )
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.tolist(), [0, 0, 0, 1, 2, 3])
        self.assertEqual(dataset.raw_class_counts, {index: 1 for index in range(6)})
        self.assertEqual(dataset.class_counts, {0: 3, 1: 1, 2: 1, 3: 1})

    def test_dates_without_flare_labels_are_filtered(self) -> None:
        incomplete_label_path = Path(self.temporary_directory.name) / "incomplete.csv"
        incomplete_label_path.write_text(
            "date,date_id,label\n"
            "2010-05-01,0,0\n"
            "2010-05-02,1,1\n"
            "2010-05-04,3,3\n"
            "2010-05-05,4,4\n",
            encoding="utf-8",
        )
        with (
            patch.object(multimodal_dataset, "__init__", self._fake_parent_init),
            patch.object(
                multimodal_dataset,
                "__getitem__",
                self._fake_parent_getitem,
            ),
        ):
            dataset = FlareDataset(
                modal_list=["hmi"],
                label_path=incomplete_label_path,
                verify_label_summary=False,
            )
            labels = next(iter(DataLoader(dataset, batch_size=4)))["label"]

        self.assertEqual(list(dataset.exist_idx), [0, 1, 3, 4])
        self.assertEqual(dataset.num_selected_before_label_filter, 6)
        self.assertEqual(dataset.missing_label_date_ids, (2, 5))
        self.assertEqual(dataset.num_dropped_for_missing_labels, 2)
        self.assertEqual(labels.tolist(), [0, 0, 1, 2])
        self.assertEqual(
            dataset.raw_class_counts,
            {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0},
        )
        self.assertEqual(dataset.class_counts, {0: 2, 1: 1, 2: 1, 3: 0})


if __name__ == "__main__":
    unittest.main()
