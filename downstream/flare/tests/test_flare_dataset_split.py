"""Tests for deterministic ratio subsets of the flare training split."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from data.dataset.SolarDataset import multimodal_dataset
from downstream.flare.data.dataset import DATASET_EPOCH
from downstream.flare.data.dataset_split import (
    FlareDatasetSplit,
    evenly_spaced_train_positions,
)
from downstream.flare.data.dataset_uni import FlareDatasetUni


class EvenlySpacedTrainPositionTests(unittest.TestCase):
    def test_temporal_bin_centers_and_full_ratio(self) -> None:
        self.assertEqual(evenly_spaced_train_positions(8, 0.25).tolist(), [2, 6])
        self.assertEqual(
            evenly_spaced_train_positions(8, 0.5).tolist(),
            [1, 3, 5, 7],
        )
        self.assertEqual(evenly_spaced_train_positions(8, 1.0).tolist(), list(range(8)))

    def test_requested_count_uses_round_half_up(self) -> None:
        self.assertEqual(len(evenly_spaced_train_positions(5, 0.5)), 3)
        self.assertEqual(len(evenly_spaced_train_positions(5, 0.1)), 1)

    def test_all_experiment_ratios_have_the_expected_real_dataset_sizes(self) -> None:
        expected_sizes = [473, 945, 1418, 1890, 2363, 2836, 3308, 3781, 4253]
        for percentage, expected_size in zip(
            range(10, 100, 10), expected_sizes, strict=True
        ):
            with self.subTest(percentage=percentage):
                positions = evenly_spaced_train_positions(
                    4726, percentage / 100
                )
                self.assertEqual(len(positions), expected_size)
                self.assertTrue(np.all(np.diff(positions) > 0))

    def test_invalid_inputs_are_rejected(self) -> None:
        for ratio in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                evenly_spaced_train_positions(10, ratio)
        with self.assertRaises(TypeError):
            evenly_spaced_train_positions(10, True)
        with self.assertRaises(ValueError):
            evenly_spaced_train_positions(0, 0.5)


class FlareDatasetSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.label_path = Path(self.temporary_directory.name) / "labels.csv"
        rows = ["date,date_id,label"]
        labels = [0, 1, 2, 3, 4, 5, 0, 3, 4, 5]
        for date_id, label in enumerate(labels):
            day = DATASET_EPOCH + timedelta(days=date_id)
            rows.append(f"{day.isoformat()},{date_id},{label}")
        self.label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _fake_parent_init(dataset, **kwargs) -> None:
        dataset.exist_idx = np.arange(10, dtype=np.int64)
        dataset.modal_list = list(kwargs["modal_list"])

    def _dataset(self, train_ratio: float) -> FlareDatasetSplit:
        return FlareDatasetSplit(
            split="train",
            validation_ratio=0.2,
            train_ratio=train_ratio,
            label_path=self.label_path,
            verify_label_summary=False,
            return_date_id=True,
        )

    def test_subset_is_drawn_from_existing_train_partition(self) -> None:
        with patch.object(multimodal_dataset, "__init__", self._fake_parent_init):
            dataset = self._dataset(0.25)

        self.assertEqual(dataset.full_train_date_ids, (0, 1, 3, 4, 5, 6, 8, 9))
        self.assertEqual(dataset.validation_date_ids, (2, 7))
        self.assertEqual(dataset.train_subset_positions, (2, 6))
        self.assertEqual(list(dataset.exist_idx), [3, 8])
        self.assertEqual(dataset.sampled_train_date_ids, (3, 8))
        self.assertEqual(dataset.train_date_ids, dataset.full_train_date_ids)
        self.assertEqual(dataset.num_full_train_samples, 8)
        self.assertEqual(dataset.num_train_samples, 8)
        self.assertEqual(dataset.num_sampled_train_samples, 2)
        self.assertEqual(
            dataset.num_total_samples,
            dataset.num_train_samples + dataset.num_validation_samples,
        )
        self.assertEqual(dataset.effective_train_ratio, 0.25)
        self.assertEqual(
            dataset.raw_class_counts,
            {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0},
        )
        self.assertEqual(dataset.class_counts, {0: 0, 1: 1, 2: 1, 3: 0})

    def test_full_ratio_preserves_the_original_train_partition(self) -> None:
        with patch.object(multimodal_dataset, "__init__", self._fake_parent_init):
            dataset = self._dataset(1.0)

        self.assertEqual(tuple(dataset.exist_idx), dataset.full_train_date_ids)
        self.assertEqual(
            dataset.num_sampled_train_samples,
            dataset.num_full_train_samples,
        )

    def test_validation_partition_is_unchanged_in_the_base_dataset(self) -> None:
        with patch.object(multimodal_dataset, "__init__", self._fake_parent_init):
            validation = FlareDatasetUni(
                split="validation",
                validation_ratio=0.2,
                label_path=self.label_path,
                verify_label_summary=False,
            )

        self.assertEqual(list(validation.exist_idx), [2, 7])

    def test_split_dataset_rejects_held_out_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports split='train'"):
            FlareDatasetSplit(
                split="validation",
                train_ratio=0.5,
                label_path=self.label_path,
                verify_label_summary=False,
            )


class FlareSplitConfigTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[3]

    def test_every_ratio_config_only_overrides_the_training_dataset(self) -> None:
        config_root = self.project_root / "configs"
        for backbone in ("cnn", "vit"):
            full_path = config_root / "flare" / f"solar_predictor_{backbone}_full.yaml"
            full = OmegaConf.to_container(OmegaConf.load(full_path), resolve=True)
            self.assertIsInstance(full, dict)

            for percentage in range(10, 100, 10):
                with self.subTest(backbone=backbone, percentage=percentage):
                    split_path = (
                        config_root
                        / "flare_split"
                        / f"solar_predictor_{backbone}_ratio{percentage}.yaml"
                    )
                    actual = OmegaConf.to_container(
                        OmegaConf.load(split_path), resolve=True
                    )
                    expected = deepcopy(full)
                    expected["data"]["params"]["train"]["target"] = (
                        "downstream.flare.data.dataset_split.FlareDatasetSplit"
                    )
                    expected["data"]["params"]["train"]["params"][
                        "train_ratio"
                    ] = percentage / 100

                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        actual["data"]["params"]["validation"],
                        full["data"]["params"]["validation"],
                    )


if __name__ == "__main__":
    unittest.main()
