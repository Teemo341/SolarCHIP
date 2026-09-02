"""Tests for deterministic uniform flare train/validation splits."""

from __future__ import annotations

import random
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset.SolarDataset import multimodal_dataset
from downstream.flare.data.dataset import DATASET_EPOCH
from downstream.flare.data.dataset_uni import (
    FlareDatasetUni,
    evenly_spaced_validation_positions,
)
from downstream.flare.test import build_evaluation_loader, evaluation_time_metadata


class UniformPositionTests(unittest.TestCase):
    def test_default_ratio_uses_temporal_bin_centers(self) -> None:
        self.assertEqual(
            evenly_spaced_validation_positions(10).tolist(),
            [2, 7],
        )
        positions = evenly_spaced_validation_positions(11, 0.3)
        self.assertEqual(positions.tolist(), [1, 5, 9])
        self.assertTrue(np.all(np.diff(positions) > 0))

    def test_invalid_ratio_and_too_small_pool_are_rejected(self) -> None:
        for ratio in (0, 1, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                evenly_spaced_validation_positions(10, ratio)
        with self.assertRaises(TypeError):
            evenly_spaced_validation_positions(10, True)
        with self.assertRaises(ValueError):
            evenly_spaced_validation_positions(1)


class FlareDatasetUniTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.label_path = Path(self.temporary_directory.name) / "labels.csv"
        self._write_labels(
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 0,
                7: 3,
                8: 4,
                9: 5,
            }
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_labels(self, labels: dict[int, int]) -> None:
        rows = ["date,date_id,label"]
        for date_id, label in sorted(labels.items()):
            day = DATASET_EPOCH + timedelta(days=date_id)
            rows.append(f"{day.isoformat()},{date_id},{label}")
        self.label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    @staticmethod
    def _fake_parent_init(dataset, **kwargs) -> None:
        dataset.exist_idx = np.arange(10, dtype=np.int64)
        dataset.modal_list = list(kwargs["modal_list"])
        dataset.parent_time_interval = tuple(kwargs["time_interval"])

    @staticmethod
    def _fake_parent_getitem(dataset, position: int) -> dict[str, torch.Tensor]:
        del dataset, position
        return {"hmi": torch.zeros(1, 4, 4)}

    def _dataset(self, split: str, validation_ratio: float = 0.2) -> FlareDatasetUni:
        return FlareDatasetUni(
            split=split,
            validation_ratio=validation_ratio,
            label_path=self.label_path,
            verify_label_summary=False,
            return_date_id=True,
        )

    def test_train_and_validation_are_deterministic_complements(self) -> None:
        with (
            patch.object(multimodal_dataset, "__init__", self._fake_parent_init),
            patch.object(
                multimodal_dataset,
                "__getitem__",
                self._fake_parent_getitem,
            ),
        ):
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            train = self._dataset("train")

            random.seed(101)
            np.random.seed(102)
            torch.manual_seed(103)
            validation = self._dataset("validation")
            validation_again = self._dataset("val")
            test = self._dataset("test")

        self.assertEqual(validation.validation_positions, (2, 7))
        self.assertEqual(list(validation.exist_idx), [2, 7])
        self.assertEqual(list(validation_again.exist_idx), [2, 7])
        self.assertEqual(list(test.exist_idx), [2, 7])
        self.assertEqual(list(train.exist_idx), [0, 1, 3, 4, 5, 6, 8, 9])
        self.assertFalse(set(train.exist_idx) & set(validation.exist_idx))
        self.assertEqual(
            sorted([*train.exist_idx, *validation.exist_idx]),
            list(range(10)),
        )
        self.assertEqual(train.all_exist_idx, tuple(range(10)))
        self.assertEqual(train.num_train_samples, 8)
        self.assertEqual(train.num_validation_samples, 2)
        self.assertEqual(train.parent_time_interval, (0, 10))

        # Counts must describe the selected partition, not the complete pool.
        self.assertEqual(
            validation.raw_class_counts, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
        )
        self.assertEqual(validation.class_counts, {0: 1, 1: 1, 2: 0, 3: 0})
        self.assertEqual(train.class_counts, {0: 3, 1: 1, 2: 2, 3: 2})

    def test_missing_labels_are_filtered_before_uniform_split(self) -> None:
        self._write_labels({0: 0, 1: 1, 3: 3, 4: 4, 6: 0, 7: 3, 8: 4, 9: 5})
        with patch.object(multimodal_dataset, "__init__", self._fake_parent_init):
            validation = self._dataset("validation", validation_ratio=0.25)

        self.assertEqual(validation.all_exist_idx, (0, 1, 3, 4, 6, 7, 8, 9))
        self.assertEqual(validation.validation_positions, (2, 6))
        self.assertEqual(list(validation.exist_idx), [3, 8])
        self.assertEqual(validation.missing_label_date_ids, (2, 5))

    def test_default_collate_retains_labels_and_date_ids(self) -> None:
        with (
            patch.object(multimodal_dataset, "__init__", self._fake_parent_init),
            patch.object(
                multimodal_dataset,
                "__getitem__",
                self._fake_parent_getitem,
            ),
        ):
            dataset = self._dataset("validation")
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

        self.assertEqual(batch["label"].dtype, torch.long)
        self.assertEqual(batch["label"].tolist(), [0, 1])
        self.assertEqual(batch["date_id"].tolist(), [2, 7])

    def test_flare_test_loader_reconstructs_validation_partition(self) -> None:
        config = OmegaConf.create(
            {
                "data": {
                    "params": {
                        "batch_size": 2,
                        "num_workers": 0,
                        "wrap": False,
                        "validation": {
                            "target": (
                                "downstream.flare.data.dataset_uni." "FlareDatasetUni"
                            ),
                            "params": {
                                "split": "validation",
                                "validation_ratio": 0.2,
                                "modal_list": ["hmi"],
                                "time_step": 1,
                                "label_path": str(self.label_path),
                                "verify_label_summary": False,
                                "return_date_id": True,
                            },
                        },
                    }
                }
            }
        )
        config.data.params.test = OmegaConf.create(
            OmegaConf.to_container(
                config.data.params.validation,
                resolve=True,
            )
        )
        config.data.params.test.params.split = "test"
        with (
            patch.object(multimodal_dataset, "__init__", self._fake_parent_init),
            patch.object(
                multimodal_dataset,
                "__getitem__",
                self._fake_parent_getitem,
            ),
        ):
            loader, dataset, split_config = build_evaluation_loader(
                config=config,
                split="validation",
                batch_size=None,
                num_workers=None,
                time_interval=None,
                time_step=None,
                device=torch.device("cpu"),
                seed=999,
            )
            batch = next(iter(loader))
            metadata = evaluation_time_metadata(split_config, dataset)
            test_loader, test_dataset, _ = build_evaluation_loader(
                config=config,
                split="test",
                batch_size=None,
                num_workers=None,
                time_interval=None,
                time_step=None,
                device=torch.device("cpu"),
                seed=111,
            )
            test_batch = next(iter(test_loader))

        self.assertEqual(list(dataset.exist_idx), [2, 7])
        self.assertEqual(batch["date_id"].tolist(), [2, 7])
        self.assertEqual(list(test_dataset.exist_idx), [2, 7])
        self.assertEqual(test_batch["date_id"].tolist(), [2, 7])
        self.assertEqual(metadata["time_interval"], [0, 10])
        self.assertEqual(
            metadata["retained_dataset_time_range"]["dataset_sample_count"], 2
        )

    def test_invalid_split_is_rejected_before_parent_initialization(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset("random")


if __name__ == "__main__":
    unittest.main()
