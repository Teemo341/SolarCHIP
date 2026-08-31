"""Tests for the unified flare-checkpoint evaluation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from downstream.flare.data.metrics import binary_true_skill_statistic
from downstream.flare.test import (
    binary_metric_values,
    class_reduction_mappings,
    compute_selected_metrics,
    evaluation_time_metadata,
    resolve_metric_names,
    resolve_resume_paths,
)


class FlareEvaluationMetricTests(unittest.TestCase):
    def test_default_group_metrics_match_hand_calculation(self) -> None:
        confusion = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ]
        )
        metrics, confusions = compute_selected_metrics(
            confusion,
            ("0AB", "C", "M", "X"),
            ("all",),
        )

        # The requested overall ACC is three-class 0AB/C/MX accuracy, so M/X
        # confusions count as correct after reduction.
        self.assertAlmostEqual(metrics["overall_acc"], 0.5)

        self.assertAlmostEqual(metrics["c_plus_pod"], 7 / 9)
        self.assertAlmostEqual(metrics["c_plus_csi"], 7 / 11)
        self.assertAlmostEqual(metrics["c_plus_far"], 2 / 9)
        self.assertAlmostEqual(metrics["c_plus_hss"], 1 / 9)
        self.assertAlmostEqual(metrics["c_plus_tss"], 1 / 9)
        self.assertAlmostEqual(metrics["c_plus_acc"], 2 / 3)

        self.assertAlmostEqual(metrics["m_plus_pod"], 2 / 3)
        self.assertAlmostEqual(metrics["m_plus_csi"], 1 / 2)
        self.assertAlmostEqual(metrics["m_plus_far"], 1 / 3)
        self.assertAlmostEqual(metrics["m_plus_hss"], 1 / 3)
        self.assertAlmostEqual(metrics["m_plus_tss"], 1 / 3)
        self.assertAlmostEqual(metrics["m_plus_acc"], 2 / 3)

        self.assertEqual(confusions["c_plus"], [[1, 2], [2, 7]])
        self.assertEqual(confusions["m_plus"], [[4, 2], [2, 4]])

    def test_metric_selector_expands_base_names_for_both_thresholds(self) -> None:
        self.assertEqual(
            resolve_metric_names(("overall_acc", "pod", "m_plus_far", "pod")),
            ("overall_acc", "c_plus_pod", "m_plus_pod", "m_plus_far"),
        )

    def test_zero_division_policy_matches_existing_flare_metrics(self) -> None:
        values = binary_metric_values(torch.tensor([[5, 0], [0, 0]]))
        self.assertEqual(
            values,
            {
                "pod": 0.0,
                "csi": 0.0,
                "far": 0.0,
                "hss": 0.0,
                "tss": 0.0,
                "acc": 1.0,
            },
        )

    def test_training_tss_marks_missing_true_class_support_invalid(self) -> None:
        for confusion in (
            torch.tensor([[5, 0], [0, 0]]),
            torch.tensor([[0, 0], [0, 5]]),
        ):
            with self.subTest(confusion=confusion.tolist()):
                tss, valid = binary_true_skill_statistic(confusion)
                self.assertEqual(float(tss), 0.0)
                self.assertEqual(float(valid), 0.0)

    def test_six_class_mapping_and_crossing_group_rejection(self) -> None:
        mappings = class_reduction_mappings(("0", "A", "B", "C", "M", "X"))
        self.assertEqual(mappings["overall"], (0, 0, 0, 1, 2, 2))
        self.assertEqual(mappings["c_plus"], (0, 0, 0, 1, 1, 1))
        self.assertEqual(mappings["m_plus"], (0, 0, 0, 0, 1, 1))
        with self.assertRaisesRegex(ValueError, "crosses"):
            class_reduction_mappings(("0ABC", "MX"))


class ResumePathTests(unittest.TestCase):
    def test_run_directory_and_checkpoint_paths_are_both_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "model" / "timestamp"
            (run / "configs").mkdir(parents=True)
            checkpoint_directory = run / "checkpoints"
            checkpoint_directory.mkdir()
            last = checkpoint_directory / "last.ckpt"
            best = checkpoint_directory / "epoch=000003.ckpt"
            last.write_bytes(b"last")
            best.write_bytes(b"best")

            self.assertEqual(resolve_resume_paths(run), (run.resolve(), last.resolve()))
            self.assertEqual(
                resolve_resume_paths(run, best.name),
                (run.resolve(), best.resolve()),
            )
            self.assertEqual(
                resolve_resume_paths(best),
                (run.resolve(), best.resolve()),
            )


class EvaluationTimeMetadataTests(unittest.TestCase):
    def test_requested_and_label_filtered_date_ranges_are_recorded(self) -> None:
        split_config = OmegaConf.create(
            {
                "target": "example.Dataset",
                "params": {
                    "time_interval": [5000, 6000],
                    "time_step": 1,
                },
            }
        )
        dataset = SimpleNamespace(
            exist_idx=[5000, 5951],
            missing_label_date_ids=(5952, 5953, 5954, 5955, 5956),
            num_dropped_for_missing_labels=5,
        )

        metadata = evaluation_time_metadata(split_config, dataset)

        self.assertEqual(metadata["time_interval"], [5000, 6000])
        self.assertEqual(metadata["time_step"], 1)
        self.assertEqual(
            metadata["requested_time_range"],
            {
                "start_date_id": 5000,
                "start_date": "2024-01-08",
                "last_date_id_inclusive": 5999,
                "last_date_inclusive": "2026-10-03",
                "end_date_id_exclusive": 6000,
                "end_date_exclusive": "2026-10-04",
            },
        )
        self.assertEqual(
            metadata["retained_dataset_time_range"],
            {
                "first_date_id": 5000,
                "first_date": "2024-01-08",
                "last_date_id": 5951,
                "last_date": "2026-08-16",
                "dataset_sample_count": 2,
            },
        )
        self.assertEqual(metadata["missing_label_filter"]["dropped_count"], 5)


if __name__ == "__main__":
    unittest.main()
