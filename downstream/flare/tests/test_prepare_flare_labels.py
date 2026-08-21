from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from downstream.flare.download_goes_flare_report import EXPECTED_COLUMNS
from downstream.flare.prepare_flare_labels import (
    LabelConversionError,
    build_daily_labels,
    load_manifest_inputs,
    parse_flare_class,
    sha256_file,
)


def make_row(
    *,
    peak: str,
    start: str,
    end: str,
    flare_id: str,
    flare_class: str,
    flux: str,
) -> dict[str, str]:
    row = {column: "" for column in EXPECTED_COLUMNS}
    row.update(
        {
            "time": peak,
            "start_time": start,
            "end_time": end,
            "flare_id": flare_id,
            "xrsb_irrad": flux,
            "flare_class": flare_class,
            "xrsb_irrad_source": "GOES-16",
        }
    )
    return row


class PrepareFlareLabelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.catalog = self.root / "catalog.csv"
        self.metadata = self.root / "metadata.json"
        self.output = self.root / "labels.csv"
        self.summary = self.root / "summary.json"
        self.metadata.write_text(
            json.dumps(
                {
                    "global_attributes": {
                        "time_coverage_start": "20240101",
                        "time_coverage_end": "20240103",
                        "flrpt_algorithm_version": "test",
                        "license": "test-only",
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_catalog(self, rows: list[dict[str, str]]) -> None:
        with self.catalog.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def read_labels(self) -> dict[str, dict[str, str]]:
        with self.output.open("r", encoding="utf-8", newline="") as handle:
            return {row["date"]: row for row in csv.DictReader(handle)}

    def test_same_day_uses_highest_letter_and_zero_fills_covered_days(self) -> None:
        rows = [
            make_row(
                peak="2024-01-01 10:10:00",
                start="2024-01-01 10:00:00",
                end="2024-01-01 10:20:00",
                flare_id="202401011000",
                flare_class="B9.9",
                flux="9.9e-7",
            ),
            make_row(
                peak="2024-01-01 12:10:00",
                start="2024-01-01 12:00:00",
                end="2024-01-01 12:20:00",
                flare_id="202401011200",
                flare_class="C1.0",
                flux="1.0e-6",
            ),
        ]
        self.write_catalog(rows)
        summary = build_daily_labels(
            catalog_path=self.catalog,
            metadata_path=self.metadata,
            output_path=self.output,
            summary_path=self.summary,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
        )
        labels = self.read_labels()
        self.assertEqual(labels["2024-01-01"]["label"], "3")
        self.assertEqual(labels["2024-01-01"]["max_flare_class"], "C1.0")
        self.assertEqual(labels["2024-01-01"]["flare_count"], "2")
        self.assertEqual(labels["2024-01-02"]["label"], "0")
        self.assertEqual(labels["2024-01-03"]["label"], "0")
        self.assertEqual(
            summary["daily_label_counts"],
            {"0": 2, "1": 0, "2": 0, "3": 1, "4": 0, "5": 0},
        )

    def test_start_and_peak_day_policies_are_explicit(self) -> None:
        rows = [
            make_row(
                peak="2024-01-02 00:03:00",
                start="2024-01-01 23:55:00",
                end="2024-01-02 00:10:00",
                flare_id="202401012355",
                flare_class="X1.2",
                flux="1.2e-4",
            )
        ]
        self.write_catalog(rows)
        build_daily_labels(
            catalog_path=self.catalog,
            metadata_path=self.metadata,
            output_path=self.output,
            summary_path=self.summary,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            event_time_column="start_time",
        )
        start_labels = self.read_labels()
        self.assertEqual(start_labels["2024-01-01"]["label"], "5")
        self.assertEqual(start_labels["2024-01-02"]["label"], "0")

        build_daily_labels(
            catalog_path=self.catalog,
            metadata_path=self.metadata,
            output_path=self.output,
            summary_path=self.summary,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            event_time_column="time",
        )
        peak_labels = self.read_labels()
        self.assertEqual(peak_labels["2024-01-01"]["label"], "0")
        self.assertEqual(peak_labels["2024-01-02"]["label"], "5")

    def test_exact_duplicate_event_row_is_counted_once(self) -> None:
        event = make_row(
            peak="2024-01-02 10:10:00",
            start="2024-01-02 10:00:00",
            end="2024-01-02 10:20:00",
            flare_id="202401021000",
            flare_class="M2.0",
            flux="2.0e-5",
        )
        self.write_catalog([event, event.copy()])
        summary = build_daily_labels(
            catalog_path=self.catalog,
            metadata_path=self.metadata,
            output_path=self.output,
            summary_path=self.summary,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
        )
        labels = self.read_labels()
        self.assertEqual(labels["2024-01-02"]["flare_count"], "1")
        self.assertEqual(summary["exact_duplicate_rows_skipped"], 1)

    def test_refuses_zero_fill_outside_metadata_coverage(self) -> None:
        self.write_catalog(
            [
                make_row(
                    peak="2024-01-01 10:10:00",
                    start="2024-01-01 10:00:00",
                    end="2024-01-01 10:20:00",
                    flare_id="202401011000",
                    flare_class="C1.0",
                    flux="1.0e-6",
                )
            ]
        )
        with self.assertRaises(LabelConversionError):
            build_daily_labels(
                catalog_path=self.catalog,
                metadata_path=self.metadata,
                output_path=self.output,
                summary_path=self.summary,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 4),
            )

    def test_manifest_local_coverage_prevents_partial_download_false_zeros(
        self,
    ) -> None:
        self.write_catalog(
            [
                make_row(
                    peak="2024-01-02 10:10:00",
                    start="2024-01-02 10:00:00",
                    end="2024-01-02 10:20:00",
                    flare_id="202401021000",
                    flare_class="C1.0",
                    flux="1.0e-6",
                )
            ]
        )
        partial_manifest = {
            "catalog_time_coverage_start": "20240102",
            "catalog_time_coverage_end": "20240103",
            "transport": "test",
            "schema_version": 2,
        }
        summary = build_daily_labels(
            catalog_path=self.catalog,
            metadata_path=self.metadata,
            output_path=self.output,
            summary_path=self.summary,
            manifest=partial_manifest,
        )
        self.assertEqual(summary["coverage_start"], "2024-01-02")
        self.assertEqual(summary["coverage_end"], "2024-01-03")
        self.assertNotIn("2024-01-01", self.read_labels())

        with self.assertRaises(LabelConversionError):
            build_daily_labels(
                catalog_path=self.catalog,
                metadata_path=self.metadata,
                output_path=self.output,
                summary_path=self.summary,
                start_date=date(2024, 1, 1),
                manifest=partial_manifest,
            )

    def test_manifest_hashes_are_checked_before_conversion(self) -> None:
        self.write_catalog(
            [
                make_row(
                    peak="2024-01-02 10:10:00",
                    start="2024-01-02 10:00:00",
                    end="2024-01-02 10:20:00",
                    flare_id="202401021000",
                    flare_class="C1.0",
                    flux="1.0e-6",
                )
            ]
        )
        manifest_path = self.root / "download_manifest.json"
        manifest = {
            "catalog_path": self.catalog.name,
            "metadata_path": self.metadata.name,
            "files": [
                {
                    "local_path": self.catalog.name,
                    "sha256": sha256_file(self.catalog),
                    "bytes": self.catalog.stat().st_size,
                },
                {
                    "local_path": self.metadata.name,
                    "sha256": sha256_file(self.metadata),
                    "bytes": self.metadata.stat().st_size,
                },
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        catalog_path, metadata_path, _ = load_manifest_inputs(manifest_path)
        self.assertEqual(catalog_path, self.catalog.resolve())
        self.assertEqual(metadata_path, self.metadata.resolve())

        self.metadata.write_text("{}", encoding="utf-8")
        with self.assertRaises(LabelConversionError):
            load_manifest_inputs(manifest_path)

    def test_class_parser_keeps_complete_mapping(self) -> None:
        self.assertEqual(parse_flare_class("A1.2", 2).label, 1)
        self.assertEqual(parse_flare_class("B9.9", 2).label, 2)
        self.assertEqual(parse_flare_class("C1.0", 2).label, 3)
        self.assertEqual(parse_flare_class("M5.0", 2).label, 4)
        self.assertEqual(parse_flare_class("X2.0", 2).label, 5)


if __name__ == "__main__":
    unittest.main()
