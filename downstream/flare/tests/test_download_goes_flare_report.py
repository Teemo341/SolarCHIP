"""Behavioral tests for the GOES flare-report downloader."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from downstream.flare import download_goes_flare_report as downloader
from downstream.flare.prepare_flare_labels import load_manifest_inputs


MISSION_FILENAME = "sci_xrsf-l2-flrpt_geo_s20240101_e20251231_v1-0-1.csv"
YEARLY_FILENAMES = {
    2024: "sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv",
    2025: "sci_xrsf-l2-flrpt_geo_y2025_v1-0-1.csv",
}


def make_catalog_payload(peak_times: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=downloader.EXPECTED_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for index, peak_time in enumerate(peak_times, start=1):
        row = {column: "" for column in downloader.EXPECTED_COLUMNS}
        row.update(
            {
                "time": peak_time,
                "start_time": peak_time,
                "flare_id": f"test-{index}",
                "xrsb_irrad": "1.0e-6",
                "flare_class": "C1.0",
                "xrsb_irrad_source": "GOES-TEST",
                "sequential_flare_num": "1",
            }
        )
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def make_metadata_payload() -> bytes:
    return json.dumps(
        {
            "global_attributes": {
                "time_coverage_start": "20240101",
                "time_coverage_end": "20251231",
                "flrpt_algorithm_version": "test-version",
                "license": "test-license",
            }
        }
    ).encode("utf-8")


def write_existing_bundle(output_dir: Path) -> dict[str, bytes]:
    catalog_payload = make_catalog_payload(["2023-01-01 00:01:00"])
    metadata_payload = b'{"snapshot": "previous"}'
    catalog_path = output_dir / downloader.CATALOG_FILENAME
    metadata_path = output_dir / downloader.METADATA_FILENAME
    catalog_path.write_bytes(catalog_payload)
    metadata_path.write_bytes(metadata_payload)
    manifest = {
        "schema_version": 3,
        "catalog_path": downloader.CATALOG_FILENAME,
        "metadata_path": downloader.METADATA_FILENAME,
        "files": [
            {
                "kind": "metadata",
                "local_path": downloader.METADATA_FILENAME,
                "sha256": downloader.sha256_bytes(metadata_payload),
                "bytes": len(metadata_payload),
            },
            {
                "kind": "catalog",
                "local_path": downloader.CATALOG_FILENAME,
                "sha256": downloader.sha256_bytes(catalog_payload),
                "bytes": len(catalog_payload),
            },
        ],
    }
    manifest_payload = (json.dumps(manifest) + "\n").encode("utf-8")
    (output_dir / downloader.MANIFEST_FILENAME).write_bytes(manifest_payload)
    return {
        downloader.CATALOG_FILENAME: catalog_payload,
        downloader.METADATA_FILENAME: metadata_payload,
        downloader.MANIFEST_FILENAME: manifest_payload,
    }


class DownloadCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata_payload = make_metadata_payload()
        self.yearly_payloads = {
            YEARLY_FILENAMES[2024]: make_catalog_payload(["2024-01-01 00:01:00"]),
            YEARLY_FILENAMES[2025]: make_catalog_payload(["2025-12-31 23:59:00"]),
        }
        self.mission_payload = make_catalog_payload(
            ["2024-01-01 00:01:00", "2025-12-31 23:59:00"]
        )

    def fetch(self, url: str, transport: str, timeout: float) -> bytes:
        del transport, timeout
        filename = url.rsplit("/", maxsplit=1)[-1]
        if filename == downloader.METADATA_FILENAME:
            return self.metadata_payload
        if filename == MISSION_FILENAME:
            return self.mission_payload
        return self.yearly_payloads[filename]

    def run_download(self, output_dir: Path, transport: str) -> tuple[Path, dict]:
        with (
            mock.patch.object(
                downloader,
                "discover_catalog_files",
                return_value=(MISSION_FILENAME, YEARLY_FILENAMES),
            ),
            mock.patch.object(downloader, "fetch_payload", side_effect=self.fetch),
            mock.patch.object(downloader.time, "sleep"),
        ):
            return downloader.download_catalog(
                output_dir=output_dir,
                start_year=2024,
                end_year=2025,
                transport=transport,
                timeout=1.0,
            )

    def test_direct_and_yearly_transports_use_same_catalog_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            direct_path, direct_manifest = self.run_download(root / "direct", "direct")
            yearly_path, yearly_manifest = self.run_download(root / "yearly", "jina")

            self.assertEqual(direct_path.name, downloader.CATALOG_FILENAME)
            self.assertEqual(yearly_path.name, downloader.CATALOG_FILENAME)
            self.assertEqual(direct_path.name, yearly_path.name)
            self.assertEqual(direct_path.read_bytes(), self.mission_payload)
            self.assertEqual(
                downloader.validate_csv_payload(yearly_path.read_bytes(), yearly_path.name).rows,
                2,
            )
            self.assertFalse(direct_manifest["year_selection_applied"])
            self.assertTrue(yearly_manifest["year_selection_applied"])

    def test_yearly_inputs_are_removed_but_provenance_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            raw_dir = output_dir / "raw_yearly"
            raw_dir.mkdir()
            (raw_dir / YEARLY_FILENAMES[2024]).write_bytes(
                self.yearly_payloads[YEARLY_FILENAMES[2024]]
            )
            legacy_catalog = output_dir / "goes_xrs_flare_report_2010_2024_merged.csv"
            legacy_catalog.write_bytes(b"legacy merged catalog")
            (output_dir / "._raw_yearly").write_bytes(b"AppleDouble")
            (output_dir / f"._{legacy_catalog.name}").write_bytes(b"AppleDouble")
            catalog_path, manifest = self.run_download(output_dir, "jina")

            self.assertTrue(catalog_path.is_file())
            self.assertFalse((output_dir / "raw_yearly").exists())
            self.assertFalse(legacy_catalog.exists())
            self.assertFalse(
                any(path.name.startswith(".goes_xrs_yearly_") for path in output_dir.iterdir())
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    downloader.CATALOG_FILENAME,
                    downloader.METADATA_FILENAME,
                    downloader.MANIFEST_FILENAME,
                },
            )

            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(len(manifest["source_yearly_catalogs"]), 2)
            self.assertTrue(
                all(
                    source["retained_locally"] is False
                    and "local_path" not in source
                    and source["sha256"]
                    for source in manifest["source_yearly_catalogs"]
                )
            )
            self.assertTrue(
                all(
                    (output_dir / record["local_path"]).is_file()
                    for record in manifest["files"]
                )
            )
            loaded_catalog, loaded_metadata, _ = load_manifest_inputs(
                output_dir / "download_manifest.json"
            )
            self.assertEqual(loaded_catalog, catalog_path.resolve())
            self.assertEqual(
                loaded_metadata,
                (output_dir / downloader.METADATA_FILENAME).resolve(),
            )

    def test_direct_download_also_removes_safe_legacy_yearly_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            raw_dir = output_dir / "raw_yearly"
            raw_dir.mkdir()
            (raw_dir / YEARLY_FILENAMES[2024]).write_bytes(
                self.yearly_payloads[YEARLY_FILENAMES[2024]]
            )

            self.run_download(output_dir, "direct")

            self.assertFalse(raw_dir.exists())
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    downloader.CATALOG_FILENAME,
                    downloader.METADATA_FILENAME,
                    downloader.MANIFEST_FILENAME,
                },
            )

    def test_unsafe_legacy_cache_aborts_without_auto_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            raw_dir = output_dir / "raw_yearly"
            raw_dir.mkdir()
            unexpected_path = raw_dir / "user-notes.txt"
            unexpected_path.write_text("keep me", encoding="utf-8")
            seen_transports: list[str] = []

            def tracking_fetch(url: str, transport: str, timeout: float) -> bytes:
                seen_transports.append(transport)
                return self.fetch(url, transport, timeout)

            with (
                mock.patch.object(
                    downloader,
                    "discover_catalog_files",
                    return_value=(MISSION_FILENAME, YEARLY_FILENAMES),
                ),
                mock.patch.object(
                    downloader, "fetch_payload", side_effect=tracking_fetch
                ),
                self.assertRaises(downloader.BundleCommitError),
            ):
                downloader.download_catalog(
                    output_dir=output_dir,
                    start_year=2024,
                    end_year=2025,
                    transport="auto",
                    timeout=1.0,
                )

            self.assertTrue(unexpected_path.is_file())
            self.assertEqual(set(seen_transports), {"direct"})
            self.assertFalse((output_dir / downloader.CATALOG_FILENAME).exists())

    def test_failed_yearly_download_cleans_staging_and_preserves_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            previous_bundle = write_existing_bundle(output_dir)

            def failing_fetch(url: str, transport: str, timeout: float) -> bytes:
                filename = url.rsplit("/", maxsplit=1)[-1]
                if filename == YEARLY_FILENAMES[2025]:
                    return b"invalid yearly payload"
                return self.fetch(url, transport, timeout)

            with (
                mock.patch.object(
                    downloader,
                    "discover_catalog_files",
                    return_value=(MISSION_FILENAME, YEARLY_FILENAMES),
                ),
                mock.patch.object(
                    downloader, "fetch_payload", side_effect=failing_fetch
                ),
                mock.patch.object(downloader.time, "sleep"),
                self.assertRaises(downloader.DownloadError),
            ):
                downloader.download_catalog(
                    output_dir=output_dir,
                    start_year=2024,
                    end_year=2025,
                    transport="jina",
                    timeout=1.0,
                )

            for filename, previous_payload in previous_bundle.items():
                self.assertEqual((output_dir / filename).read_bytes(), previous_payload)
            self.assertFalse(
                any(path.name.startswith(".goes_xrs_yearly_") for path in output_dir.iterdir())
            )
            load_manifest_inputs(output_dir / downloader.MANIFEST_FILENAME)

    def test_commit_failure_rolls_back_the_complete_existing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            previous_bundle = write_existing_bundle(output_dir)
            real_replace = downloader.os.replace
            manifest_install_failed = False
            final_manifest_path = output_dir.resolve() / downloader.MANIFEST_FILENAME
            seen_transports: list[str] = []

            def tracking_fetch(url: str, transport: str, timeout: float) -> bytes:
                seen_transports.append(transport)
                return self.fetch(url, transport, timeout)

            def fail_manifest_install(source: str | Path, destination: str | Path) -> None:
                nonlocal manifest_install_failed
                destination_path = Path(destination)
                if (
                    destination_path == final_manifest_path
                    and not manifest_install_failed
                ):
                    manifest_install_failed = True
                    raise OSError("simulated manifest install failure")
                real_replace(source, destination)

            with (
                mock.patch.object(
                    downloader,
                    "discover_catalog_files",
                    return_value=(MISSION_FILENAME, YEARLY_FILENAMES),
                ),
                mock.patch.object(
                    downloader, "fetch_payload", side_effect=tracking_fetch
                ),
                mock.patch.object(downloader.time, "sleep"),
                mock.patch.object(
                    downloader.os, "replace", side_effect=fail_manifest_install
                ),
                self.assertRaises(downloader.BundleCommitError),
            ):
                downloader.download_catalog(
                    output_dir=output_dir,
                    start_year=2024,
                    end_year=2025,
                    transport="auto",
                    timeout=1.0,
                )

            for filename, previous_payload in previous_bundle.items():
                self.assertEqual((output_dir / filename).read_bytes(), previous_payload)
            load_manifest_inputs(output_dir / downloader.MANIFEST_FILENAME)
            self.assertEqual(set(seen_transports), {"direct"})


if __name__ == "__main__":
    unittest.main()
