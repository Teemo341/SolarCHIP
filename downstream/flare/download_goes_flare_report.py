#!/usr/bin/env python3
"""Download the NOAA/NCEI science-quality GOES XRS Flare Report.

The official mission-length CSV is preferred.  Some networks cannot reach
``data.ngdc.noaa.gov`` directly; in that case the script can retrieve the
official yearly CSV payloads through Jina Reader, strip its text preamble,
validate every CSV row, and merge the yearly files locally.  Both transports
write the retained catalog to the same stable filename.  Yearly payloads are
temporary merge inputs and are removed after a successful merge; their source
URLs and hashes remain in the manifest for provenance.  The manifest records
which transport was used, so a proxy-fetched file is never presented as a
byte-for-byte download from NOAA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OFFICIAL_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/multi/l2/data/xrsf-l2-flrpt_science/csv/"
)
JINA_PREFIX = "https://r.jina.ai/http://"
MISSION_PATTERN = re.compile(
    r"sci_xrsf-l2-flrpt_geo_s(?P<start>\d{8})_e(?P<end>\d{8})_v[\d-]+\.csv"
)
YEAR_PATTERN = re.compile(
    r"sci_xrsf-l2-flrpt_geo_y(?P<year>\d{4})_v(?P<version>[\d-]+)\.csv"
)
METADATA_FILENAME = "sci_xrsf-l2-flrpt_geo_metadata.json"
CATALOG_FILENAME = "goes_xrs_flare_report.csv"
MANIFEST_FILENAME = "download_manifest.json"
LEGACY_MERGED_PATTERN = re.compile(
    r"goes_xrs_flare_report_\d{4}_\d{4}_merged\.csv"
)
JINA_MARKER = b"Markdown Content:\n"
EXPECTED_COLUMNS = [
    "time",
    "start_time",
    "end_time",
    "flare_id",
    "xrsb_irrad",
    "flare_class",
    "xrsb_irrad_source",
    "background_irrad",
    "integrated_irrad_peak",
    "integrated_irrad_end",
    "flare_loc_swpc_hgs_lon",
    "flare_loc_swpc_hgs_lat",
    "flare_loc_swpc_source",
    "flare_loc_xrs_hgs_lon",
    "flare_loc_xrs_hgs_lat",
    "flare_loc_xrs_hpc_x",
    "flare_loc_xrs_hpc_y",
    "flare_loc_xrs_source",
    "sequential_flare_num",
    "event_id_swpc",
    "peak_saturated",
    "active_region",
]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "noaa_goes_xrs"


class DownloadError(RuntimeError):
    """Raised when discovery, transport, or payload validation fails."""


class BundleCommitError(RuntimeError):
    """Raised when validated files cannot be installed as one local bundle."""


@dataclass(frozen=True)
class CsvValidation:
    """Validated row count and peak-time bounds for one catalog payload."""

    rows: int
    first_peak_date: str
    last_peak_date: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_bytes(url: str, timeout: float, retries: int = 3) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "SolarCHIP-flare-catalog-downloader/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            last_error = error
            if error.code != 429 or attempt == retries - 1:
                break
            retry_after = error.headers.get("Retry-After", "10")
            try:
                wait_seconds = max(1.0, float(retry_after))
            except ValueError:
                wait_seconds = 10.0
            time.sleep(min(wait_seconds, 30.0))
        except (TimeoutError, URLError, OSError) as error:
            last_error = error
            if attempt == retries - 1:
                break
            time.sleep(2.0 * (attempt + 1))
    raise DownloadError(f"Failed to fetch {url}: {last_error}")


def _jina_url(official_url: str) -> str:
    without_scheme = re.sub(r"^https?://", "", official_url)
    return JINA_PREFIX + without_scheme


def _strip_jina_preamble(payload: bytes, source_url: str) -> bytes:
    marker_index = payload.find(JINA_MARKER)
    if marker_index < 0:
        preview = payload[:200].decode("utf-8", errors="replace")
        raise DownloadError(
            f"Jina response for {source_url} has no content marker: {preview!r}"
        )
    content = payload[marker_index + len(JINA_MARKER) :]
    return content.lstrip(b"\r\n")


def fetch_payload(official_url: str, transport: str, timeout: float) -> bytes:
    if transport == "direct":
        return _request_bytes(official_url, timeout=timeout)
    if transport == "jina":
        wrapped = _request_bytes(_jina_url(official_url), timeout=timeout)
        return _strip_jina_preamble(wrapped, official_url)
    raise ValueError(f"Unsupported transport: {transport}")


def discover_catalog_files(
    transport: str, timeout: float
) -> tuple[str, dict[int, str]]:
    listing = fetch_payload(OFFICIAL_BASE_URL, transport, timeout).decode(
        "utf-8", errors="strict"
    )
    mission_matches = sorted(
        {match.group(0) for match in MISSION_PATTERN.finditer(listing)}
    )
    if not mission_matches:
        raise DownloadError("No mission-length flare-report CSV found in NOAA listing")

    yearly_candidates: dict[int, list[tuple[tuple[int, ...], str]]] = {}
    for match in YEAR_PATTERN.finditer(listing):
        year = int(match.group("year"))
        version = tuple(int(part) for part in match.group("version").split("-"))
        yearly_candidates.setdefault(year, []).append((version, match.group(0)))
    yearly_files = {
        year: max(candidates, key=lambda item: item[0])[1]
        for year, candidates in yearly_candidates.items()
    }
    return mission_matches[-1], yearly_files


def validate_csv_payload(payload: bytes, source_name: str) -> CsvValidation:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DownloadError(f"{source_name} is not valid UTF-8 CSV") from error

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as error:
        raise DownloadError(f"{source_name} is empty") from error
    if header != EXPECTED_COLUMNS:
        raise DownloadError(
            f"Unexpected columns in {source_name}: {header}; expected {EXPECTED_COLUMNS}"
        )

    row_count = 0
    first_peak_date: str | None = None
    last_peak_date: str | None = None
    for line_number, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != len(EXPECTED_COLUMNS):
            raise DownloadError(
                f"{source_name}:{line_number} has {len(row)} columns; "
                f"expected {len(EXPECTED_COLUMNS)}"
            )
        try:
            peak_date = datetime.fromisoformat(row[0].strip()).date().isoformat()
        except ValueError as error:
            raise DownloadError(
                f"{source_name}:{line_number} has invalid peak time {row[0]!r}"
            ) from error
        if first_peak_date is None:
            first_peak_date = peak_date
        if last_peak_date is not None and peak_date < last_peak_date:
            raise DownloadError(
                f"{source_name}:{line_number} is not ordered by peak time"
            )
        last_peak_date = peak_date
        row_count += 1
    if row_count == 0:
        raise DownloadError(f"{source_name} contains no flare rows")
    assert first_peak_date is not None and last_peak_date is not None
    return CsvValidation(row_count, first_peak_date, last_peak_date)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def merge_yearly_csvs(paths: Iterable[Path], output_path: Path) -> int:
    ordered_paths = list(paths)
    if not ordered_paths:
        raise DownloadError("No yearly CSV files were selected for merging")

    rows_written = 0
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(EXPECTED_COLUMNS)
    for path in ordered_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPECTED_COLUMNS:
                raise DownloadError(f"Unexpected columns while merging {path}")
            for row in reader:
                writer.writerow([row[column] for column in EXPECTED_COLUMNS])
                rows_written += 1
    atomic_write(output_path, output.getvalue().encode("utf-8"))
    return rows_written


def _parse_metadata(payload: bytes) -> dict:
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DownloadError("NOAA metadata payload is not valid JSON") from error
    global_attributes = metadata.get("global_attributes", {})
    for required_key in ("time_coverage_start", "time_coverage_end"):
        if required_key not in global_attributes:
            raise DownloadError(f"NOAA metadata is missing {required_key}")
    return metadata


def find_legacy_download_artifacts(
    output_dir: Path, *, include_yearly_cache: bool
) -> list[Path]:
    """Return downloader-owned legacy paths that may be removed after commit.

    The old ``raw_yearly`` directory is accepted only when every child is a
    regular NOAA yearly catalog.  Unexpected content aborts the update instead
    of being deleted.
    """

    legacy_paths: list[Path] = []
    if include_yearly_cache:
        raw_dir = output_dir / "raw_yearly"
        if raw_dir.exists() or raw_dir.is_symlink():
            if raw_dir.is_symlink() or not raw_dir.is_dir():
                raise BundleCommitError(
                    f"Refusing to remove unexpected yearly-cache path: {raw_dir}"
                )
            unexpected_children = [
                child
                for child in raw_dir.iterdir()
                if child.is_symlink()
                or not child.is_file()
                or YEAR_PATTERN.fullmatch(child.name) is None
            ]
            if unexpected_children:
                rendered = ", ".join(str(path) for path in unexpected_children)
                raise BundleCommitError(
                    "Refusing to remove yearly cache containing unexpected paths: "
                    f"{rendered}"
                )
            legacy_paths.append(raw_dir)

    for child in output_dir.iterdir():
        if child.name in {CATALOG_FILENAME, METADATA_FILENAME, MANIFEST_FILENAME}:
            continue
        if not (
            LEGACY_MERGED_PATTERN.fullmatch(child.name)
            or MISSION_PATTERN.fullmatch(child.name)
        ):
            continue
        if child.is_symlink() or not child.is_file():
            raise BundleCommitError(
                f"Refusing to remove unexpected legacy path: {child}"
            )
        legacy_paths.append(child)
    return legacy_paths


def remove_managed_appledouble_files(output_dir: Path) -> None:
    """Remove macOS/NAS ``._*`` sidecars for downloader-owned artifacts only."""

    stable_names = {CATALOG_FILENAME, METADATA_FILENAME, MANIFEST_FILENAME}
    for child in output_dir.iterdir():
        if not child.name.startswith("._"):
            continue
        represented_name = child.name[2:]
        is_managed = (
            represented_name in stable_names
            or represented_name == "raw_yearly"
            or LEGACY_MERGED_PATTERN.fullmatch(represented_name) is not None
            or MISSION_PATTERN.fullmatch(represented_name) is not None
        )
        if not is_managed:
            continue
        if child.is_symlink() or not child.is_file():
            raise BundleCommitError(
                f"Refusing to remove unexpected AppleDouble path: {child}"
            )
        try:
            child.unlink()
        except OSError as error:
            raise BundleCommitError(
                f"Cannot remove downloader AppleDouble sidecar {child}: {error}"
            ) from error


def commit_download_bundle(
    staging_dir: Path,
    output_dir: Path,
    legacy_paths: Iterable[Path],
) -> None:
    """Install a validated bundle and roll back ordinary commit failures."""

    filenames = (METADATA_FILENAME, CATALOG_FILENAME, MANIFEST_FILENAME)
    staged_paths = [staging_dir / filename for filename in filenames]
    missing = [path for path in staged_paths if not path.is_file()]
    if missing:
        raise BundleCommitError(f"Staged download bundle is incomplete: {missing}")

    final_paths = [output_dir / filename for filename in filenames]
    legacy_list = list(legacy_paths)
    for path in [*final_paths, *legacy_list]:
        if not path.exists() and not path.is_symlink():
            continue
        if path in legacy_list and path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise BundleCommitError(f"Refusing to replace unexpected output path: {path}")

    rollback_dir = staging_dir / ".rollback"
    rollback_dir.mkdir()
    backups: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        backup_candidates = [
            path
            for path in [*final_paths, *legacy_list]
            if path.exists() or path.is_symlink()
        ]
        for index, original_path in enumerate(backup_candidates):
            backup_path = rollback_dir / f"{index:03d}_{original_path.name}"
            os.replace(original_path, backup_path)
            backups.append((original_path, backup_path))

        for staged_path, final_path in zip(staged_paths, final_paths):
            os.replace(staged_path, final_path)
            installed.append((final_path, staged_path))
    except BaseException as commit_error:
        rollback_errors: list[str] = []
        for final_path, staged_path in reversed(installed):
            try:
                os.replace(final_path, staged_path)
            except OSError as error:
                rollback_errors.append(f"remove {final_path}: {error}")
        for original_path, backup_path in reversed(backups):
            try:
                os.replace(backup_path, original_path)
            except OSError as error:
                rollback_errors.append(f"restore {original_path}: {error}")
        if rollback_errors:
            recovery_dir = output_dir / f".goes_xrs_recovery_{time.time_ns()}"
            try:
                os.replace(staging_dir, recovery_dir)
                recovery_note = f"; recovery files preserved at {recovery_dir}"
            except OSError as recovery_error:
                recovery_note = f"; could not preserve recovery files: {recovery_error}"
            raise BundleCommitError(
                "Download bundle commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
                + recovery_note
            ) from commit_error
        if isinstance(commit_error, (KeyboardInterrupt, SystemExit)):
            raise
        raise BundleCommitError(
            f"Download bundle commit failed and was rolled back: {commit_error}"
        ) from commit_error


def _stage_download_bundle(
    output_dir: Path,
    start_year: int,
    end_year: int | None,
    transport: str,
    timeout: float,
) -> tuple[Path, dict]:
    mission_filename, yearly_files = discover_catalog_files(transport, timeout)

    metadata_url = OFFICIAL_BASE_URL + METADATA_FILENAME
    metadata_payload = fetch_payload(metadata_url, transport, timeout)
    metadata = _parse_metadata(metadata_payload)
    metadata_path = output_dir / METADATA_FILENAME
    atomic_write(metadata_path, metadata_payload)

    coverage = metadata["global_attributes"]
    source_coverage_start = str(coverage["time_coverage_start"])
    source_coverage_end = str(coverage["time_coverage_end"])
    mission_match = MISSION_PATTERN.fullmatch(mission_filename)
    if mission_match is None:
        raise DownloadError(f"Cannot parse mission filename {mission_filename}")
    if (
        mission_match.group("start") != source_coverage_start
        or mission_match.group("end") != source_coverage_end
    ):
        raise DownloadError(
            "NOAA listing and metadata changed during download: "
            f"mission={mission_match.group('start')}..{mission_match.group('end')}, "
            f"metadata={source_coverage_start}..{source_coverage_end}"
        )
    coverage_end_year = int(str(coverage["time_coverage_end"])[:4])
    selected_end_year = coverage_end_year if end_year is None else end_year
    if selected_end_year < start_year:
        raise DownloadError("end_year must be greater than or equal to start_year")

    files_manifest: list[dict] = [
        {
            "kind": "metadata",
            "local_path": metadata_path.name,
            "source_url": metadata_url,
            "sha256": sha256_bytes(metadata_payload),
            "bytes": len(metadata_payload),
            "official_payload_verbatim": transport == "direct",
        }
    ]
    source_yearly_catalogs: list[dict] = []
    if transport == "direct":
        source_url = OFFICIAL_BASE_URL + mission_filename
        payload = fetch_payload(source_url, transport, timeout)
        validation = validate_csv_payload(payload, mission_filename)
        if (
            validation.first_peak_date.replace("-", "") != source_coverage_start
            or validation.last_peak_date.replace("-", "") != source_coverage_end
        ):
            raise DownloadError(
                "Mission catalog event bounds disagree with NOAA metadata: "
                f"catalog={validation.first_peak_date}..{validation.last_peak_date}, "
                f"metadata={source_coverage_start}..{source_coverage_end}"
            )
        catalog_path = output_dir / CATALOG_FILENAME
        atomic_write(catalog_path, payload)
        files_manifest.append(
            {
                "kind": "catalog",
                "local_path": catalog_path.name,
                "source_url": source_url,
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "rows": validation.rows,
                "first_peak_date": validation.first_peak_date,
                "last_peak_date": validation.last_peak_date,
                "official_payload_verbatim": True,
            }
        )
        catalog_kind = "official_mission_length"
        catalog_coverage_start = source_coverage_start
        catalog_coverage_end = source_coverage_end
    else:
        selected_validations: list[CsvValidation] = []
        catalog_path = output_dir / CATALOG_FILENAME
        with tempfile.TemporaryDirectory(
            dir=output_dir, prefix=".goes_xrs_yearly_"
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            selected_paths: list[Path] = []
            for year in range(start_year, selected_end_year + 1):
                filename = yearly_files.get(year)
                if filename is None:
                    raise DownloadError(f"NOAA listing has no yearly CSV for {year}")
                source_url = OFFICIAL_BASE_URL + filename
                payload = fetch_payload(source_url, transport, timeout)
                validation = validate_csv_payload(payload, filename)
                local_path = staging_dir / filename
                atomic_write(local_path, payload)
                selected_paths.append(local_path)
                selected_validations.append(validation)
                source_yearly_catalogs.append(
                    {
                        "source_filename": filename,
                        "source_url": source_url,
                        "sha256": sha256_bytes(payload),
                        "bytes": len(payload),
                        "rows": validation.rows,
                        "first_peak_date": validation.first_peak_date,
                        "last_peak_date": validation.last_peak_date,
                        "official_payload_verbatim": False,
                        "retained_locally": False,
                    }
                )
                # Keep within the public reader's request-rate limit.
                time.sleep(2.5)

            if selected_end_year == coverage_end_year:
                last_peak_date = selected_validations[-1].last_peak_date.replace(
                    "-", ""
                )
                if last_peak_date != source_coverage_end:
                    raise DownloadError(
                        "Current-year catalog and metadata changed during download: "
                        f"catalog ends {last_peak_date}, metadata ends "
                        f"{source_coverage_end}"
                    )
            merged_rows = merge_yearly_csvs(selected_paths, catalog_path)

        files_manifest.append(
            {
                "kind": "catalog",
                "local_path": catalog_path.name,
                "source_url": None,
                "sha256": sha256_file(catalog_path),
                "bytes": catalog_path.stat().st_size,
                "rows": merged_rows,
                "first_peak_date": selected_validations[0].first_peak_date,
                "last_peak_date": selected_validations[-1].last_peak_date,
                "official_payload_verbatim": False,
                "derived_from_source_filenames": [
                    record["source_filename"] for record in source_yearly_catalogs
                ],
            }
        )
        catalog_kind = "locally_merged_official_yearly_payloads"
        catalog_coverage_start = max(source_coverage_start, f"{start_year:04d}0101")
        catalog_coverage_end = min(source_coverage_end, f"{selected_end_year:04d}1231")

    manifest = {
        "schema_version": 3,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_base_url": OFFICIAL_BASE_URL,
        "official_mission_filename_at_discovery": mission_filename,
        "transport": transport,
        "catalog_kind": catalog_kind,
        "catalog_path": str(catalog_path.relative_to(output_dir)),
        "metadata_path": metadata_path.name,
        "requested_start_year": start_year,
        "requested_end_year": end_year,
        "year_selection_applied": transport == "jina",
        "source_time_coverage_start": source_coverage_start,
        "source_time_coverage_end": source_coverage_end,
        "catalog_time_coverage_start": catalog_coverage_start,
        "catalog_time_coverage_end": catalog_coverage_end,
        "algorithm_version": coverage.get("flrpt_algorithm_version"),
        "license": coverage.get("license"),
        "transport_note": (
            "Direct NOAA bytes."
            if transport == "direct"
            else (
                "NOAA yearly CSV text was fetched through Jina Reader because the "
                "runtime could not connect to data.ngdc.noaa.gov. The reader preamble "
                "was removed and every row was schema-validated. Intermediate yearly "
                "files were deleted after merging; their source URLs and hashes are "
                "recorded in source_yearly_catalogs. The merged catalog is not claimed "
                "to be a byte-for-byte NOAA mission-file download."
            )
        ),
        "files": files_manifest,
        "source_yearly_catalogs": source_yearly_catalogs,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return catalog_path, manifest


def _download_with_transport(
    output_dir: Path,
    start_year: int,
    end_year: int | None,
    transport: str,
    timeout: float,
) -> tuple[Path, dict]:
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".goes_xrs_bundle_"
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        _, manifest = _stage_download_bundle(
            staging_dir, start_year, end_year, transport, timeout
        )
        legacy_paths = find_legacy_download_artifacts(
            output_dir, include_yearly_cache=True
        )
        commit_download_bundle(staging_dir, output_dir, legacy_paths)
        remove_managed_appledouble_files(output_dir)
    return output_dir / CATALOG_FILENAME, manifest


def download_catalog(
    output_dir: Path,
    start_year: int = 2010,
    end_year: int | None = None,
    transport: str = "auto",
    timeout: float = 30.0,
) -> tuple[Path, dict]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if transport in {"direct", "jina"}:
        return _download_with_transport(
            output_dir, start_year, end_year, transport, timeout
        )

    try:
        return _download_with_transport(
            output_dir, start_year, end_year, "direct", timeout
        )
    except DownloadError as direct_error:
        print(f"Direct NOAA download failed: {direct_error}")
        print("Falling back to validated yearly CSV payloads through Jina Reader.")
        return _download_with_transport(
            output_dir, start_year, end_year, "jina", timeout
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Download directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="First yearly file for Jina fallback; direct always keeps the full mission",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Last yearly file for Jina fallback; defaults to metadata coverage year. "
            "Direct always keeps the full mission"
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "direct", "jina"),
        default="auto",
        help="Use NOAA directly, Jina text transport, or automatic fallback",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path, manifest = download_catalog(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        transport=args.transport,
        timeout=args.timeout,
    )
    print(f"Catalog: {catalog_path}")
    print(f"Transport: {manifest['transport']}")
    print(
        "Local catalog coverage: "
        f"{manifest['catalog_time_coverage_start']} through "
        f"{manifest['catalog_time_coverage_end']}"
    )
    print(f"Manifest: {args.output_dir.resolve() / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
