#!/usr/bin/env python3
"""Convert a NOAA GOES XRS flare-event catalog into daily SolarCHIP labels.

For a SolarCHIP sample assigned calendar date D, the default target is the
highest GOES letter class among events whose UTC ``start_time`` falls in
[D, D + 1 day).
Use ``--event-time-column time`` to group by the catalog's peak-time index
instead.  Labels are 0=no catalogued flare, 1=A, 2=B, 3=C, 4=M, and 5=X.
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
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "noaa_goes_xrs" / "download_manifest.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "flare_daily_labels.csv"
DEFAULT_SUMMARY = DEFAULT_DATA_DIR / "flare_daily_labels.summary.json"
DATASET_EPOCH = date(2010, 5, 1)
CLASS_TO_LABEL = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
LABEL_TO_NAME = {0: "none", 1: "A", 2: "B", 3: "C", 4: "M", 5: "X"}
CLASS_BASE_FLUX = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}
CLASS_PATTERN = re.compile(r"^\s*([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)?\s*$", re.I)
REQUIRED_CATALOG_COLUMNS = {
    "time",
    "start_time",
    "end_time",
    "flare_id",
    "xrsb_irrad",
    "flare_class",
}
OUTPUT_COLUMNS = [
    "date",
    "date_id",
    "label",
    "label_name",
    "max_flare_class",
    "max_xrsb_irrad_w_m2",
    "flare_count",
]


class LabelConversionError(RuntimeError):
    """Raised when a catalog cannot be converted without ambiguous labels."""


@dataclass(frozen=True)
class ParsedClass:
    text: str
    letter: str
    label: int
    coefficient: float
    equivalent_flux: float


@dataclass
class DailyAggregate:
    label: int = 0
    max_flare_class: str = ""
    max_flux: float | None = None
    flare_count: int = 0
    _max_rank: tuple[int, float] = (0, 0.0)

    def add(self, flare_class: ParsedClass, measured_flux: float | None) -> None:
        self.flare_count += 1
        comparison_flux = (
            measured_flux if measured_flux is not None else flare_class.equivalent_flux
        )
        rank = (flare_class.label, comparison_flux)
        if rank > self._max_rank:
            self.label = flare_class.label
            self.max_flare_class = flare_class.text
            self.max_flux = measured_flux
            self._max_rank = rank


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid ISO date {value!r}") from error


def parse_timestamp(value: str, row_number: int, column: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise LabelConversionError(f"Row {row_number} has empty {column}")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise LabelConversionError(
            f"Row {row_number} has invalid {column}={value!r}"
        ) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def parse_flare_class(value: str, row_number: int) -> ParsedClass:
    match = CLASS_PATTERN.fullmatch(value)
    if match is None:
        raise LabelConversionError(
            f"Row {row_number} has unsupported flare_class={value!r}"
        )
    letter = match.group(1).upper()
    coefficient = float(match.group(2)) if match.group(2) else 0.0
    normalized_text = f"{letter}{match.group(2)}" if match.group(2) else letter
    return ParsedClass(
        text=normalized_text,
        letter=letter,
        label=CLASS_TO_LABEL[letter],
        coefficient=coefficient,
        equivalent_flux=coefficient * CLASS_BASE_FLUX[letter],
    )


def parse_optional_float(value: str, row_number: int, column: str) -> float | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError as error:
        raise LabelConversionError(
            f"Row {row_number} has invalid {column}={value!r}"
        ) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_appledouble_sidecar(path: Path) -> None:
    """Remove a macOS/NAS metadata sidecar created for this managed output."""

    sidecar = path.with_name(f"._{path.name}")
    if not sidecar.exists() and not sidecar.is_symlink():
        return
    if sidecar.is_symlink() or not sidecar.is_file():
        raise LabelConversionError(
            f"Refusing to remove unexpected AppleDouble path: {sidecar}"
        )
    try:
        sidecar.unlink()
    except OSError as error:
        raise LabelConversionError(
            f"Cannot remove AppleDouble sidecar {sidecar}: {error}"
        ) from error


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        remove_appledouble_sidecar(path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _manifest_file_record(manifest: dict, relative_path: str) -> dict:
    records = [
        record
        for record in manifest.get("files", [])
        if record.get("local_path") == relative_path
    ]
    if len(records) != 1:
        raise LabelConversionError(
            f"Manifest must contain exactly one hash record for {relative_path!r}"
        )
    return records[0]


def _validate_manifest_file(base_dir: Path, relative_path: str, manifest: dict) -> Path:
    path = (base_dir / relative_path).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError as error:
        raise LabelConversionError(
            f"Manifest path escapes its data directory: {relative_path!r}"
        ) from error
    record = _manifest_file_record(manifest, relative_path)
    if not path.is_file():
        raise LabelConversionError(f"Manifest input does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != record.get("sha256"):
        raise LabelConversionError(
            f"SHA256 mismatch for {path}: {actual_hash} != {record.get('sha256')}"
        )
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise LabelConversionError(
            f"Byte-size mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    return path


def load_manifest_inputs(manifest_path: Path) -> tuple[Path, Path, dict]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabelConversionError(
            f"Cannot read manifest {manifest_path}: {error}"
        ) from error
    base_dir = manifest_path.parent
    try:
        catalog_relative_path = str(manifest["catalog_path"])
        metadata_relative_path = str(manifest["metadata_path"])
    except KeyError as error:
        raise LabelConversionError(
            f"Manifest {manifest_path} is missing {error.args[0]}"
        ) from error
    catalog_path = _validate_manifest_file(base_dir, catalog_relative_path, manifest)
    metadata_path = _validate_manifest_file(base_dir, metadata_relative_path, manifest)
    return catalog_path, metadata_path, manifest


def parse_compact_manifest_date(manifest: dict, key: str) -> date:
    try:
        return datetime.strptime(str(manifest[key]), "%Y%m%d").date()
    except (KeyError, ValueError) as error:
        raise LabelConversionError(
            f"Download manifest is missing a valid {key!r}"
        ) from error


def load_coverage(metadata_path: Path) -> tuple[date, date, dict]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        attributes = metadata["global_attributes"]
        coverage_start = datetime.strptime(
            str(attributes["time_coverage_start"]), "%Y%m%d"
        ).date()
        coverage_end = datetime.strptime(
            str(attributes["time_coverage_end"]), "%Y%m%d"
        ).date()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise LabelConversionError(
            f"Cannot determine coverage from metadata {metadata_path}: {error}"
        ) from error
    return coverage_start, coverage_end, metadata


def aggregate_catalog(
    catalog_path: Path,
    start_date: date,
    end_date: date,
    event_time_column: str,
) -> tuple[dict[date, DailyAggregate], dict]:
    daily = {
        start_date + timedelta(days=offset): DailyAggregate()
        for offset in range((end_date - start_date).days + 1)
    }
    exact_rows_seen: set[tuple[str, ...]] = set()
    catalog_rows_seen = 0
    duplicate_rows_skipped = 0
    events_in_coverage = 0
    event_class_counts: Counter[str] = Counter()

    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_CATALOG_COLUMNS - fieldnames
        if event_time_column not in fieldnames:
            missing_columns.add(event_time_column)
        if missing_columns:
            raise LabelConversionError(
                f"Catalog {catalog_path} is missing columns: {sorted(missing_columns)}"
            )

        ordered_fieldnames = reader.fieldnames or []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            catalog_rows_seen += 1
            row_key = tuple(row.get(column, "") for column in ordered_fieldnames)
            if row_key in exact_rows_seen:
                duplicate_rows_skipped += 1
                continue
            exact_rows_seen.add(row_key)

            event_time = parse_timestamp(
                row[event_time_column], row_number, event_time_column
            )
            event_day = event_time.date()
            if event_day < start_date or event_day > end_date:
                continue

            flare_class = parse_flare_class(row["flare_class"], row_number)
            measured_flux = parse_optional_float(
                row["xrsb_irrad"], row_number, "xrsb_irrad"
            )
            daily[event_day].add(flare_class, measured_flux)
            events_in_coverage += 1
            event_class_counts[flare_class.letter] += 1

    stats = {
        "catalog_rows_seen": catalog_rows_seen,
        "exact_duplicate_rows_skipped": duplicate_rows_skipped,
        "events_in_requested_coverage": events_in_coverage,
        "event_class_counts": {
            name: event_class_counts.get(name, 0) for name in CLASS_TO_LABEL
        },
    }
    return daily, stats


def write_daily_labels(
    daily: dict[date, DailyAggregate], output_path: Path, dataset_epoch: date
) -> tuple[Counter[int], int]:
    # Use csv.writer on an in-memory text stream so the final file can be replaced atomically.
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    class_counts: Counter[int] = Counter()
    for day in sorted(daily):
        aggregate = daily[day]
        label = aggregate.label
        class_counts[label] += 1
        writer.writerow(
            {
                "date": day.isoformat(),
                "date_id": (day - dataset_epoch).days,
                "label": label,
                "label_name": LABEL_TO_NAME[label],
                "max_flare_class": aggregate.max_flare_class,
                "max_xrsb_irrad_w_m2": (
                    "" if aggregate.max_flux is None else f"{aggregate.max_flux:.9e}"
                ),
                "flare_count": aggregate.flare_count,
            }
        )
    atomic_write_text(output_path, output.getvalue())
    return class_counts, len(daily)


def build_daily_labels(
    catalog_path: Path,
    metadata_path: Path,
    output_path: Path,
    summary_path: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    event_time_column: str = "start_time",
    dataset_epoch: date = DATASET_EPOCH,
    manifest: dict | None = None,
) -> dict:
    catalog_path = catalog_path.resolve()
    metadata_path = metadata_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()

    source_coverage_start, source_coverage_end, metadata = load_coverage(metadata_path)
    if manifest is None:
        catalog_coverage_start = source_coverage_start
        catalog_coverage_end = source_coverage_end
    else:
        catalog_coverage_start = parse_compact_manifest_date(
            manifest, "catalog_time_coverage_start"
        )
        catalog_coverage_end = parse_compact_manifest_date(
            manifest, "catalog_time_coverage_end"
        )
        if (
            catalog_coverage_start < source_coverage_start
            or catalog_coverage_end > source_coverage_end
        ):
            raise LabelConversionError(
                "Manifest local catalog coverage is outside NOAA metadata coverage"
            )

    requested_start = (
        max(dataset_epoch, catalog_coverage_start) if start_date is None else start_date
    )
    requested_end = catalog_coverage_end if end_date is None else end_date
    if requested_start < catalog_coverage_start or requested_end > catalog_coverage_end:
        raise LabelConversionError(
            "Requested label coverage "
            f"{requested_start}..{requested_end} is outside the locally downloaded "
            f"catalog coverage {catalog_coverage_start}..{catalog_coverage_end}; "
            "refusing to create silent zero labels"
        )
    if requested_start < dataset_epoch:
        raise LabelConversionError(
            f"start_date {requested_start} precedes dataset_epoch {dataset_epoch}"
        )
    if requested_end < requested_start:
        raise LabelConversionError("end_date must not precede start_date")
    if event_time_column not in {"start_time", "time"}:
        raise LabelConversionError(
            "event_time_column must be 'start_time' or peak-time column 'time'"
        )

    daily, catalog_stats = aggregate_catalog(
        catalog_path=catalog_path,
        start_date=requested_start,
        end_date=requested_end,
        event_time_column=event_time_column,
    )
    class_counts, number_of_days = write_daily_labels(
        daily=daily, output_path=output_path, dataset_epoch=dataset_epoch
    )

    attributes = metadata.get("global_attributes", {})
    summary = {
        "schema_version": 2,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": sha256_file(catalog_path),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "source_algorithm_version": attributes.get("flrpt_algorithm_version"),
        "source_license": attributes.get("license"),
        "manifest_transport": None if manifest is None else manifest.get("transport"),
        "manifest_schema_version": (
            None if manifest is None else manifest.get("schema_version")
        ),
        "event_time_column": event_time_column,
        "day_semantics": (
            f"UTC calendar day of {event_time_column}; highest catalogued GOES "
            "letter class wins"
        ),
        "zero_semantics": (
            "No flare event is catalogued in that UTC day; this is not proof that "
            "no physical A/B microflare occurred"
        ),
        "dataset_epoch": dataset_epoch.isoformat(),
        "source_metadata_coverage_start": source_coverage_start.isoformat(),
        "source_metadata_coverage_end": source_coverage_end.isoformat(),
        "local_catalog_coverage_start": catalog_coverage_start.isoformat(),
        "local_catalog_coverage_end": catalog_coverage_end.isoformat(),
        "coverage_start": requested_start.isoformat(),
        "coverage_end": requested_end.isoformat(),
        "number_of_days": number_of_days,
        "label_mapping": {
            "0": "none",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "M",
            "5": "X",
        },
        "daily_label_counts": {
            str(label): class_counts.get(label, 0) for label in range(6)
        },
        **catalog_stats,
    }
    summary["output_path"] = str(output_path)
    summary["output_sha256"] = sha256_file(output_path)
    summary["summary_path"] = str(summary_path)
    atomic_write_text(
        summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Download manifest used to locate catalog/metadata (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Override catalog path instead of reading it from --manifest",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Override NOAA metadata JSON path instead of reading it from --manifest",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=None,
        help=(
            "First output day; defaults to the later of the SolarCHIP epoch and "
            "the locally downloaded catalog start"
        ),
    )
    parser.add_argument("--end-date", type=parse_iso_date, default=None)
    parser.add_argument(
        "--event-time-column",
        choices=("start_time", "time"),
        default="start_time",
        help="Group by flare start time (forecast-window semantics) or peak time",
    )
    parser.add_argument("--dataset-epoch", type=parse_iso_date, default=DATASET_EPOCH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.catalog is None) != (args.metadata is None):
        raise SystemExit("--catalog and --metadata must be supplied together")
    manifest: dict | None = None
    if args.catalog is None or args.metadata is None:
        manifest_catalog, manifest_metadata, manifest = load_manifest_inputs(
            args.manifest.resolve()
        )
    else:
        manifest_catalog = args.catalog
        manifest_metadata = args.metadata
    catalog_path = manifest_catalog if args.catalog is None else args.catalog
    metadata_path = manifest_metadata if args.metadata is None else args.metadata

    summary = build_daily_labels(
        catalog_path=catalog_path,
        metadata_path=metadata_path,
        output_path=args.output,
        summary_path=args.summary,
        start_date=args.start_date,
        end_date=args.end_date,
        event_time_column=args.event_time_column,
        dataset_epoch=args.dataset_epoch,
        manifest=manifest,
    )
    print(f"Daily labels: {summary['output_path']}")
    print(f"Coverage: {summary['coverage_start']} through {summary['coverage_end']}")
    print(f"Daily label counts: {summary['daily_label_counts']}")
    print(f"Summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
