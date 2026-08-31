"""Evaluate SolarPredictor and flare comparison checkpoints with common metrics.

The resume path may be either a training run directory or one concrete
checkpoint.  The logged project config is always reused so every model sees
the same dataset preprocessing that was recorded during training.

Examples
--------
Evaluate ``last.ckpt`` and report every metric::

    python -m downstream.flare.test \
        -r logs/compare_flare/deepswm/2026-08-30T22-48-11

Evaluate one checkpoint and select metrics manually::

    python -m downstream.flare.test \
        -r logs/compare_flare/deepswm/2026-08-30T22-48-11/checkpoints/last.ckpt \
        --metrics overall_acc pod csi far hss tss acc

``overall_acc`` is the accuracy after reducing the flare classes to
``0AB / C / MX``.  The binary metrics are calculated independently for C+
(``>=C``) and M+ (``>=M``) from each model's final decoded class prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if not (Path.cwd() / "data").is_dir():
    os.chdir(REPO_ROOT)

from data.build import WrappedDataset, instantiate_from_config
from downstream.flare.data.class_groups import (
    DEFAULT_CLASS_GROUPS,
    normalize_class_groups,
)
from downstream.flare.data.dataset import DATASET_EPOCH
from downstream.flare.data.metrics import (
    binary_metric_values,
    class_reduction_mappings,
    collapse_confusion,
)


BINARY_METRICS = ("pod", "csi", "far", "hss", "tss", "acc")
THRESHOLDS = ("c_plus", "m_plus")
CANONICAL_METRICS = (
    "overall_acc",
    *(f"{threshold}_{metric}" for threshold in THRESHOLDS for metric in BINARY_METRICS),
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test a SolarPredictor or flare comparison checkpoint with common "
            "overall/C+/M+ metrics."
        )
    )
    parser.add_argument(
        "-r",
        "--resume",
        required=True,
        help="Training run directory or a concrete .ckpt path.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Optional checkpoint override when --resume is a run directory. "
            "Accepts a path or a filename under RUN/checkpoints."
        ),
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Logged dataset split to evaluate (default: validation).",
    )
    parser.add_argument(
        "--time_interval",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Override the selected split's logged [start, end) interval.",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=None,
        help="Override the selected split's logged time_step.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the logged DataModule batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override the logged DataModule worker count.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, mps, or musa:0.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help=(
            "Metrics to report. Use all; overall_acc; a base name "
            "(pod/csi/far/hss/tss/acc) to select both thresholds; or an "
            "explicit name such as c_plus_tss or m_plus_far."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Evaluate only the first N batches (smoke testing).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "JSON output path. Default: logs/test_results/"
            "MODEL_RUN_CHECKPOINT_SPLIT_metrics.json."
        ),
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Print results without writing a JSON file.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser


def _absolute_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _find_run_directory(checkpoint: Path) -> Path:
    for candidate in checkpoint.parents:
        if (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a training run containing configs/ above {checkpoint}"
    )


def _resolve_checkpoint_override(run_directory: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        run_directory / raw,
        run_directory / "checkpoints" / raw,
    ]
    if raw.suffix != ".ckpt":
        candidates.extend(path.with_suffix(".ckpt") for path in list(candidates))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve checkpoint override {value!r}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_resume_paths(
    resume: str | Path, checkpoint_override: str | None = None
) -> tuple[Path, Path]:
    """Resolve ``(run_directory, checkpoint)`` from the accepted CLI layouts."""

    path = _absolute_path(resume)
    if path.is_file():
        if checkpoint_override is not None:
            raise ValueError("--ckpt cannot be combined with a checkpoint --resume path")
        if path.suffix != ".ckpt":
            raise ValueError(f"Resume file must end in .ckpt: {path}")
        return _find_run_directory(path), path

    if not path.is_dir():
        raise FileNotFoundError(f"Resume path does not exist: {path}")
    run_directory = path.parent if path.name == "checkpoints" else path
    if not (run_directory / "configs").is_dir():
        raise FileNotFoundError(
            f"Training run has no configs/ directory: {run_directory}"
        )
    if checkpoint_override is not None:
        checkpoint = _resolve_checkpoint_override(run_directory, checkpoint_override)
    else:
        checkpoint = run_directory / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            available = sorted((run_directory / "checkpoints").glob("*.ckpt"))
            if len(available) == 1:
                checkpoint = available[0]
            else:
                raise FileNotFoundError(
                    f"No last.ckpt in {run_directory / 'checkpoints'}; "
                    f"available checkpoints: {[path.name for path in available]}"
                )
    return run_directory.resolve(), checkpoint.resolve()


def load_run_config(run_directory: Path) -> tuple[DictConfig, list[Path]]:
    config_directory = run_directory / "configs"
    paths = sorted(config_directory.glob("*-project.yaml"))
    if not paths:
        paths = sorted(config_directory.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML configs found under {config_directory}")

    configs = [OmegaConf.load(path) for path in paths]
    merged = OmegaConf.merge(*configs)
    if "model" not in merged or "data" not in merged:
        raise KeyError(
            f"Merged run config must contain model and data sections: {paths}"
        )
    return merged, paths


def _model_config_for_restore(
    model_config: Mapping[str, Any] | DictConfig, run_directory: Path
) -> DictConfig:
    restored = OmegaConf.create(
        OmegaConf.to_container(model_config, resolve=True)
    )
    target = str(restored.target)
    params = restored.setdefault("params", {})
    if target == "compare.flare.pcnn.module.PCNN":
        # A full downstream checkpoint contains every backbone tensor. Avoid a
        # redundant ImageNet download before those tensors are restored.
        params["defer_imagenet_weights"] = True
    if target.endswith(".SolarPredictor"):
        # SolarPredictor checkpoints are required to be self-contained. A
        # directory is deliberately not a file, so its constructor builds only
        # the architecture and on_load_checkpoint verifies the complete state.
        params["pretrained_ckpt_path"] = str(run_directory)
    return restored


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"Checkpoint is not readable by the current user: {path}. "
            "Grant read permission on the training host or run the test as "
            "the checkpoint owner."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must contain a mapping, got {type(checkpoint).__name__}"
        )
    return checkpoint


def load_model_from_run(
    config: DictConfig, run_directory: Path, checkpoint_path: Path
) -> tuple[torch.nn.Module, str]:
    model_config = _model_config_for_restore(config.model, run_directory)
    model_target = str(model_config.target)
    if not model_target.startswith(("compare.flare.", "downstream.flare.")):
        raise ValueError(
            "Flare test only accepts compare.flare or downstream.flare models, "
            f"got {model_target!r}"
        )
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise KeyError(
                f"Lightning checkpoint has no state_dict mapping: {checkpoint_path}"
            )

    model = instantiate_from_config(model_config)
    on_load_checkpoint = getattr(model, "on_load_checkpoint", None)
    if callable(on_load_checkpoint) and "state_dict" in checkpoint:
        on_load_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_target


def _copy_split_config(
    config: DictConfig,
    split: str,
    time_interval: Sequence[int] | None,
    time_step: int | None,
) -> DictConfig:
    params = config.data.get("params", {})
    available = [
        name for name in ("train", "validation", "test", "predict") if name in params
    ]
    if split not in params:
        raise KeyError(
            f"Data config has no split {split!r}; available splits: {available}"
        )
    split_config = OmegaConf.create(
        OmegaConf.to_container(params[split], resolve=True)
    )
    split_params = split_config.setdefault("params", {})
    if time_interval is not None:
        if len(time_interval) != 2 or int(time_interval[0]) >= int(time_interval[1]):
            raise ValueError("--time_interval requires START < END")
        split_params["time_interval"] = [int(time_interval[0]), int(time_interval[1])]
    if time_step is not None:
        if time_step < 1:
            raise ValueError("--time_step must be positive")
        split_params["time_step"] = int(time_step)
    return split_config


def build_evaluation_loader(
    config: DictConfig,
    split: str,
    batch_size: int | None,
    num_workers: int | None,
    time_interval: Sequence[int] | None,
    time_step: int | None,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, Any, DictConfig]:
    split_config = _copy_split_config(config, split, time_interval, time_step)
    dataset = instantiate_from_config(split_config)
    data_params = config.data.get("params", {})
    if bool(data_params.get("wrap", False)):
        dataset = WrappedDataset(dataset)
    if data_params.get("custom_collate_fn") is not None:
        raise ValueError(
            "Flare evaluation requires the default collate function so that the "
            "label key is retained"
        )

    resolved_batch_size = int(
        data_params.get("batch_size", 1) if batch_size is None else batch_size
    )
    resolved_workers = int(
        data_params.get("num_workers", 0) if num_workers is None else num_workers
    )
    if resolved_batch_size < 1 or resolved_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=device.type in {"cuda", "musa"},
        drop_last=False,
        generator=generator,
    )
    return loader, dataset, split_config


def _dataset_attribute(source: Any, name: str) -> Any | None:
    """Read one attribute through the project's optional WrappedDataset."""

    current = source
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "data", None)
    return None


def _date_for_id(date_id: int) -> str:
    return (DATASET_EPOCH + timedelta(days=int(date_id))).isoformat()


def evaluation_time_metadata(
    split_config: DictConfig, dataset: Any
) -> dict[str, Any]:
    """Describe the requested interval and the final label-filtered dataset."""

    params = split_config.get("params", {})
    raw_interval = params.get("time_interval")
    if raw_interval is None or len(raw_interval) != 2:
        raise ValueError("Evaluation dataset config must define time_interval")
    start_id, end_id = (int(raw_interval[0]), int(raw_interval[1]))
    if start_id >= end_id:
        raise ValueError("Evaluation time_interval must satisfy START < END")
    time_step = int(params.get("time_step", 1))
    if time_step < 1:
        raise ValueError("Evaluation time_step must be positive")

    raw_exist_idx = _dataset_attribute(dataset, "exist_idx")
    if raw_exist_idx is None:
        raise ValueError(
            f"Dataset {type(dataset).__name__} does not expose exist_idx"
        )
    retained_date_ids = [int(value) for value in raw_exist_idx]
    if not retained_date_ids:
        raise ValueError("Evaluation dataset has no retained date IDs")
    first_retained_id = min(retained_date_ids)
    last_retained_id = max(retained_date_ids)

    missing_date_ids = tuple(
        int(value)
        for value in (
            _dataset_attribute(dataset, "missing_label_date_ids") or ()
        )
    )
    missing_metadata: dict[str, Any] = {
        "dropped_count": int(
            _dataset_attribute(dataset, "num_dropped_for_missing_labels")
            or len(missing_date_ids)
        )
    }
    if missing_date_ids:
        missing_metadata.update(
            {
                "first_dropped_date_id": missing_date_ids[0],
                "first_dropped_date": _date_for_id(missing_date_ids[0]),
                "last_dropped_date_id": missing_date_ids[-1],
                "last_dropped_date": _date_for_id(missing_date_ids[-1]),
            }
        )

    return {
        "time_interval": [start_id, end_id],
        "time_step": time_step,
        "requested_time_range": {
            "start_date_id": start_id,
            "start_date": _date_for_id(start_id),
            "last_date_id_inclusive": end_id - 1,
            "last_date_inclusive": _date_for_id(end_id - 1),
            "end_date_id_exclusive": end_id,
            "end_date_exclusive": _date_for_id(end_id),
        },
        "retained_dataset_time_range": {
            "first_date_id": first_retained_id,
            "first_date": _date_for_id(first_retained_id),
            "last_date_id": last_retained_id,
            "last_date": _date_for_id(last_retained_id),
            "dataset_sample_count": len(retained_date_ids),
        },
        "missing_label_filter": missing_metadata,
    }


def _print_evaluation_time_metadata(metadata: Mapping[str, Any]) -> None:
    requested = metadata["requested_time_range"]
    retained = metadata["retained_dataset_time_range"]
    print(
        "Test time interval "
        f"[{metadata['time_interval'][0]}, {metadata['time_interval'][1]}): "
        f"{requested['start_date']} to {requested['end_date_exclusive']} "
        f"(end exclusive), time_step={metadata['time_step']}"
    )
    print(
        "Retained dataset time range: "
        f"{retained['first_date']} (date_id={retained['first_date_id']}) to "
        f"{retained['last_date']} (date_id={retained['last_date_id']}), "
        f"samples={retained['dataset_sample_count']}"
    )
    dropped_count = int(metadata["missing_label_filter"]["dropped_count"])
    if dropped_count:
        print(f"Missing flare labels dropped {dropped_count} selected dates")


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif hasattr(torch, "musa") and torch.musa.is_available():
            normalized = "musa"
        elif torch.backends.mps.is_available():
            normalized = "mps"
        else:
            normalized = "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS was requested but is unavailable: {value}")
    if device.type == "musa" and not (
        hasattr(torch, "musa") and torch.musa.is_available()
    ):
        raise RuntimeError(f"MUSA was requested but is unavailable: {value}")
    return device


def _find_class_groups(source: Any) -> tuple[str, ...] | None:
    visited: set[int] = set()
    current = source
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        groups = getattr(current, "class_groups", None)
        if groups is not None:
            return normalize_class_groups(groups)
        next_source = getattr(current, "dataset", None)
        if next_source is None:
            next_source = getattr(current, "data", None)
        current = next_source
    return None


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


@torch.inference_mode()
def collect_grouped_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int | None,
    quiet: bool,
) -> tuple[torch.Tensor, int]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("--max_batches must be positive")
    if not callable(getattr(model, "predict_step", None)):
        raise TypeError(f"{type(model).__name__} does not implement predict_step")

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    batch_count = 0
    iterator = tqdm(loader, desc="Testing", disable=quiet)
    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        if not isinstance(batch, Mapping) or "label" not in batch:
            raise KeyError("Every flare batch must be a mapping containing label")
        device_batch = _move_to_device(batch, device)
        output = model.predict_step(device_batch, batch_index)
        if not isinstance(output, Mapping) or "prediction" not in output:
            raise KeyError(
                f"{type(model).__name__}.predict_step must return prediction"
            )
        predictions = output["prediction"].detach().cpu().long().reshape(-1)
        labels = batch["label"].detach().cpu().long().reshape(-1)
        if predictions.shape != labels.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: {tuple(predictions.shape)} != "
                f"{tuple(labels.shape)}"
            )
        if labels.numel() and (
            int(labels.min()) < 0
            or int(predictions.min()) < 0
            or int(labels.max()) >= num_classes
            or int(predictions.max()) >= num_classes
        ):
            raise ValueError(
                f"Labels and predictions must be in [0,{num_classes - 1}]"
            )
        encoded = labels * num_classes + predictions
        confusion.add_(
            torch.bincount(encoded, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )
        )
        batch_count += 1
    if int(confusion.sum()) == 0:
        raise RuntimeError("Evaluation produced no samples")
    return confusion, batch_count


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def resolve_metric_names(requested: Sequence[str]) -> tuple[str, ...]:
    if not requested:
        raise ValueError("At least one metric must be requested")
    selected: list[str] = []
    aliases = {
        "overall_accuracy": "overall_acc",
        "accuracy": "acc",
        **{
            f"{threshold}_accuracy": f"{threshold}_acc"
            for threshold in THRESHOLDS
        },
    }

    def add(name: str) -> None:
        if name not in selected:
            selected.append(name)

    for raw_name in requested:
        name = aliases.get(raw_name.strip().lower(), raw_name.strip().lower())
        if name == "all":
            for metric in CANONICAL_METRICS:
                add(metric)
        elif name == "overall_acc":
            add(name)
        elif name in BINARY_METRICS:
            for threshold in THRESHOLDS:
                add(f"{threshold}_{name}")
        elif name in CANONICAL_METRICS:
            add(name)
        else:
            raise ValueError(
                f"Unknown metric {raw_name!r}. Use all, overall_acc, "
                f"{list(BINARY_METRICS)}, or one of {list(CANONICAL_METRICS[1:])}"
            )
    return tuple(selected)


def compute_selected_metrics(
    grouped_confusion: torch.Tensor,
    class_groups: Sequence[str],
    requested: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    selected = resolve_metric_names(requested)
    mappings = class_reduction_mappings(class_groups)
    overall = collapse_confusion(grouped_confusion, mappings["overall"], 3)
    c_plus = collapse_confusion(grouped_confusion, mappings["c_plus"], 2)
    m_plus = collapse_confusion(grouped_confusion, mappings["m_plus"], 2)
    binary_values = {
        "c_plus": binary_metric_values(c_plus),
        "m_plus": binary_metric_values(m_plus),
    }
    overall_acc = _safe_ratio(float(overall.diag().sum()), float(overall.sum()))

    metrics: dict[str, float] = {}
    for name in selected:
        if name == "overall_acc":
            metrics[name] = overall_acc
            continue
        threshold, metric = name.rsplit("_", 1)
        metrics[name] = binary_values[threshold][metric]

    confusions = {
        "grouped": grouped_confusion.tolist(),
        "overall_0_C_M": overall.to(torch.long).tolist(),
        "c_plus": c_plus.to(torch.long).tolist(),
        "m_plus": m_plus.to(torch.long).tolist(),
        "binary_layout": [["TN", "FP"], ["FN", "TP"]],
    }
    return metrics, confusions


def _default_output_path(
    run_directory: Path, checkpoint: Path, split: str
) -> Path:
    return (
        REPO_ROOT
        / "logs"
        / "test_results"
        / (
            f"{run_directory.parent.name}_{run_directory.name}_"
            f"{checkpoint.stem}_{split}_metrics.json"
        )
    )


def main(args: Sequence[str] | None = None) -> dict[str, Any]:
    options = get_parser().parse_args(args)
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    run_directory, checkpoint_path = resolve_resume_paths(
        options.resume, options.ckpt
    )
    config, config_paths = load_run_config(run_directory)
    device = resolve_device(options.device)
    model, model_target = load_model_from_run(
        config, run_directory, checkpoint_path
    )
    model = model.to(device)

    loader, dataset, split_config = build_evaluation_loader(
        config=config,
        split=options.split,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        time_interval=options.time_interval,
        time_step=options.time_step,
        device=device,
        seed=options.seed,
    )
    dataset_target = str(split_config.target)
    time_metadata = evaluation_time_metadata(split_config, dataset)
    if not options.quiet:
        _print_evaluation_time_metadata(time_metadata)
    class_groups = _find_class_groups(dataset)
    if class_groups is None:
        raise ValueError(
            f"Dataset {type(dataset).__name__} does not expose class_groups"
        )
    model_groups = _find_class_groups(model)
    if model_groups is not None and model_groups != class_groups:
        raise ValueError(
            f"Model/dataset class_groups mismatch: {model_groups} != {class_groups}"
        )
    if (
        model_target.startswith("compare.flare.")
        and class_groups != DEFAULT_CLASS_GROUPS
    ):
        raise ValueError(
            "The comparison models are hard-coded for class_groups "
            f"{list(DEFAULT_CLASS_GROUPS)}, got {list(class_groups)}"
        )

    grouped_confusion, batch_count = collect_grouped_confusion(
        model=model,
        loader=loader,
        device=device,
        num_classes=len(class_groups),
        max_batches=options.max_batches,
        quiet=options.quiet,
    )
    metrics, confusions = compute_selected_metrics(
        grouped_confusion, class_groups, options.metrics
    )
    result: dict[str, Any] = {
        "run_directory": str(run_directory),
        "checkpoint": str(checkpoint_path),
        "config_paths": [str(path) for path in config_paths],
        "model_target": model_target,
        "dataset_target": dataset_target,
        "split": options.split,
        **time_metadata,
        "device": str(device),
        "class_groups": list(class_groups),
        "overall_class_groups": ["0AB", "C", "MX"],
        "sample_count": int(grouped_confusion.sum()),
        "batch_count": batch_count,
        "zero_division": 0.0,
        "metrics": metrics,
        "confusion_matrices": confusions,
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if not options.no_save:
        output_path = (
            _absolute_path(options.output)
            if options.output is not None
            else _default_output_path(run_directory, checkpoint_path, options.split)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        if not options.quiet:
            print(f"Saved metrics to {output_path}")
        result["output_path"] = str(output_path)
    return result


if __name__ == "__main__":
    main()
