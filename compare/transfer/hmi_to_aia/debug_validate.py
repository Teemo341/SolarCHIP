#!/usr/bin/env python3
"""Dependency-free static validation for the two comparison implementations.

This intentionally does not import model modules: the preparation environment
does not contain PyTorch. It validates syntax, file/config coverage, dotted
target resolution, constructor keys, modal consistency, key hyperparameters,
and documentation boundaries. If PyYAML is available it also parses the YAML
semantically; otherwise that optional check is reported as skipped.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


MODALS = (
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
TARGET_STATS = {
    "0094": (0.7777174413423921, 0.6687824480747059),
    "0131": (1.5390822848548482, 1.0912435219464955),
    "0171": (4.058454606964953, 2.0019955472124473),
    "0193": (4.2085857338462915, 2.031354106628024),
    "0211": (3.4567200476856237, 1.8330093045320472),
    "0304": (1.7023200257985773, 1.4301543643478678),
    "0335": (0.917317310470085, 0.8385782332253703),
    "1600": (2.733473493754583, 1.8048273547243583),
    "1700": (4.482010005696708, 2.2629650384920006),
    "4500": (6.433152566577294, 3.16857582465075),
}
ALGORITHMS = {
    "dash_pix2pixhd": {
        "target": "compare.transfer.hmi_to_aia.dash_pix2pixhd.module.DashPix2PixHD",
        "class": "DashPix2PixHD",
        "module": "compare/transfer/hmi_to_aia/dash_pix2pixhd/module.py",
        "max_epochs": "200",
        "required": (
            "n_downsample: 4",
            "n_residual: 9",
            "n_discriminators: 2",
            "lambda_feature_matching: 10.0",
            "lambda_l1: 0.0",
            "decay_start_epoch: 100",
        ),
    },
    "sdoml_cnn": {
        "target": "compare.transfer.hmi_to_aia.sdoml_cnn.module.GalvezSDOMLCNN",
        "class": "GalvezSDOMLCNN",
        "module": "compare/transfer/hmi_to_aia/sdoml_cnn/module.py",
        "max_epochs": "15",
        "required": (
            "hidden_channels: 128",
            "num_layers: 11",
            "learning_rate: 1.0e-3",
            "momentum: 0.99",
            "weight_decay: 1.0e-8",
            "lr_step_size: 5",
        ),
    },
}


def fail(message: str) -> None:
    raise AssertionError(message)


def python_tree(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    return tree


def top_level_classes(path: Path) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in python_tree(path).body
        if isinstance(node, ast.ClassDef)
    }


def constructor_parameters(path: Path, class_name: str) -> set[str]:
    class_node = top_level_classes(path).get(class_name)
    if class_node is None:
        fail(f"{path}: missing top-level class {class_name}")
    initializer = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ),
        None,
    )
    if initializer is None:
        return set()
    positional = [*initializer.args.posonlyargs, *initializer.args.args]
    return {
        argument.arg
        for argument in [*positional, *initializer.args.kwonlyargs]
        if argument.arg != "self"
    }


def model_parameter_keys(config_text: str) -> set[str]:
    match = re.search(
        r"(?ms)^model:\s*$\n(?P<body>.*?)(?=^[^ \n][^\n]*:\s*$)", config_text
    )
    if match is None:
        fail("Config has no model section")
    params = re.search(r"(?ms)^  params:\s*$\n(?P<body>.*)", match.group("body"))
    if params is None:
        fail("Config model section has no params mapping")
    return set(re.findall(r"(?m)^    ([A-Za-z_]\w*):", params.group("body")))


def validate_python(code_root: Path, repo_root: Path) -> None:
    python_files = sorted(code_root.rglob("*.py"))
    if not python_files:
        fail("No Python files found")
    for path in python_files:
        python_tree(path)
    for metadata in ALGORITHMS.values():
        expected_path = repo_root / metadata["module"]
        if metadata["class"] not in top_level_classes(expected_path):
            fail(f"{expected_path}: missing top-level model class {metadata['class']}")


def validate_config(
    path: Path,
    modal: str,
    algorithm: str,
    metadata: dict[str, object],
    repo_root: Path,
) -> None:
    text = path.read_text(encoding="utf-8")
    if f"target: {metadata['target']}" not in text:
        fail(f"{path}: wrong model target")
    if not re.search(r"source_modal:\s*['\"]hmi['\"]", text):
        fail(f"{path}: missing source_modal=hmi")
    if not re.search(rf"target_modal:\s*['\"]{modal}['\"]", text):
        fail(f"{path}: target_modal mismatch")
    if "input_channels: 1" not in text or "output_channels: 1" not in text:
        fail(f"{path}: project comparison must be configured 1-to-1")
    modal_list = f"modal_list: ['hmi', '{modal}']"
    if text.count(modal_list) != 2:
        fail(f"{path}: train/validation modal_list mismatch")
    if f"max_epochs: {metadata['max_epochs']}" not in text:
        fail(f"{path}: wrong trainer max_epochs")
    if "shuffle_val_dataloader: false" not in text:
        fail(f"{path}: validation order is not fixed")
    expected_strategy = (
        "ddp_find_unused_parameters_true" if algorithm == "dash_pix2pixhd" else "ddp"
    )
    if f"strategy: {expected_strategy}" not in text:
        fail(f"{path}: wrong DDP strategy for {algorithm}")
    expected_batch_size = 1 if algorithm == "dash_pix2pixhd" else 2
    if not re.search(rf"(?m)^    batch_size:\s*{expected_batch_size}\s*$", text):
        fail(f"{path}: wrong project batch size for {algorithm}")
    if "torch_augment_type: [1024, 0.0, 0]" not in text:
        fail(f"{path}: validation augmentation is not deterministic")
    if "monitor: val/loss" not in text or "save_last: true" not in text:
        fail(f"{path}: checkpoint contract incomplete")
    for required in metadata["required"]:
        if required not in text:
            fail(f"{path}: missing {required!r}")
    if modal == "4500" and "extension" not in text.lower():
        fail(f"{path}: 4500 must be documented as an extension")
    if (
        algorithm == "dash_pix2pixhd"
        and modal == "0304"
        and "direct paper task" not in text
    ):
        fail(f"{path}: Dash 0304 direct-evidence marker missing")

    expected_mean, expected_std = TARGET_STATS[modal]
    mean_match = re.search(r"(?m)^    target_mean:\s*([^ #]+)", text)
    std_match = re.search(r"(?m)^    target_std:\s*([^ #]+)", text)
    if mean_match is None or float(mean_match.group(1)) != expected_mean:
        fail(f"{path}: target_mean differs from SolarDataset.modal_status")
    if std_match is None or float(std_match.group(1)) != expected_std:
        fail(f"{path}: target_std differs from SolarDataset.modal_status")

    constructor_keys = constructor_parameters(
        repo_root / metadata["module"],
        str(metadata["class"]),
    )
    unknown_keys = model_parameter_keys(text) - constructor_keys
    if unknown_keys:
        fail(
            f"{path}: model params not accepted by constructor: {sorted(unknown_keys)}"
        )


def validate_configs(repo_root: Path) -> None:
    config_root = repo_root / "configs" / "compare" / "hmi_to_aia"
    for algorithm, metadata in ALGORITHMS.items():
        directory = config_root / algorithm
        configs = sorted(directory.glob("hmi_to_*.yaml"))
        if len(configs) != len(MODALS):
            fail(f"{directory}: expected 10 configs, found {len(configs)}")
        expected = {f"hmi_to_{modal}.yaml" for modal in MODALS}
        if {path.name for path in configs} != expected:
            fail(
                f"{directory}: config filenames do not cover the ten project modalities"
            )
        for modal in MODALS:
            validate_config(
                directory / f"hmi_to_{modal}.yaml",
                modal,
                algorithm,
                metadata,
                repo_root,
            )


def validate_semantic_yaml(repo_root: Path) -> bool:
    try:
        import yaml
    except ImportError:
        return False

    config_root = repo_root / "configs" / "compare" / "hmi_to_aia"
    for algorithm, metadata in ALGORITHMS.items():
        for modal in MODALS:
            path = config_root / algorithm / f"hmi_to_{modal}.yaml"
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
            if (
                not isinstance(parsed, dict)
                or not {"model", "data", "lightning"} <= parsed.keys()
            ):
                fail(f"{path}: semantic YAML root contract is invalid")
            if parsed["model"]["target"] != metadata["target"]:
                fail(f"{path}: semantic model target mismatch")
            if parsed["model"]["params"]["target_modal"] != modal:
                fail(f"{path}: semantic target_modal mismatch")
            for split in ("train", "validation"):
                modal_list = parsed["data"]["params"][split]["params"]["modal_list"]
                if modal_list != ["hmi", modal]:
                    fail(f"{path}: semantic {split} modal_list mismatch")
    return True


def validate_documentation(code_root: Path) -> None:
    top = (code_root / "README.md").read_text(encoding="utf-8")
    dash = (code_root / "dash_pix2pixhd" / "README.md").read_text(encoding="utf-8")
    galvez = (code_root / "sdoml_cnn" / "README.md").read_text(encoding="utf-8")
    combined = "\n".join((top, dash, galvez)).lower()
    for phrase in ("not a strict reproduction", "4500", "single-band los adaptation"):
        if phrase not in combined:
            fail(f"Documentation boundary missing phrase: {phrase}")
    if "local enhancer" not in dash.lower():
        fail("Dash paper/code generator conflict is not documented")
    if "not a gan" not in galvez.lower():
        fail("Galvez deterministic-model boundary is not documented")


def main() -> int:
    code_root = Path(__file__).resolve().parent
    repo_root = code_root.parents[2]
    validate_python(code_root, repo_root)
    validate_configs(repo_root)
    validate_documentation(code_root)
    yaml_checked = validate_semantic_yaml(repo_root)
    print("Static debug validation passed")
    print(f"  Python files: {len(list(code_root.rglob('*.py')))}")
    print("  Configs: 20 (10 Dash Pix2PixHD + 10 Galvez SDOML-CNN)")
    print(
        "  YAML semantic parse: passed"
        if yaml_checked
        else "  YAML semantic parse: skipped (PyYAML unavailable)"
    )
    print("  Tensor execution: skipped (PyTorch unavailable)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Static debug validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
