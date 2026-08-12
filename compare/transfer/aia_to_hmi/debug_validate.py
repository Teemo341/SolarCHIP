#!/usr/bin/env python3
"""Dependency-free static checks for the AIA-to-HMI comparisons.

The preparation environment intentionally has no PyTorch. This script avoids
importing model modules and instead validates Python syntax/AST contracts plus
the twenty YAML experiment files. If PyYAML is installed, it additionally
performs a semantic YAML parse; otherwise that optional check is reported as
skipped.
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
HMI_MEAN = -0.0033644122878536808
HMI_STD = 1.4462468177923982
ALGORITHMS = {
    "dannehl_pix2pixcc": {
        "target": (
            "compare.transfer.aia_to_hmi.dannehl_pix2pixcc.module.DannehlPix2PixCC"
        ),
        "class": "DannehlPix2PixCC",
        "module": "compare/transfer/aia_to_hmi/dannehl_pix2pixcc/module.py",
        "required": (
            "generator_channels: 64",
            "discriminator_channels: 64",
            "n_downsample: 4",
            "n_residual: 9",
            "n_cc_scales: 4",
            "lambda_lsgan: 2.0",
            "lambda_feature_matching: 10.0",
            "lambda_cc: 5.0",
            "learning_rate: 2.0e-4",
        ),
    },
    "i2iwfilm": {
        "target": "compare.transfer.aia_to_hmi.i2iwfilm.module.SayezI2IwFiLM",
        "class": "SayezI2IwFiLM",
        "module": "compare/transfer/aia_to_hmi/i2iwfilm/module.py",
        "required": (
            "base_channels: 32",
            "guidance_dim: 256",
            "stage1_epochs: 100",
            "max_epochs: 200",
            "lambda_reconstruction: 1.0",
            "lambda_guidance: 1.0",
            "learning_rate: 1.0e-4",
            "weight_decay: 1.0e-4",
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


def class_method_names(path: Path, class_name: str) -> set[str]:
    class_node = top_level_classes(path).get(class_name)
    if class_node is None:
        fail(f"{path}: missing top-level class {class_name}")
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_python(code_root: Path, repo_root: Path) -> None:
    python_files = sorted(
        path for path in code_root.rglob("*.py") if not path.name.startswith("._")
    )
    if not python_files:
        fail("No Python files found")
    for path in python_files:
        python_tree(path)

    required_methods = {
        "forward",
        "training_step",
        "validation_step",
        "test_step",
        "configure_optimizers",
        "log_images",
    }
    for metadata in ALGORITHMS.values():
        module_path = repo_root / str(metadata["module"])
        methods = class_method_names(module_path, str(metadata["class"]))
        missing = required_methods - methods
        if missing:
            fail(f"{module_path}: missing Lightning methods {sorted(missing)}")

        source = module_path.read_text(encoding="utf-8")
        if "import pytorch_lightning as pl" not in source:
            fail(f"{module_path}: must use the trainer's pytorch_lightning namespace")
        for key_fragment in (
            "visualization/{self.source_modal}/condition",
            "visualization/hmi/target",
            "visualization/hmi/generated",
        ):
            if key_fragment not in source:
                fail(f"{module_path}: missing safe image logger key {key_fragment!r}")

    dannehl_source = (
        repo_root / str(ALGORITHMS["dannehl_pix2pixcc"]["module"])
    ).read_text(encoding="utf-8")
    for fragment in (
        "self.automatic_optimization = False",
        "self.manual_backward",
        "self.toggle_optimizer",
    ):
        if fragment not in dannehl_source:
            fail(f"Dannehl manual-optimization contract missing {fragment!r}")

    sayez_source = (repo_root / str(ALGORITHMS["i2iwfilm"]["module"])).read_text(
        encoding="utf-8"
    )
    if re.search(r"\bdiscriminator\b", sayez_source, flags=re.IGNORECASE):
        fail("I2IwFiLM module unexpectedly contains a discriminator")


def validate_config(
    path: Path,
    modal: str,
    algorithm: str,
    metadata: dict[str, object],
    repo_root: Path,
) -> None:
    text = path.read_text(encoding="utf-8")
    if f"target: {metadata['target']}" not in text:
        fail(f"{path}: wrong dotted model target")
    if not re.search(rf"source_modal:\s*['\"]{modal}['\"]", text):
        fail(f"{path}: source_modal mismatch")
    if not re.search(r"target_modal:\s*['\"]hmi['\"]", text):
        fail(f"{path}: target_modal must be hmi")

    channel_contracts = (
        ("input_channels: 1", "output_channels: 1"),
        ("in_channels: 1", "out_channels: 1"),
    )
    if not any(
        all(fragment in text for fragment in pair) for pair in channel_contracts
    ):
        fail(f"{path}: comparison must be configured as one input to one output")

    modal_list = f"modal_list: ['hmi', '{modal}']"
    if text.count(modal_list) != 2:
        fail(f"{path}: train/validation modal_list direction contract is wrong")
    for fragment in (
        "enhance_type: ['log1p', 'zscore']",
        "time_interval: [0, 5000]",
        "time_interval: [5000, 5400]",
        "torch_augment_type: [1024, 0.5, 360]",
        "torch_augment_type: [1024, 0.0, 0]",
        "shuffle_val_dataloader: false",
        "strategy: ddp_find_unused_parameters_true",
        "max_epochs: 200",
        "monitor: val/loss",
        "save_last: true",
        "save_weights_only: false",
    ):
        if fragment not in text:
            fail(f"{path}: missing project contract {fragment!r}")
    if not re.search(r"(?m)^    batch_size:\s*1\s*$", text):
        fail(f"{path}: expected project batch_size=1")
    for required in metadata["required"]:
        if required not in text:
            fail(f"{path}: missing paper/project default {required!r}")

    mean = re.search(r"(?m)^    (?:target_mean|hmi_mean):\s*([^ #]+)", text)
    std = re.search(r"(?m)^    (?:target_std|hmi_std):\s*([^ #]+)", text)
    if mean is None or float(mean.group(1)) != HMI_MEAN:
        fail(f"{path}: target_mean differs from SolarDataset.modal_status['hmi']")
    if std is None or float(std.group(1)) != HMI_STD:
        fail(f"{path}: target_std differs from SolarDataset.modal_status['hmi']")

    lowered = text.lower()
    if modal == "0304" and "direct paper task" not in lowered:
        fail(f"{path}: 0304 direct-evidence marker missing")
    if modal != "0304" and "transfer" not in lowered and "extension" not in lowered:
        fail(f"{path}: extrapolation marker missing")
    if modal == "4500" and "project extension" not in lowered:
        fail(f"{path}: 4500 project-extension marker missing")

    constructor_keys = constructor_parameters(
        repo_root / str(metadata["module"]), str(metadata["class"])
    )
    unknown = model_parameter_keys(text) - constructor_keys
    if unknown:
        fail(f"{path}: constructor does not accept model params {sorted(unknown)}")


def validate_configs(repo_root: Path) -> None:
    config_root = repo_root / "configs" / "compare" / "aia_to_hmi"
    for algorithm, metadata in ALGORITHMS.items():
        directory = config_root / algorithm
        configs = sorted(directory.glob("aia_*_to_hmi.yaml"))
        expected = {f"aia_{modal}_to_hmi.yaml" for modal in MODALS}
        if {path.name for path in configs} != expected:
            fail(f"{directory}: configs do not exactly cover the ten AIA modalities")
        for modal in MODALS:
            validate_config(
                directory / f"aia_{modal}_to_hmi.yaml",
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

    root = repo_root / "configs" / "compare" / "aia_to_hmi"
    for algorithm, metadata in ALGORITHMS.items():
        for modal in MODALS:
            path = root / algorithm / f"aia_{modal}_to_hmi.yaml"
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
            if (
                not isinstance(parsed, dict)
                or not {"model", "data", "lightning"} <= parsed.keys()
            ):
                fail(f"{path}: semantic YAML root contract is invalid")
            if parsed["model"]["target"] != metadata["target"]:
                fail(f"{path}: semantic dotted target mismatch")
            params = parsed["model"]["params"]
            if params["source_modal"] != modal or params["target_modal"] != "hmi":
                fail(f"{path}: semantic model direction mismatch")
            for split in ("train", "validation"):
                modal_list = parsed["data"]["params"][split]["params"]["modal_list"]
                if modal_list != ["hmi", modal]:
                    fail(f"{path}: semantic {split} modal_list mismatch")
    return True


def validate_documentation(code_root: Path) -> None:
    top = (code_root / "README.md").read_text(encoding="utf-8").lower()
    dannehl = (
        (code_root / "dannehl_pix2pixcc" / "README.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    sayez = (code_root / "i2iwfilm" / "README.md").read_text(encoding="utf-8").lower()
    combined = "\n".join((top, dannehl, sayez))
    for phrase in (
        "not bit-exact",
        "4500",
        "aia 304",
        "substitute",
        "source-only",
    ):
        if phrase not in combined:
            fail(f"Documentation boundary missing phrase {phrase!r}")
    if "paper/code" not in dannehl and "paper and code" not in dannehl:
        fail("Dannehl paper/code architecture conflict is not documented")
    if not any(phrase in sayez for phrase in ("not a gan", "无判别器", "无 gan")):
        fail("Sayez non-adversarial boundary is not documented")
    if not any(phrase in sayez for phrase in ("two-stage", "两个训练阶段")):
        fail("Sayez two-stage method is not documented")
    if not any(phrase in sayez for phrase in ("single run", "一次 `trainer.fit")):
        fail("Sayez single-run two-stage adaptation is not documented")


def main() -> int:
    code_root = Path(__file__).resolve().parent
    repo_root = code_root.parents[2]
    validate_python(code_root, repo_root)
    validate_configs(repo_root)
    validate_documentation(code_root)
    yaml_checked = validate_semantic_yaml(repo_root)
    print("Static debug validation passed")
    python_count = sum(
        1 for path in code_root.rglob("*.py") if not path.name.startswith("._")
    )
    print(f"  Python files: {python_count}")
    print("  Configs: 20 (10 Dannehl Pix2PixCC + 10 Sayez I2IwFiLM)")
    print(
        "  YAML semantic parse: passed"
        if yaml_checked
        else "  YAML semantic parse: skipped (PyYAML unavailable)"
    )
    print("  Tensor/autograd/DDP execution: skipped (PyTorch unavailable)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Static debug validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
