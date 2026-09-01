"""Lightning classifier built from a pretrained SolarCHIP HMI encoder."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from auxiliary.clip.model import AttentionPool2d
from auxiliary.ldm.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from solarchip.modules.CNN import AE_CNN, VAE_CNN
from solarchip.modules.ViT import AE_ViT
from solarchip.utils.util import instantiate_from_config

from downstream.flare.data.class_groups import (
    DEFAULT_CLASS_GROUPS,
    build_raw_label_to_group,
    normalize_class_groups,
)


class PretrainedCheckpointError(RuntimeError):
    """Raised when a SolarCHIP checkpoint cannot be loaded without guessing."""


def _normalize_loss_type(loss_type: str) -> str:
    if not isinstance(loss_type, str):
        raise TypeError(f"loss_type must be a string, got {type(loss_type).__name__}")
    normalized = loss_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cross_entropy": "cross_entropy",
        "crossentropy": "cross_entropy",
        "ce": "cross_entropy",
        "focal": "focal",
        "focal_loss": "focal",
        "focalloss": "focal",
        "focoloss": "focal",
    }
    if normalized not in aliases:
        raise ValueError(
            f"loss_type must be 'cross_entropy' or 'focal', got {loss_type!r}"
        )
    return aliases[normalized]


def _plain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached plain-dict copy of a dict or OmegaConf config."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            resolved = OmegaConf.to_container(config, resolve=True)
            if not isinstance(resolved, dict):
                raise TypeError("base_model must resolve to a mapping")
            return resolved
    except ImportError:
        pass

    if not isinstance(config, Mapping):
        raise TypeError(f"base_model must be a mapping, got {type(config).__name__}")
    return copy.deepcopy(dict(config))


def _load_tensor_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load a Lightning or raw tensor state dict using PyTorch's safe loader."""

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        raise PretrainedCheckpointError(
            f"Cannot load pretrained checkpoint {checkpoint_path}: {error}"
        ) from error

    if not isinstance(payload, Mapping):
        raise PretrainedCheckpointError(
            f"Checkpoint {checkpoint_path} must contain a mapping, "
            f"got {type(payload).__name__}"
        )

    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise PretrainedCheckpointError(
            f"Checkpoint {checkpoint_path} does not contain a non-empty state dict"
        )
    if any(not isinstance(key, str) for key in state):
        raise PretrainedCheckpointError("Every checkpoint state key must be a string")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise PretrainedCheckpointError(
            "A raw checkpoint mapping may contain tensors only; pass a Lightning "
            "checkpoint with its weights under 'state_dict'"
        )

    keys = list(state)
    module_prefixed = [key.startswith("module.") for key in keys]
    if any(module_prefixed) and not all(module_prefixed):
        raise PretrainedCheckpointError(
            "Checkpoint mixes 'module.'-prefixed and unprefixed state keys"
        )
    if all(module_prefixed):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


def _substate(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _strict_load(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    component: str,
) -> None:
    if not state:
        raise PretrainedCheckpointError(
            f"Checkpoint contains no parameters for required component {component!r}"
        )
    try:
        module.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise PretrainedCheckpointError(
            f"Pretrained {component} does not strictly match base_model config: {error}"
        ) from error


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith(".") else f"{prefix}."


def _select_checkpoint_layout(
    state: Mapping[str, torch.Tensor],
    encoder_keys: set[str],
    pretrained_prefix: str | None,
    allow_shared_all: bool,
) -> tuple[str, str | None]:
    """Select one unambiguous SolarCHIP, raw-AE, or raw-encoder layout."""

    if pretrained_prefix is not None:
        root = _normalize_prefix(pretrained_prefix)
        allowed_namespaced_roots = {"model_dict.hmi.", "model_dict.all."}
        if root.startswith("model_dict.") and root not in allowed_namespaced_roots:
            raise PretrainedCheckpointError(
                "A namespaced SolarCHIP checkpoint prefix must select "
                "model_dict.hmi; model_dict.all additionally requires the explicit "
                "shared-encoder opt-in"
            )
        if root == "model_dict.all." and not allow_shared_all:
            raise PretrainedCheckpointError(
                "model_dict.all is shared across modalities; set allow_shared_all=True "
                "only when that shared encoder is intentionally used as HMI"
            )
        if not any(key.startswith(root) for key in state):
            raise PretrainedCheckpointError(
                f"No checkpoint keys start with explicit prefix {root!r}"
            )
        if root == "" and set(state) == encoder_keys:
            return "raw_encoder", None
        return "autoencoder", root

    candidates: list[tuple[str, str | None]] = []
    if any(key.startswith("model_dict.hmi.") for key in state):
        candidates.append(("autoencoder", "model_dict.hmi."))
    if any(key.startswith("model_dict.all.") for key in state):
        if allow_shared_all:
            candidates.append(("autoencoder", "model_dict.all."))
        elif not candidates:
            raise PretrainedCheckpointError(
                "Checkpoint only exposes model_dict.all, which is a shared-modal "
                "encoder. Set allow_shared_all=True to opt in explicitly."
            )
    if any(key.startswith("encoder.") for key in state):
        candidates.append(("autoencoder", ""))
    if set(state) == encoder_keys:
        candidates.append(("raw_encoder", None))

    if len(candidates) != 1:
        raise PretrainedCheckpointError(
            "Could not select one unambiguous HMI checkpoint layout. "
            f"Detected candidates: {candidates or 'none'}"
        )
    return candidates[0]


class SolarPredictor(pl.LightningModule):
    """Classify daily flares using only the pretrained SolarCHIP HMI branch.

    The backbone-specific main mapping produces a 256-dimensional feature:

    * CNN: spatial encoder output -> independent attention pooling.
    * ViT: raw encoder CLS token -> learned linear mapping.

    An optional pretrained SolarCHIP contrastive global feature is mapped to the
    same width and added through a learnable residual gate. No decoder or
    non-HMI modality is registered in the final model.
    """

    def __init__(
        self,
        base_model: Mapping[str, Any],
        pretrained_ckpt_path: str | Path,
        class_groups: Sequence[str] | None = DEFAULT_CLASS_GROUPS,
        representation_dim: int = 256,
        head_hidden_dim: int = 256,
        dropout: float = 0.2,
        use_contrastive_residual: bool = True,
        contrastive_gate_init: float = 0.1,
        attention_heads: int = 8,
        pretrained_prefix: str | None = None,
        allow_shared_all: bool = False,
        learning_rate: float = 1e-4,
        encoder_learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        class_weights: Sequence[float] | None = None,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        metric_class_ids: Sequence[int] | None = None,
        train_backbone: bool = True,
        freeze_encoder_epochs: int = 0,
        scheduler: str = "cosine",
        max_epochs: int = 200,
        min_learning_rate: float = 0.0,
    ) -> None:
        super().__init__()

        resolved_class_groups = normalize_class_groups(class_groups)
        num_classes = len(resolved_class_groups)
        if representation_dim <= 0 or head_hidden_dim <= 0:
            raise ValueError("representation_dim and head_hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if not isinstance(train_backbone, bool):
            raise TypeError("train_backbone must be a boolean")
        if freeze_encoder_epochs < 0:
            raise ValueError("freeze_encoder_epochs cannot be negative")
        if learning_rate <= 0 or encoder_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative")

        self.loss_type = _normalize_loss_type(loss_type)
        self.focal_gamma = float(focal_gamma)
        if not math.isfinite(self.focal_gamma) or self.focal_gamma < 0:
            raise ValueError("focal_gamma must be finite and non-negative")

        resolved_base_model = _plain_config(base_model)
        params = resolved_base_model.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError("base_model.params must be a mapping")
        # The outer SolarCHIP checkpoint is loaded below with an explicit prefix.
        # Prevent the inner AE constructor from trying to load it a second time.
        params["ckpt_path"] = None

        checkpoint_path = Path(pretrained_ckpt_path).expanduser().resolve()
        pretrained_ae = instantiate_from_config(resolved_base_model)
        if isinstance(pretrained_ae, AE_CNN):
            self.backbone_kind = "cnn"
            self._is_variational_cnn = False
        elif isinstance(pretrained_ae, VAE_CNN):
            self.backbone_kind = "cnn"
            self._is_variational_cnn = True
        elif isinstance(pretrained_ae, AE_ViT):
            self.backbone_kind = "vit"
            self._is_variational_cnn = False
        else:
            raise TypeError(
                "base_model target must instantiate AE_CNN, VAE_CNN, or AE_ViT; "
                f"got {type(pretrained_ae).__name__}"
            )

        checkpoint_state: dict[str, torch.Tensor] | None = None
        layout: str | None = None
        root: str | None = None
        self._pretrained_weights_loaded = checkpoint_path.is_file()
        self._restored_from_downstream_checkpoint = False
        if self._pretrained_weights_loaded:
            checkpoint_state = _load_tensor_state_dict(checkpoint_path)
            encoder_keys = set(pretrained_ae.encoder.state_dict())
            layout, root = _select_checkpoint_layout(
                checkpoint_state,
                encoder_keys=encoder_keys,
                pretrained_prefix=pretrained_prefix,
                allow_shared_all=allow_shared_all,
            )
            if layout == "raw_encoder":
                encoder_state = checkpoint_state
            else:
                assert root is not None
                encoder_state = _substate(checkpoint_state, f"{root}encoder.")
            _strict_load(pretrained_ae.encoder, encoder_state, "HMI encoder")
        self.hmi_encoder = pretrained_ae.encoder

        self.hmi_quant_conv: nn.Module | None = None
        if self._is_variational_cnn:
            if checkpoint_state is not None:
                if layout == "raw_encoder":
                    raise PretrainedCheckpointError(
                        "A bare VAE encoder checkpoint lacks the required quant_conv"
                    )
                assert root is not None
                quant_state = _substate(checkpoint_state, f"{root}quant_conv.")
                _strict_load(pretrained_ae.quant_conv, quant_state, "HMI quant_conv")
            self.hmi_quant_conv = pretrained_ae.quant_conv

        self.class_groups = resolved_class_groups
        self.num_classes = num_classes
        raw_label_to_group = build_raw_label_to_group(self.class_groups)
        self.register_buffer(
            "raw_label_to_group",
            torch.tensor(
                [raw_label_to_group[label] for label in range(6)],
                dtype=torch.long,
            ),
            persistent=True,
        )
        self.representation_dim = int(representation_dim)
        self.use_contrastive_residual = bool(use_contrastive_residual)
        self.train_backbone = train_backbone
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.learning_rate = float(learning_rate)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler_name = scheduler.lower()
        self.max_epochs = int(max_epochs)
        self.min_learning_rate = float(min_learning_rate)

        self.cnn_cls_proj: nn.Module | None = None
        self.contrastive_projector: nn.Parameter | None = None

        if self.backbone_kind == "cnn":
            feature_dim = int(pretrained_ae.feature_dim)
            feature_size = int(pretrained_ae.feature_size)
            if feature_dim % attention_heads != 0:
                raise ValueError(
                    f"CNN feature_dim={feature_dim} must be divisible by "
                    f"attention_heads={attention_heads}"
                )
            self.main_mapper = AttentionPool2d(
                spacial_dim=feature_size,
                embed_dim=feature_dim,
                num_heads=attention_heads,
                output_dim=representation_dim,
            )
            raw_feature_dim = feature_dim
        else:
            raw_feature_dim = int(pretrained_ae.encoder.conv1.out_channels)
            self.main_mapper = nn.Sequential(
                nn.LayerNorm(raw_feature_dim),
                nn.Linear(raw_feature_dim, representation_dim),
            )

        if self.use_contrastive_residual:
            if checkpoint_state is not None:
                if layout == "raw_encoder":
                    raise PretrainedCheckpointError(
                        "use_contrastive_residual=True requires a full AE checkpoint"
                    )
                assert root is not None
                projector_key = f"{root}contrasive_porject"
                if projector_key not in checkpoint_state:
                    raise PretrainedCheckpointError(
                        f"Checkpoint is missing pretrained {projector_key!r}"
                    )
                projector = checkpoint_state[projector_key]
            else:
                # This random value is only an architecture placeholder. Direct
                # use is blocked below; a downstream Lightning checkpoint must
                # restore the actual learned tensor before forward/training.
                projector = pretrained_ae.contrasive_porject.detach().clone()
            expected_projector_shape = tuple(pretrained_ae.contrasive_porject.shape)
            if tuple(projector.shape) != expected_projector_shape:
                raise PretrainedCheckpointError(
                    f"Contrastive projector shape {tuple(projector.shape)} does not "
                    f"match base_model shape {expected_projector_shape}"
                )
            if projector.shape[0] != raw_feature_dim:
                raise PretrainedCheckpointError(
                    "Contrastive projector input dimension does not match encoder"
                )

            if self.backbone_kind == "cnn":
                if checkpoint_state is not None:
                    assert root is not None
                    cls_state = _substate(checkpoint_state, f"{root}cls_proj.")
                    _strict_load(pretrained_ae.cls_proj, cls_state, "HMI CNN cls_proj")
                self.cnn_cls_proj = pretrained_ae.cls_proj

            self.contrastive_projector = nn.Parameter(projector.detach().clone())
            contrastive_dim = int(projector.shape[1])
            self.contrastive_adapter = nn.Sequential(
                nn.LayerNorm(contrastive_dim),
                nn.Linear(contrastive_dim, representation_dim, bias=False),
            )
            self.contrastive_gate = nn.Parameter(
                torch.tensor(float(contrastive_gate_init), dtype=torch.float32)
            )
        else:
            self.contrastive_adapter = None
            self.register_parameter("contrastive_gate", None)

        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        if not self.train_backbone:
            for parameter in self._pretrained_parameters():
                parameter.requires_grad_(False)
            for module in self._pretrained_modules():
                module.eval()

        resolved_class_weights = self._validate_class_weights(
            class_weights, num_classes
        )
        self.register_buffer(
            "class_weights",
            resolved_class_weights,
            persistent=True,
        )

        resolved_metric_ids = (
            list(range(num_classes))
            if metric_class_ids is None
            else [int(value) for value in metric_class_ids]
        )
        if not resolved_metric_ids or len(set(resolved_metric_ids)) != len(
            resolved_metric_ids
        ):
            raise ValueError("metric_class_ids must contain unique class IDs")
        if any(value < 0 or value >= num_classes for value in resolved_metric_ids):
            raise ValueError("metric_class_ids contains an ID outside num_classes")
        self.register_buffer(
            "metric_class_ids",
            torch.tensor(resolved_metric_ids, dtype=torch.long),
            persistent=True,
        )
        # Keep train-epoch metric computation rank-local. A distributed
        # compute here runs after Lightning's rank-zero progress bar may have
        # started resolving sync_dist train_loss, which can invert collective
        # ordering across ranks. Validation/test remain globally synchronized
        # because they drive checkpoint selection and reported evaluation.
        self.train_confusion = MulticlassConfusionMatrix(
            num_classes=num_classes,
            sync_on_compute=False,
        )
        self.val_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

        self.save_hyperparameters(
            {
                "base_model": resolved_base_model,
                "pretrained_ckpt_path": str(checkpoint_path),
                "class_groups": list(self.class_groups),
                "representation_dim": representation_dim,
                "head_hidden_dim": head_hidden_dim,
                "dropout": dropout,
                "use_contrastive_residual": use_contrastive_residual,
                "contrastive_gate_init": contrastive_gate_init,
                "attention_heads": attention_heads,
                "pretrained_prefix": pretrained_prefix,
                "allow_shared_all": allow_shared_all,
                "learning_rate": learning_rate,
                "encoder_learning_rate": encoder_learning_rate,
                "weight_decay": weight_decay,
                "class_weights": None
                if class_weights is None
                else [float(value) for value in class_weights],
                "loss_type": self.loss_type,
                "focal_gamma": self.focal_gamma,
                "metric_class_ids": resolved_metric_ids,
                "train_backbone": self.train_backbone,
                "freeze_encoder_epochs": freeze_encoder_epochs,
                "scheduler": scheduler,
                "max_epochs": max_epochs,
                "min_learning_rate": min_learning_rate,
            }
        )

        # Make the deletion contract explicit: the temporary autoencoder and its
        # decoder/non-HMI siblings are not registered by SolarPredictor.
        del pretrained_ae

    def _weights_are_available(self) -> bool:
        return (
            self._pretrained_weights_loaded or self._restored_from_downstream_checkpoint
        )

    def _require_weights(self) -> None:
        if not self._weights_are_available():
            checkpoint_path = self.hparams["pretrained_ckpt_path"]
            raise FileNotFoundError(
                "Pretrained SolarCHIP checkpoint is unavailable and no complete "
                "downstream SolarPredictor checkpoint has been restored. "
                f"Expected: {checkpoint_path}"
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["flare_class_groups"] = list(self.class_groups)
        checkpoint["flare_loss_config"] = {
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
            "reduction": "weighted_mean",
            "class_weights": None
            if self.class_weights is None
            else self.class_weights.detach().cpu().tolist(),
        }
        checkpoint["flare_optimization_config"] = {
            "train_backbone": self.train_backbone,
        }

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        saved_groups = checkpoint.get("flare_class_groups")
        if saved_groups is None:
            hyperparameters = checkpoint.get("hyper_parameters")
            if isinstance(hyperparameters, Mapping):
                saved_groups = hyperparameters.get("class_groups")
        if saved_groups is None:
            raise PretrainedCheckpointError(
                "Downstream checkpoint has no class_groups metadata and cannot be "
                "safely matched to this classifier"
            )
        try:
            normalized_saved_groups = normalize_class_groups(saved_groups)
        except (TypeError, ValueError) as error:
            raise PretrainedCheckpointError(
                f"Downstream checkpoint contains invalid class_groups: {error}"
            ) from error
        if normalized_saved_groups != self.class_groups:
            raise PretrainedCheckpointError(
                "Downstream checkpoint class_groups do not match the current model: "
                f"{list(normalized_saved_groups)} != {list(self.class_groups)}"
            )

        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise PretrainedCheckpointError(
                "A self-contained SolarPredictor restore requires a Lightning "
                "checkpoint with a 'state_dict' mapping"
            )

        saved_optimization_config = checkpoint.get("flare_optimization_config")
        if saved_optimization_config is None:
            hyperparameters = checkpoint.get("hyper_parameters")
            if not isinstance(hyperparameters, Mapping):
                hyperparameters = {}
            # SolarPredictor checkpoints created before this option always
            # trained the backbone, so True is the only safe legacy meaning.
            saved_train_backbone = hyperparameters.get("train_backbone", True)
        elif isinstance(saved_optimization_config, Mapping):
            saved_train_backbone = saved_optimization_config.get("train_backbone")
        else:
            raise PretrainedCheckpointError(
                "Downstream checkpoint contains invalid "
                "flare_optimization_config metadata"
            )
        if not isinstance(saved_train_backbone, bool):
            raise PretrainedCheckpointError(
                "Downstream checkpoint contains invalid train_backbone metadata"
            )
        if saved_train_backbone != self.train_backbone:
            raise PretrainedCheckpointError(
                "Downstream checkpoint train_backbone does not match the current "
                f"model: {saved_train_backbone!r} != {self.train_backbone!r}"
            )

        saved_loss_config = checkpoint.get("flare_loss_config")
        if saved_loss_config is None:
            hyperparameters = checkpoint.get("hyper_parameters")
            if not isinstance(hyperparameters, Mapping):
                hyperparameters = {}
            saved_loss_type = hyperparameters.get("loss_type", "cross_entropy")
            saved_focal_gamma = hyperparameters.get("focal_gamma")
            saved_reduction = "weighted_mean"
        elif isinstance(saved_loss_config, Mapping):
            saved_loss_type = saved_loss_config.get("loss_type")
            saved_focal_gamma = saved_loss_config.get("focal_gamma")
            saved_reduction = saved_loss_config.get("reduction")
        else:
            raise PretrainedCheckpointError(
                "Downstream checkpoint contains invalid flare_loss_config metadata"
            )

        try:
            normalized_saved_loss_type = _normalize_loss_type(saved_loss_type)
        except (TypeError, ValueError) as error:
            raise PretrainedCheckpointError(
                f"Downstream checkpoint contains invalid loss_type: {error}"
            ) from error
        if saved_reduction != "weighted_mean":
            raise PretrainedCheckpointError(
                "Downstream checkpoint loss reduction must be 'weighted_mean'"
            )
        if normalized_saved_loss_type != self.loss_type:
            raise PretrainedCheckpointError(
                "Downstream checkpoint loss_type does not match the current model: "
                f"{normalized_saved_loss_type!r} != {self.loss_type!r}"
            )
        if self.loss_type == "focal":
            try:
                normalized_saved_gamma = float(saved_focal_gamma)
            except (TypeError, ValueError) as error:
                raise PretrainedCheckpointError(
                    "Downstream focal checkpoint has no valid focal_gamma"
                ) from error
            if (
                not math.isfinite(normalized_saved_gamma)
                or normalized_saved_gamma != self.focal_gamma
            ):
                raise PretrainedCheckpointError(
                    "Downstream checkpoint focal_gamma does not match the current "
                    f"model: {normalized_saved_gamma!r} != {self.focal_gamma!r}"
                )

        saved_class_weights = state.get("class_weights")
        if self.class_weights is None:
            class_weights_match = saved_class_weights is None
        else:
            class_weights_match = (
                torch.is_tensor(saved_class_weights)
                and tuple(saved_class_weights.shape) == tuple(self.class_weights.shape)
                and torch.equal(
                    saved_class_weights.detach().cpu(),
                    self.class_weights.detach().cpu(),
                )
            )
        if not class_weights_match:
            raise PretrainedCheckpointError(
                "Downstream checkpoint class_weights do not match the current model"
            )

        expected_state = self.state_dict()
        missing = sorted(set(expected_state).difference(state))
        wrong_shapes = sorted(
            key
            for key, expected in expected_state.items()
            if key in state
            and (
                not torch.is_tensor(state[key])
                or tuple(state[key].shape) != tuple(expected.shape)
            )
        )
        if missing or wrong_shapes:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:8]}")
            if wrong_shapes:
                details.append(f"wrong-shape keys: {wrong_shapes[:8]}")
            raise PretrainedCheckpointError(
                "Downstream checkpoint is not a complete SolarPredictor state ("
                + "; ".join(details)
                + ")"
            )
        saved_mapping = state.get("raw_label_to_group")
        if not torch.equal(
            saved_mapping.detach().cpu(), self.raw_label_to_group.detach().cpu()
        ):
            raise PretrainedCheckpointError(
                "Downstream checkpoint raw-label grouping does not match the "
                "current model class_groups"
            )
        saved_metric_ids = state.get("metric_class_ids")
        if not torch.equal(
            saved_metric_ids.detach().cpu(), self.metric_class_ids.detach().cpu()
        ):
            raise PretrainedCheckpointError(
                "Downstream checkpoint metric_class_ids do not match the current "
                "model"
            )
        self._restored_from_downstream_checkpoint = True

    @staticmethod
    def _collect_dataset_class_groups(
        source: Any,
    ) -> list[tuple[str, tuple[str, ...]]]:
        declarations: list[tuple[str, tuple[str, ...]]] = []
        seen: set[int] = set()

        def visit(value: Any, path: str) -> None:
            if value is None or id(value) in seen:
                return
            seen.add(id(value))

            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(item, f"{path}.{key}")
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
                return

            declared_groups = getattr(value, "class_groups", None)
            if declared_groups is not None:
                try:
                    normalized = normalize_class_groups(declared_groups)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Dataset {path} has invalid class_groups: {error}"
                    ) from error
                declarations.append((path, normalized))
                return

            for attribute in ("dataset", "datasets"):
                if hasattr(value, attribute):
                    visit(getattr(value, attribute), f"{path}.{attribute}")
            if value.__class__.__name__ == "WrappedDataset" and hasattr(value, "data"):
                visit(value.data, f"{path}.data")

        visit(source, "data")
        return declarations

    def _validate_dataset_class_groups(self, require_declaration: bool = True) -> bool:
        datamodule = getattr(self.trainer, "datamodule", None)
        datamodule_datasets = getattr(datamodule, "datasets", None)
        if datamodule_datasets:
            source_items = list(datamodule_datasets.items())
        else:
            source_items = list(
                {
                    "train": getattr(self.trainer, "train_dataloader", None),
                    "validation": getattr(self.trainer, "val_dataloaders", None),
                    "test": getattr(self.trainer, "test_dataloaders", None),
                    "predict": getattr(self.trainer, "predict_dataloaders", None),
                }.items()
            )

        declarations: list[tuple[str, tuple[str, ...]]] = []
        missing_declarations: list[str] = []
        for split, source in source_items:
            if source is None:
                continue
            split_declarations = self._collect_dataset_class_groups(source)
            if not split_declarations and split != "predict":
                missing_declarations.append(str(split))
            declarations.extend(
                (f"{split}:{path}", groups) for path, groups in split_declarations
            )

        if missing_declarations:
            raise RuntimeError(
                "Cannot verify SolarPredictor class_groups because these attached "
                "datasets do not expose class_groups: "
                f"{missing_declarations}"
            )
        if not declarations:
            if require_declaration:
                raise RuntimeError(
                    "Cannot verify SolarPredictor class_groups because no attached "
                    "dataset exposes a class_groups attribute"
                )
            return False

        mismatches = [
            (path, groups)
            for path, groups in declarations
            if groups != self.class_groups
        ]
        if mismatches:
            details = "; ".join(f"{path}={list(groups)}" for path, groups in mismatches)
            raise ValueError(
                "SolarPredictor and dataset class_groups must match exactly. "
                f"model={list(self.class_groups)}; {details}"
            )
        return True

    def on_fit_start(self) -> None:
        self._require_weights()
        # DataModuleFromConfig.setup() has populated .datasets at this point.
        # Direct DataLoader inputs are not attached until later in the fit loop.
        self._validate_dataset_class_groups(require_declaration=False)

    def on_train_start(self) -> None:
        self._validate_dataset_class_groups()

    def on_validation_start(self) -> None:
        self._require_weights()
        self._validate_dataset_class_groups()

    def on_test_start(self) -> None:
        self._require_weights()
        self._validate_dataset_class_groups()

    def on_predict_start(self) -> None:
        self._require_weights()

    @staticmethod
    def _validate_class_weights(
        class_weights: Sequence[float] | None, num_classes: int
    ) -> torch.Tensor | None:
        if class_weights is None:
            return None
        weights = torch.tensor(list(class_weights), dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(
                f"class_weights must have {num_classes} values, got {weights.numel()}"
            )
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("class_weights must be finite and strictly positive")
        return weights

    def _pretrained_modules(self) -> list[nn.Module]:
        modules = [self.hmi_encoder]
        if self.hmi_quant_conv is not None:
            modules.append(self.hmi_quant_conv)
        if self.cnn_cls_proj is not None:
            modules.append(self.cnn_cls_proj)
        return modules

    def _backbone_is_frozen(self) -> bool:
        return (not self.train_backbone) or (
            self.training and self.current_epoch < self.freeze_encoder_epochs
        )

    def train(self, mode: bool = True) -> SolarPredictor:
        super().train(mode)
        if mode and self._backbone_is_frozen():
            for module in self._pretrained_modules():
                module.eval()
        return self

    def on_train_epoch_start(self) -> None:
        for module in self._pretrained_modules():
            module.train(not self._backbone_is_frozen())

    def _encode_hmi(self, hmi: torch.Tensor) -> torch.Tensor:
        latent = self.hmi_encoder(hmi)
        if self.hmi_quant_conv is not None:
            moments = self.hmi_quant_conv(latent)
            latent = DiagonalGaussianDistribution(moments).mode()
        return latent

    def _contrastive_feature(self, latent: torch.Tensor) -> torch.Tensor:
        if self.contrastive_projector is None:
            raise RuntimeError("Contrastive residual is disabled")
        if self.backbone_kind == "cnn":
            if self.cnn_cls_proj is None:
                raise RuntimeError("CNN contrastive cls_proj is unavailable")
            global_feature = self.cnn_cls_proj(latent)
        else:
            global_feature = latent[:, 0]
        return global_feature @ self.contrastive_projector

    def encode_features(self, hmi: torch.Tensor) -> torch.Tensor:
        """Map an HMI batch to the shared 256-dimensional representation."""

        self._require_weights()
        if hmi.ndim != 4:
            raise ValueError(f"Expected HMI [B,C,H,W], got shape {tuple(hmi.shape)}")

        backbone_frozen = self._backbone_is_frozen()
        if backbone_frozen:
            with torch.no_grad():
                latent = self._encode_hmi(hmi)
                contrastive = (
                    self._contrastive_feature(latent)
                    if self.use_contrastive_residual
                    else None
                )
            latent = latent.detach()
        else:
            latent = self._encode_hmi(hmi)
            contrastive = (
                self._contrastive_feature(latent)
                if self.use_contrastive_residual
                else None
            )

        if self.backbone_kind == "cnn":
            main_feature = self.main_mapper(latent)
        else:
            main_feature = self.main_mapper(latent[:, 0])

        if contrastive is None:
            return main_feature
        assert self.contrastive_adapter is not None
        residual = self.contrastive_adapter(contrastive)
        gate = torch.tanh(self.contrastive_gate)
        return main_feature + gate * residual

    def forward(self, hmi: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_features(hmi))

    def _classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(logits, labels, weight=self.class_weights)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        one_minus_pt = (-torch.expm1(log_pt)).clamp_min(torch.finfo(log_pt.dtype).eps)
        focal_factor = one_minus_pt.pow(self.focal_gamma)
        per_sample_loss = -focal_factor * log_pt

        if self.class_weights is None:
            return per_sample_loss.mean()

        sample_weights = self.class_weights[labels]
        return (sample_weights * per_sample_loss).sum() / sample_weights.sum()

    def _shared_step(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "hmi" not in batch or "label" not in batch:
            raise KeyError("Batch must contain 'hmi' and 'label' tensors")
        labels = batch["label"].long()
        if (
            labels.numel() == 0
            or (labels < 0).any()
            or (labels >= self.num_classes).any()
        ):
            raise ValueError(
                f"Grouped labels must be in [0, {self.num_classes - 1}] for "
                f"class_groups={list(self.class_groups)}"
            )
        logits = self(batch["hmi"].float())
        loss = self._classification_loss(logits, labels)
        return loss, logits, labels

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels = self._shared_step(batch)
        self.train_confusion.update(logits.detach(), labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        return loss

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels = self._shared_step(batch)
        self.val_confusion.update(logits, labels)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        return loss

    def test_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels = self._shared_step(batch)
        self.test_confusion.update(logits, labels)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        return loss

    def predict_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx
        logits = self(batch["hmi"].float())
        output = {
            "logits": logits,
            "prediction": logits.argmax(dim=-1),
        }
        if "date_id" in batch:
            output["date_id"] = batch["date_id"]
        return output

    def _log_confusion_metrics(
        self,
        confusion_metric: MulticlassConfusionMatrix,
        split: str,
    ) -> None:
        confusion = confusion_metric.compute().to(torch.float32)
        total = confusion.sum()
        if total <= 0:
            confusion_metric.reset()
            return

        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        recall = true_positive / support.clamp_min(1.0)
        precision = true_positive / predicted.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        active_ids = self.metric_class_ids

        metrics = {
            f"{split}_accuracy": true_positive.sum() / total,
            f"{split}_balanced_accuracy": recall[active_ids].mean(),
            f"{split}_macro_f1": f1[active_ids].mean(),
        }
        if active_ids.numel() < self.num_classes:
            active_mask = torch.zeros(
                self.num_classes, dtype=torch.bool, device=confusion.device
            )
            active_mask[active_ids] = True
            metrics[f"{split}_ignored_prediction_rate"] = (
                predicted[~active_mask].sum() / total
            )

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False)
        confusion_metric.reset()

    def on_train_epoch_end(self) -> None:
        self._log_confusion_metrics(self.train_confusion, "train")

    def on_validation_epoch_end(self) -> None:
        self._log_confusion_metrics(self.val_confusion, "val")

    def on_test_epoch_end(self) -> None:
        self._log_confusion_metrics(self.test_confusion, "test")

    def _pretrained_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self._pretrained_modules():
            parameters.extend(module.parameters())
        if self.contrastive_projector is not None:
            parameters.append(self.contrastive_projector)

        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
        return unique

    def configure_optimizers(self):
        pretrained_parameters = self._pretrained_parameters()
        pretrained_ids = {id(parameter) for parameter in pretrained_parameters}
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in pretrained_ids and parameter.requires_grad
        ]
        parameter_groups = []
        if self.train_backbone:
            parameter_groups.append(
                {
                    "params": pretrained_parameters,
                    "lr": self.encoder_learning_rate,
                }
            )
        parameter_groups.append(
            {"params": new_parameters, "lr": self.learning_rate}
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name in {"none", ""}:
            return optimizer
        if self.scheduler_name != "cosine":
            raise ValueError(
                f"Unsupported scheduler {self.scheduler_name!r}; use 'cosine' or 'none'"
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.min_learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
