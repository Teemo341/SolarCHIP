"""PyTorch Lightning P-CNN adaptation for daily HMI flare prediction.

The spatial path follows Francisco et al. (2025): crop the polar half of a
full-disk magnetogram, resize to 224 x 448, split it into eight non-overlapping
112 x 112 patches, encode every patch with one shared ImageNet-pretrained
EfficientNetV2-S, and max-pool patch logits. SolarCHIP extends the original
separate C+ and M+ binary models with a third cumulative X+ head so that one
model can be decoded as 0AB/C/M/X.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import lightning.pytorch as pl
except ImportError:  # SolarCHIP currently uses the legacy package name.
    import pytorch_lightning as pl


_CLASS_NAMES = ("0AB", "C", "M", "X")
_HEAD_NAMES = ("c_plus", "m_plus", "x_plus")


class _CumulativePatchHead(nn.Module):
    """Paper-style BN -> dropout -> scalar classifier for one patch."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.dropout(self.batch_norm(features))).squeeze(-1)


class PCNN(pl.LightningModule):
    """HMI-only three-head P-CNN with patch-max multiple-instance learning.

    ``pretrained`` must remain true: the approved comparison contains only the
    ImageNet-pretrained EfficientNetV2-S experiment. ``train_class_counts`` is
    optional because it is normally derived from the attached train dataset at
    fit start; the derived counts and weights are stored in the checkpoint.
    """

    num_classes = 4
    num_heads = 3

    def __init__(
        self,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 1.0e-4,
        dropout: float = 0.2,
        max_epochs: int = 15,
        pretrained: bool = True,
        train_class_counts: Mapping[int, int] | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if not pretrained:
            raise ValueError(
                "PCNN only supports the approved ImageNet-pretrained baseline; "
                "the from-scratch ablation is intentionally disabled."
            )

        self.backbone, feature_dim = self._build_imagenet_backbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.patch_heads = nn.ModuleList(
            _CumulativePatchHead(feature_dim, dropout) for _ in range(self.num_heads)
        )
        self._freeze_backbone_batch_norm()

        self.register_buffer(
            "train_class_counts",
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        self.register_buffer(
            "loss_weight_table",
            torch.ones(self.num_heads, self.num_classes, dtype=torch.float32),
        )
        self.register_buffer(
            "train_counts_ready",
            torch.tensor(False, dtype=torch.bool),
        )

        # Epoch metrics are accumulated locally and reduced once per epoch.
        for split in ("train", "val", "test"):
            self.register_buffer(
                f"_{split}_class_confusion",
                torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_{split}_head_confusion",
                torch.zeros(self.num_heads, 2, 2, dtype=torch.long),
                persistent=False,
            )
            # p(C+) < p(M+), p(M+) < p(X+), and any threshold violation.
            self.register_buffer(
                f"_{split}_inconsistency_counts",
                torch.zeros(3, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_{split}_sample_count",
                torch.tensor(0, dtype=torch.long),
                persistent=False,
            )

        if train_class_counts is not None:
            self.set_train_class_counts(train_class_counts)

    @staticmethod
    def _build_imagenet_backbone() -> tuple[nn.Module, int]:
        try:
            from torchvision.models import (
                EfficientNet_V2_S_Weights,
                efficientnet_v2_s,
            )
        except Exception as error:
            raise RuntimeError(
                "P-CNN requires torchvision with efficientnet_v2_s and "
                "EfficientNet_V2_S_Weights.IMAGENET1K_V1."
            ) from error

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        try:
            network = efficientnet_v2_s(weights=weights)
        except Exception as error:
            raise RuntimeError(
                "Could not load the required torchvision EfficientNetV2-S "
                "IMAGENET1K_V1 weights. P-CNN will not silently use random "
                "weights or a fallback CNN. Cache/download "
                f"{weights.url!r} and retry."
            ) from error

        feature_dim = int(network.classifier[1].in_features)
        return network.features, feature_dim

    def _freeze_backbone_batch_norm(self) -> None:
        """Freeze both affine parameters and running statistics in the backbone."""

        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "PCNN":
        # nn.Module.train() recursively re-enables BN training, so restore the
        # paper's frozen-backbone-BN contract after every mode transition.
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @staticmethod
    def _normalise_counts(
        class_counts: Mapping[int, int] | Sequence[int],
    ) -> Tensor:
        if isinstance(class_counts, Mapping):
            try:
                values = [class_counts[index] for index in range(4)]
            except KeyError as error:
                raise ValueError(
                    "train class_counts mapping must contain integer keys 0,1,2,3"
                ) from error
        else:
            values = list(class_counts)
        if len(values) != 4:
            raise ValueError(
                f"train_class_counts must contain four values, got {len(values)}"
            )
        counts = torch.as_tensor(values, dtype=torch.long)
        if (counts < 0).any():
            raise ValueError("train_class_counts cannot contain negative values")
        if int(counts.sum()) == 0:
            raise ValueError("train_class_counts cannot be all zero")
        return counts

    def set_train_class_counts(
        self, class_counts: Mapping[int, int] | Sequence[int]
    ) -> dict[str, int]:
        """Set four-class train counts and derive all loss-weight rows.

        C+ and X+ receive inverse binary-frequency weights with equal total
        positive and negative contribution. M+ maps the paper's
        quiet/B/C/M/X ratio 2/2/1/8/8 to grouped 0AB/C/M/X = 2/1/8/8. Every
        row is normalized to expected weight one on the training distribution,
        and the three head losses are then averaged equally.
        """

        counts_cpu = self._normalise_counts(class_counts)
        counts = counts_cpu.to(device=self.train_class_counts.device)
        total = counts.sum().to(torch.float32)

        c_negative = counts[0].to(torch.float32)
        c_positive = counts[1:].sum().to(torch.float32)
        x_negative = counts[:3].sum().to(torch.float32)
        x_positive = counts[3].to(torch.float32)
        if min(
            float(c_negative),
            float(c_positive),
            float(x_negative),
            float(x_positive),
        ) <= 0:
            raise ValueError(
                "C+ and X+ balancing each require at least one positive and one "
                f"negative training example; got counts {counts_cpu.tolist()}"
            )

        c_weights = torch.stack(
            (
                total / (2.0 * c_negative),
                total / (2.0 * c_positive),
                total / (2.0 * c_positive),
                total / (2.0 * c_positive),
            )
        )
        # 0AB combines only classes assigned the original weight 2.
        m_weights = torch.tensor(
            (2.0, 1.0, 8.0, 8.0),
            device=counts.device,
            dtype=torch.float32,
        )
        x_weights = torch.stack(
            (
                total / (2.0 * x_negative),
                total / (2.0 * x_negative),
                total / (2.0 * x_negative),
                total / (2.0 * x_positive),
            )
        )
        table = torch.stack((c_weights, m_weights, x_weights))

        expected_weight = (
            (table * counts.to(torch.float32).unsqueeze(0)).sum(1) / total
        )
        table = table / expected_weight.unsqueeze(1)
        self.train_class_counts.copy_(counts)
        self.loss_weight_table.copy_(table)
        self.train_counts_ready.fill_(True)
        return {
            name: int(counts_cpu[index]) for index, name in enumerate(_CLASS_NAMES)
        }

    @classmethod
    def _find_dataset_class_counts(cls, source: Any) -> Mapping[int, int] | None:
        visited: set[int] = set()

        def visit(value: Any) -> Mapping[int, int] | None:
            if value is None or id(value) in visited:
                return None
            visited.add(id(value))
            counts = getattr(value, "class_counts", None)
            if counts is not None:
                return counts
            datasets = getattr(value, "datasets", None)
            if isinstance(datasets, Mapping):
                if "train" in datasets:
                    found = visit(datasets["train"])
                    if found is not None:
                        return found
                for dataset in datasets.values():
                    found = visit(dataset)
                    if found is not None:
                        return found
            elif datasets is not None:
                found = visit(datasets)
                if found is not None:
                    return found
            for attribute in ("dataset", "data"):
                if hasattr(value, attribute):
                    found = visit(getattr(value, attribute))
                    if found is not None:
                        return found
            return None

        return visit(source)

    def fetch_train_class_counts(self, source: Any | None = None) -> dict[str, int]:
        """Fetch counts from a DataModule/dataset, set weights, and return them."""

        if source is None:
            trainer = getattr(self, "_trainer", None)
            source = getattr(trainer, "datamodule", None)
        counts = self._find_dataset_class_counts(source)
        if counts is None:
            if bool(self.train_counts_ready):
                return {
                    name: int(self.train_class_counts[index].item())
                    for index, name in enumerate(_CLASS_NAMES)
                }
            raise RuntimeError(
                "P-CNN could not derive train class counts. Attach the existing "
                "SolarCHIP DataModule before fit, or pass train_class_counts "
                "explicitly when constructing the model."
            )
        return self.set_train_class_counts(counts)

    def _ensure_train_counts(self) -> None:
        if not bool(self.train_counts_ready):
            self.fetch_train_class_counts()

    def on_fit_start(self) -> None:
        self._ensure_train_counts()

    def on_validation_start(self) -> None:
        self._ensure_train_counts()

    def on_test_start(self) -> None:
        self._ensure_train_counts()

    @staticmethod
    def preprocess_hmi(hmi: Tensor) -> Tensor:
        """Return eight RGB patches with shape ``[B,8,3,112,112]``."""

        if hmi.ndim == 3:
            hmi = hmi.unsqueeze(1)
        if hmi.ndim != 4 or hmi.shape[1] != 1:
            raise ValueError(
                "P-CNN expects HMI shaped [B,1,H,W] (or [B,H,W]); "
                f"got {tuple(hmi.shape)}"
            )
        height, width = hmi.shape[-2:]
        crop_height = height // 2
        if crop_height < 1 or width < 1:
            raise ValueError(f"HMI spatial dimensions are too small: {(height, width)}")
        top = (height - crop_height) // 2
        hmi = hmi[..., top : top + crop_height, :].to(torch.float32)
        hmi = F.interpolate(
            hmi,
            size=(224, 448),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        # Keep SolarCHIP z-score values; do not apply ImageNet mean/std again.
        rgb = hmi.repeat(1, 3, 1, 1)
        patches = rgb.unfold(2, 112, 112).unfold(3, 112, 112)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(hmi.shape[0], 8, 3, 112, 112)

    def patch_logits(self, hmi: Tensor) -> Tensor:
        patches = self.preprocess_hmi(hmi)
        batch_size, patch_count = patches.shape[:2]
        feature_map = self.backbone(patches.flatten(0, 1))
        features = self.global_pool(feature_map).flatten(1)
        logits = torch.stack([head(features) for head in self.patch_heads], dim=-1)
        return logits.view(batch_size, patch_count, self.num_heads)

    def forward(self, hmi: Tensor) -> Tensor:
        """Return full-disk cumulative logits ``[B,3]`` for C+/M+/X+."""

        return self.patch_logits(hmi).amax(dim=1)

    @staticmethod
    def cumulative_targets(labels: Tensor) -> Tensor:
        labels = labels.long()
        return torch.stack(
            (labels >= 1, labels >= 2, labels >= 3), dim=1
        ).to(torch.float32)

    @staticmethod
    def decode_probabilities(probabilities: Tensor) -> Tensor:
        """Use the highest positive cumulative head; do not project monotonicity."""

        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(
                f"Expected cumulative probabilities [B,3], got {tuple(probabilities.shape)}"
            )
        positive = probabilities >= 0.5
        class_ids = torch.arange(
            1, 4, device=probabilities.device, dtype=torch.long
        ).unsqueeze(0)
        return (positive.to(torch.long) * class_ids).amax(dim=1)

    def _loss(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        self._ensure_train_counts()
        targets = self.cumulative_targets(labels)
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        sample_weights = self.loss_weight_table[:, labels].transpose(0, 1)
        head_losses = (element_loss * sample_weights).mean(dim=0)
        return head_losses.mean(), head_losses

    @staticmethod
    def _validate_labels(labels: Tensor) -> Tensor:
        labels = labels.long().view(-1)
        if labels.numel() == 0:
            raise ValueError("P-CNN received an empty label tensor")
        if int(labels.min()) < 0 or int(labels.max()) > 3:
            raise ValueError(
                f"P-CNN labels must be in 0..3, got [{int(labels.min())}, {int(labels.max())}]"
            )
        return labels

    def _update_epoch_metrics(
        self,
        split: str,
        labels: Tensor,
        targets: Tensor,
        probabilities: Tensor,
        predictions: Tensor,
    ) -> None:
        class_confusion = getattr(self, f"_{split}_class_confusion")
        class_indices = labels * self.num_classes + predictions
        class_confusion.add_(
            torch.bincount(
                class_indices, minlength=self.num_classes**2
            ).view(self.num_classes, self.num_classes)
        )

        binary_predictions = probabilities >= 0.5
        binary_targets = targets.to(torch.bool)
        head_confusion = getattr(self, f"_{split}_head_confusion")
        for head_index in range(self.num_heads):
            binary_indices = (
                binary_targets[:, head_index].long() * 2
                + binary_predictions[:, head_index].long()
            )
            head_confusion[head_index].add_(
                torch.bincount(binary_indices, minlength=4).view(2, 2)
            )

        c_lt_m = probabilities[:, 0] < probabilities[:, 1]
        m_lt_x = probabilities[:, 1] < probabilities[:, 2]
        action_violation = (
            (~binary_predictions[:, 0] & binary_predictions[:, 1])
            | (~binary_predictions[:, 1] & binary_predictions[:, 2])
        )
        inconsistency = getattr(self, f"_{split}_inconsistency_counts")
        inconsistency.add_(
            torch.stack((c_lt_m.sum(), m_lt_x.sum(), action_violation.sum())).long()
        )
        getattr(self, f"_{split}_sample_count").add_(labels.numel())

    def _shared_step(
        self, batch: Mapping[str, Tensor], split: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        if "hmi" not in batch or "label" not in batch:
            raise KeyError("P-CNN batch must contain both 'hmi' and 'label'")
        labels = self._validate_labels(batch["label"])
        logits = self(batch["hmi"])
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"HMI/label batch mismatch: {logits.shape[0]} != {labels.shape[0]}"
            )
        loss, head_losses = self._loss(logits, labels)
        probabilities = logits.sigmoid()
        predictions = self.decode_probabilities(probabilities)
        targets = self.cumulative_targets(labels)
        self._update_epoch_metrics(
            split, labels, targets, probabilities.detach(), predictions.detach()
        )

        is_train = split == "train"
        self.log(
            f"{split}_loss",
            loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.numel(),
        )
        for index, name in enumerate(_HEAD_NAMES):
            self.log(
                f"{split}_{name}_loss",
                head_losses[index],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=labels.numel(),
            )
        return loss, logits, predictions

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss, _, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss, _, _ = self._shared_step(batch, "test")
        return loss

    def predict_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        del batch_idx, dataloader_idx
        logits = self(batch["hmi"])
        probabilities = logits.sigmoid()
        output = {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": self.decode_probabilities(probabilities),
        }
        if "date_id" in batch:
            output["date_id"] = batch["date_id"]
        return output

    @staticmethod
    def _distributed_sum(value: Tensor) -> Tensor:
        value = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @staticmethod
    def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
        epsilon = torch.finfo(denominator.dtype).eps
        return torch.where(
            denominator.abs() > epsilon,
            numerator / denominator,
            torch.zeros_like(numerator),
        )

    def _log_epoch_metrics(self, split: str) -> None:
        confusion_buffer = getattr(self, f"_{split}_class_confusion")
        head_buffer = getattr(self, f"_{split}_head_confusion")
        inconsistency_buffer = getattr(self, f"_{split}_inconsistency_counts")
        count_buffer = getattr(self, f"_{split}_sample_count")

        confusion = self._distributed_sum(confusion_buffer).to(torch.float64)
        head_confusion = self._distributed_sum(head_buffer).to(torch.float64)
        inconsistency = self._distributed_sum(inconsistency_buffer).to(torch.float64)
        sample_count = self._distributed_sum(count_buffer).to(torch.float64)
        if float(sample_count) <= 0:
            return

        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted_count = confusion.sum(dim=0)
        recall = self._safe_ratio(true_positive, support)
        precision = self._safe_ratio(true_positive, predicted_count)
        f1 = self._safe_ratio(2.0 * precision * recall, precision + recall)
        metrics: dict[str, Tensor] = {
            f"{split}_accuracy": true_positive.sum() / sample_count,
            f"{split}_macro_f1": f1.mean(),
            f"{split}_balanced_accuracy": recall.mean(),
            f"{split}_prob_c_lt_m_rate": inconsistency[0] / sample_count,
            f"{split}_prob_m_lt_x_rate": inconsistency[1] / sample_count,
            f"{split}_threshold_inconsistency_rate": inconsistency[2] / sample_count,
        }

        for index, name in enumerate(_HEAD_NAMES):
            matrix = head_confusion[index]
            tn, fp = matrix[0, 0], matrix[0, 1]
            fn, tp = matrix[1, 0], matrix[1, 1]
            total = tp + tn + fp + fn
            sensitivity = self._safe_ratio(tp, tp + fn)
            false_positive_rate = self._safe_ratio(fp, fp + tn)
            hss_numerator = 2.0 * (tp * tn - fp * fn)
            hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
            mcc_denominator = torch.sqrt(
                ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp_min(1.0)
            )
            metrics.update(
                {
                    f"{split}_{name}_accuracy": self._safe_ratio(tp + tn, total),
                    f"{split}_{name}_tss": sensitivity - false_positive_rate,
                    f"{split}_{name}_hss": self._safe_ratio(
                        hss_numerator, hss_denominator
                    ),
                    f"{split}_{name}_mcc": (tp * tn - fp * fn) / mcc_denominator,
                    f"{split}_{name}_f1": self._safe_ratio(
                        2.0 * tp, 2.0 * tp + fp + fn
                    ),
                    f"{split}_{name}_recall": sensitivity,
                    f"{split}_{name}_far": self._safe_ratio(fp, tp + fp),
                }
            )

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        confusion_buffer.zero_()
        head_buffer.zero_()
        inconsistency_buffer.zero_()
        count_buffer.zero_()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Paper protocol: AdamW, fixed LR, no scheduler, exactly 15 epochs.
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
