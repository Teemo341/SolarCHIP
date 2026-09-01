"""PyTorch Lightning training module for the HMI-only DeepSWM adaptation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - compatibility with repository envs
    import pytorch_lightning as pl

from .model import HMIOnlyDeepSWM


CLASS_NAMES = ("0AB", "C", "M", "X")
NUM_CLASSES = len(CLASS_NAMES)


def inverse_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Return inverse-frequency weights normalized to mean one."""

    counts = counts.to(torch.float64)
    if counts.shape != (NUM_CLASSES,):
        raise ValueError(f"expected {NUM_CLASSES} class counts, got {tuple(counts.shape)}")
    if not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError(
            "every training class must have at least one sample to define "
            f"inverse-frequency weights; got {counts.tolist()}"
        )
    weights = counts.reciprocal()
    return (weights / weights.mean()).to(torch.float32)


def gerrity_score_matrix(class_probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the ordered-category Gandin-Murphy-Gerrity score matrix.

    The implementation follows the Gerrity construction used by the paper's
    metric code. Probabilities must come from the training split only.
    """

    probabilities = class_probabilities.to(torch.float64)
    if probabilities.shape != (NUM_CLASSES,):
        raise ValueError(
            f"expected {NUM_CLASSES} class probabilities, got {probabilities.shape}"
        )
    if not torch.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError(
            "GMGS requires non-zero finite training probability for every class"
        )
    probabilities = probabilities / probabilities.sum()
    # CUDA cumsum has no deterministic implementation.  There are only four
    # ordered classes, so form the three prefix sums explicitly and keep the
    # trainer's global deterministic-algorithm guarantee intact.
    cumulative_values = []
    running_probability = probabilities.new_zeros(())
    for probability in probabilities[:-1]:
        running_probability = running_probability + probability
        cumulative_values.append(running_probability)
    cumulative = torch.stack(cumulative_values)
    odds = (1.0 - cumulative) / cumulative

    score = torch.empty(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.float64,
        device=probabilities.device,
    )
    denominator = float(NUM_CLASSES - 1)
    for true_class in range(NUM_CLASSES):
        for predicted_class in range(NUM_CLASSES):
            lower = min(true_class, predicted_class)
            upper = max(true_class, predicted_class)
            inverse_sum = (1.0 / odds[:lower]).sum()
            upper_sum = odds[upper:].sum()
            score[true_class, predicted_class] = (
                inverse_sum - float(upper - lower) + upper_sum
            ) / denominator
    return score.to(torch.float32)


class SplitAccumulator:
    """Rank-local sufficient statistics with explicit epoch-end reduction.

    These tensors deliberately are not registered module buffers. PyTorch DDP
    may broadcast registered buffers before forward calls, which would corrupt
    rank-local running statistics before their final SUM reduction.
    """

    def __init__(self) -> None:
        self.confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        self.mplus_brier_sum = torch.zeros((), dtype=torch.float64)
        self.sample_count = torch.zeros((), dtype=torch.long)
        self.records: list[tuple[int, int, tuple[float, ...]]] = []

    def _move_to(self, device: torch.device) -> None:
        if self.confusion.device == device:
            return
        self.confusion = self.confusion.to(device)
        self.mplus_brier_sum = self.mplus_brier_sum.to(device)
        self.sample_count = self.sample_count.to(device)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        date_ids: torch.Tensor | None = None,
    ) -> None:
        self._move_to(labels.device)
        probabilities = logits.float().softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        encoded = labels * NUM_CLASSES + predictions
        self.confusion.add_(
            torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
                NUM_CLASSES, NUM_CLASSES
            )
        )
        event_probability = probabilities[:, 2:].sum(dim=-1)
        event_target = (labels >= 2).to(event_probability.dtype)
        self.mplus_brier_sum.add_(
            (event_probability - event_target).square().sum().to(torch.float64)
        )
        self.sample_count.add_(labels.numel())
        if date_ids is not None:
            flat_date_ids = date_ids.reshape(-1)
            if flat_date_ids.numel() != labels.numel():
                raise ValueError(
                    "date_id count must match labels for distributed metric "
                    f"deduplication, got {flat_date_ids.numel()} and {labels.numel()}"
                )
            self.records.extend(
                (
                    int(date_id),
                    int(label),
                    tuple(float(value) for value in probability),
                )
                for date_id, label, probability in zip(
                    flat_date_ids.detach().cpu().tolist(),
                    labels.detach().cpu().tolist(),
                    probabilities.detach().cpu().tolist(),
                )
            )

    @staticmethod
    def _statistics_from_unique_records(
        records: Sequence[tuple[int, int, tuple[float, ...]]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grouped: dict[int, tuple[int, list[tuple[float, ...]]]] = {}
        for date_id, label, probabilities in records:
            if len(probabilities) != NUM_CLASSES:
                raise ValueError(
                    f"expected {NUM_CLASSES} probabilities, got {len(probabilities)}"
                )
            if date_id in grouped:
                existing_label, values = grouped[date_id]
                if existing_label != label:
                    raise RuntimeError(
                        f"date_id {date_id} has conflicting labels "
                        f"{existing_label} and {label}"
                    )
                values.append(probabilities)
            else:
                grouped[date_id] = (label, [probabilities])

        confusion = torch.zeros(
            NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device
        )
        mplus_brier_sum = torch.zeros((), dtype=torch.float64, device=device)
        for label, probability_rows in grouped.values():
            mean_probability = torch.tensor(
                probability_rows, dtype=torch.float64, device=device
            ).mean(dim=0)
            prediction = int(mean_probability.argmax())
            confusion[label, prediction] += 1
            event_probability = mean_probability[2:].sum()
            event_target = float(label >= 2)
            mplus_brier_sum += (event_probability - event_target).square()
        sample_count = torch.tensor(len(grouped), dtype=torch.long, device=device)
        return confusion, mplus_brier_sum, sample_count

    @torch.no_grad()
    def synchronized(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return global sufficient statistics without mutating local state."""

        if device is not None:
            self._move_to(device)
        if self.records:
            records = list(self.records)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered: list[
                    list[tuple[int, int, tuple[float, ...]]] | None
                ] = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, records)
                records = [
                    record
                    for rank_records in gathered
                    if rank_records is not None
                    for record in rank_records
                ]
            return self._statistics_from_unique_records(
                records, device=self.confusion.device
            )
        confusion = self.confusion.detach().clone()
        mplus_brier_sum = self.mplus_brier_sum.detach().clone()
        sample_count = self.sample_count.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for statistic in (confusion, mplus_brier_sum, sample_count):
                torch.distributed.all_reduce(
                    statistic, op=torch.distributed.ReduceOp.SUM
                )
        return confusion, mplus_brier_sum, sample_count

    @torch.no_grad()
    def reset(self) -> None:
        self.confusion.zero_()
        self.mplus_brier_sum.zero_()
        self.sample_count.zero_()
        self.records.clear()


class DeepSWM(pl.LightningModule):
    """Trainable DeepSWM-derived daily HMI-only four-class classifier.

    Stage 1 jointly trains every branch for 20 epochs. At epoch 20, stage 2
    freezes SSE, the encoder-only SparseMAE branch, LT-SSM and mixing SSM, and
    trains only the classification head for 15 further epochs. Loss components
    can use inverse-frequency weights calculated exclusively from the attached
    training split, or unit weights for a true unweighted loss ablation.
    """

    def __init__(
        self,
        window_length: int = 1,
        image_size: int = 256,
        learning_rate: float = 4e-5,
        weight_decay: float = 0.05,
        adam_betas: Sequence[float] = (0.9, 0.95),
        stage1_epochs: int = 20,
        stage2_epochs: int = 15,
        max_epochs: int | None = None,
        ce_weight: float = 1.0,
        gmgs_weight: float = 1.0,
        bss_weight: float = 2.0,
        class_weight_mode: str = "inverse_frequency",
        train_class_counts: Sequence[int] | Mapping[int, int] | None = None,
        dim: int = 64,
        sequence_length: int = 128,
        sparse_embed_dim: int = 128,
        sparse_depth: int = 8,
        sparse_patch_size: int = 8,
        lt_depth: int = 1,
        mixing_depth: int = 1,
        sse_dropout: float = 0.6,
        dcsm_dropout: float = 0.6,
        stssm_dropout: float = 0.6,
        ltssm_dropout: float = 0.6,
        mixing_dropout: float = 0.6,
        head_dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if stage1_epochs < 1 or stage2_epochs < 1:
            raise ValueError("both training stages must contain at least one epoch")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if len(adam_betas) != 2 or not all(0 <= float(beta) < 1 for beta in adam_betas):
            raise ValueError("adam_betas must contain two values in [0,1)")
        for name, value in {
            "ce_weight": ce_weight,
            "gmgs_weight": gmgs_weight,
            "bss_weight": bss_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        class_weight_mode = str(class_weight_mode).strip().lower()
        if class_weight_mode not in {"none", "inverse_frequency"}:
            raise ValueError(
                "class_weight_mode must be 'none' or 'inverse_frequency', got "
                f"{class_weight_mode!r}"
            )

        expected_epochs = stage1_epochs + stage2_epochs
        if max_epochs is not None and int(max_epochs) != expected_epochs:
            raise ValueError(
                f"max_epochs={max_epochs} disagrees with the two stages "
                f"({stage1_epochs}+{stage2_epochs}={expected_epochs})"
            )
        self.save_hyperparameters()
        self.network = HMIOnlyDeepSWM(
            window_length=window_length,
            image_size=image_size,
            dim=dim,
            sequence_length=sequence_length,
            sparse_embed_dim=sparse_embed_dim,
            sparse_depth=sparse_depth,
            sparse_patch_size=sparse_patch_size,
            lt_depth=lt_depth,
            mixing_depth=mixing_depth,
            sse_dropout=sse_dropout,
            dcsm_dropout=dcsm_dropout,
            stssm_dropout=stssm_dropout,
            ltssm_dropout=ltssm_dropout,
            mixing_dropout=mixing_dropout,
            head_dropout=head_dropout,
        )

        self.register_buffer("training_class_counts", torch.zeros(NUM_CLASSES))
        self.register_buffer("class_weights", torch.ones(NUM_CLASSES))
        self.register_buffer(
            "training_class_probabilities", torch.full((NUM_CLASSES,), 0.25)
        )
        self.register_buffer(
            "gmgs_score_matrix", torch.zeros(NUM_CLASSES, NUM_CLASSES)
        )
        self.register_buffer("statistics_ready", torch.tensor(False))

        self.train_metrics = SplitAccumulator()
        self.val_metrics = SplitAccumulator()
        self.test_metrics = SplitAccumulator()
        self.last_confusion_matrices: dict[str, torch.Tensor] = {}
        self._stage2_active = False

        if train_class_counts is not None:
            self.set_training_class_counts(train_class_counts)

    @property
    def expected_max_epochs(self) -> int:
        return int(self.hparams.stage1_epochs + self.hparams.stage2_epochs)

    @torch.no_grad()
    def set_training_class_counts(
        self, counts: Sequence[int] | Mapping[int, int]
    ) -> None:
        if isinstance(counts, Mapping):
            ordered = [counts.get(index, 0) for index in range(NUM_CLASSES)]
        else:
            ordered = list(counts)
        count_tensor = torch.as_tensor(
            ordered, dtype=torch.float32, device=self.training_class_counts.device
        )
        if self.hparams.class_weight_mode == "none":
            weights = torch.ones_like(count_tensor)
        else:
            weights = inverse_frequency_weights(count_tensor).to(count_tensor.device)
        probabilities = count_tensor / count_tensor.sum()
        score_matrix = gerrity_score_matrix(probabilities).to(count_tensor.device)

        self.training_class_counts.copy_(count_tensor)
        self.class_weights.copy_(weights)
        self.training_class_probabilities.copy_(probabilities)
        self.gmgs_score_matrix.copy_(score_matrix)
        self.statistics_ready.fill_(True)

    @staticmethod
    def _find_class_counts(dataset: Any) -> Mapping[int, int] | Sequence[int] | None:
        seen: set[int] = set()
        current = dataset
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            counts = getattr(current, "class_counts", None)
            if counts is not None:
                return counts
            next_dataset = getattr(current, "dataset", None)
            if next_dataset is None:
                next_dataset = getattr(current, "data", None)
            current = next_dataset
        return None

    def _initialize_statistics_from_trainer(self) -> None:
        if bool(self.statistics_ready):
            return
        candidates = []
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            datasets = getattr(datamodule, "datasets", None)
            if isinstance(datasets, Mapping) and "train" in datasets:
                candidates.append(datasets["train"])
        train_dataloader = getattr(self.trainer, "train_dataloader", None)
        if train_dataloader is not None:
            candidates.append(getattr(train_dataloader, "dataset", None))

        for dataset in candidates:
            counts = self._find_class_counts(dataset)
            if counts is not None:
                self.set_training_class_counts(counts)
                return
        raise RuntimeError(
            "DeepSWM could not obtain train-split class_counts. Use "
            "DeepSWMSequenceDataset/FlareDataset through DataModuleFromConfig, "
            "or pass train_class_counts explicitly for a custom loader."
        )

    def _set_training_stage(self, stage2: bool) -> None:
        self._stage2_active = bool(stage2)
        if stage2:
            self.network.freeze_feature_extractor()
        else:
            self.network.unfreeze_feature_extractor()

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Discard legacy running-metric buffers from pre-DDP-fix checkpoints."""

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            return
        metric_prefixes = ("train_metrics.", "val_metrics.", "test_metrics.")
        for key in list(state_dict):
            if key.startswith(metric_prefixes):
                del state_dict[key]

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._stage2_active:
            for module in self.network.feature_extractors:
                module.eval()
            self.network.classification_head.train()
        return self

    def on_fit_start(self) -> None:
        self._initialize_statistics_from_trainer()
        configured_max_epochs = int(getattr(self.trainer, "max_epochs", -1))
        if configured_max_epochs != self.expected_max_epochs:
            warnings.warn(
                "DeepSWM's configured two-stage schedule expects "
                f"max_epochs={self.expected_max_epochs}, but Trainer has "
                f"max_epochs={configured_max_epochs}. This is acceptable for a "
                "smoke test but not the formal protocol.",
                stacklevel=2,
            )
        self._set_training_stage(self.current_epoch >= self.hparams.stage1_epochs)

    def on_train_start(self) -> None:
        self._initialize_statistics_from_trainer()

    def on_validation_start(self) -> None:
        # Also support standalone Trainer.validate() outside fit. During a
        # normal fit this is an idempotent no-op because on_fit_start already
        # stored the train-only climatology in persistent buffers.
        self._initialize_statistics_from_trainer()

    def on_test_start(self) -> None:
        # A restored checkpoint already carries these buffers. For a fresh
        # model, require an attached DataModule with its train split instead
        # of deriving loss weights or BSS climatology from test data.
        self._initialize_statistics_from_trainer()

    def on_train_epoch_start(self) -> None:
        self._set_training_stage(self.current_epoch >= self.hparams.stage1_epochs)
        self.log(
            "train_stage",
            2.0 if self._stage2_active else 1.0,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    def _prepare_hmi(self, hmi: torch.Tensor) -> torch.Tensor:
        if hmi.ndim == 4:
            if self.hparams.window_length != 1:
                raise ValueError(
                    "single-image batches are accepted only when window_length=1; "
                    "use DeepSWMSequenceDataset for T>1"
                )
            hmi = hmi.unsqueeze(1)
        if hmi.ndim != 5:
            raise ValueError(f"expected HMI [B,T,1,H,W], got {tuple(hmi.shape)}")
        if hmi.shape[1] != self.hparams.window_length or hmi.shape[2] != 1:
            raise ValueError(
                f"expected [B,{self.hparams.window_length},1,H,W], got "
                f"{tuple(hmi.shape)}"
            )
        hmi = hmi.float()
        if not torch.isfinite(hmi).all():
            raise ValueError("HMI batch contains NaN or infinite values")
        batch_size, time_length = hmi.shape[:2]
        resized = F.interpolate(
            hmi.flatten(0, 1),
            size=(self.hparams.image_size, self.hparams.image_size),
            mode="area",
        )
        return resized.reshape(batch_size, time_length, 1, *resized.shape[-2:])

    def forward(self, hmi: torch.Tensor) -> torch.Tensor:
        return self.network(self._prepare_hmi(hmi))

    def _weighted_mean(self, values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sample_weights = self.class_weights[labels].to(values.dtype)
        return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

    def loss_components(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not bool(self.statistics_ready):
            raise RuntimeError("training class statistics have not been initialized")
        probabilities = logits.float().softmax(dim=-1)
        one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).to(probabilities.dtype)

        cross_entropy = F.cross_entropy(logits.float(), labels, reduction="none")
        predicted_classes = probabilities.argmax(dim=-1)
        sample_scores = self.gmgs_score_matrix[labels, predicted_classes]
        # Paper definition: -s_(i*,j*) log p_(i*). The score selection is
        # intentionally hard/non-differentiable, matching the published loss.
        gmgs_oriented = -sample_scores * probabilities.gather(
            1, labels[:, None]
        ).squeeze(1).clamp_min(1e-8).log()
        # The paper's differentiable BSS-oriented term is the multiclass Brier
        # score; the actual M+ Brier skill score is reported as a metric.
        brier = (probabilities - one_hot).square().sum(dim=-1)

        return {
            "ce": self._weighted_mean(cross_entropy, labels),
            "gmgs": self._weighted_mean(gmgs_oriented, labels),
            "brier": self._weighted_mean(brier, labels),
        }

    def _shared_step(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if "hmi" not in batch or "label" not in batch:
            raise KeyError("batch must contain 'hmi' and 'label'")
        labels = batch["label"].long().reshape(-1)
        if labels.numel() == 0 or (labels < 0).any() or (labels >= NUM_CLASSES).any():
            raise ValueError("labels must use 0AB/C/M/X indices 0/1/2/3")
        logits = self(batch["hmi"])
        components = self.loss_components(logits, labels)
        loss = (
            self.hparams.ce_weight * components["ce"]
            + self.hparams.gmgs_weight * components["gmgs"]
            + self.hparams.bss_weight * components["brier"]
        )
        return loss, logits, labels, components

    def _log_losses(
        self,
        split: str,
        loss: torch.Tensor,
        components: Mapping[str, torch.Tensor],
        batch_size: int,
        on_step: bool,
    ) -> None:
        values = {
            f"{split}_loss": loss,
            f"{split}_ce_loss": components["ce"],
            f"{split}_gmgs_loss": components["gmgs"],
            f"{split}_brier_loss": components["brier"],
        }
        self.log_dict(
            values,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels, components = self._shared_step(batch)
        self.train_metrics.update(logits.detach(), labels)
        self._log_losses("train", loss, components, labels.numel(), on_step=True)
        return loss

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels, components = self._shared_step(batch)
        self.val_metrics.update(logits, labels, batch.get("date_id"))
        self._log_losses("val", loss, components, labels.numel(), on_step=False)
        return loss

    def test_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, logits, labels, components = self._shared_step(batch)
        self.test_metrics.update(logits, labels, batch.get("date_id"))
        self._log_losses("test", loss, components, labels.numel(), on_step=False)
        return loss

    def predict_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx
        logits = self(batch["hmi"])
        output = {
            "logits": logits,
            "probabilities": logits.softmax(dim=-1),
            "prediction": logits.argmax(dim=-1),
        }
        if "date_id" in batch:
            output["date_id"] = batch["date_id"]
        return output

    @torch.no_grad()
    def _epoch_metric_values(
        self,
        confusion: torch.Tensor,
        mplus_brier_sum: torch.Tensor,
        sample_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        confusion = confusion.to(torch.float64)
        total = confusion.sum()
        if total <= 0:
            return {}
        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        recall = true_positive / support.clamp_min(1.0)
        precision = true_positive / predicted.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

        tn = confusion[:2, :2].sum()
        fp = confusion[:2, 2:].sum()
        fn = confusion[2:, :2].sum()
        tp = confusion[2:, 2:].sum()
        tss_mplus = tp / (tp + fn).clamp_min(1.0) - fp / (fp + tn).clamp_min(1.0)

        event_climatology = self.training_class_probabilities[2:].sum().to(torch.float64)
        climatology_brier = event_climatology * (1.0 - event_climatology)
        mean_brier = mplus_brier_sum / sample_count.clamp_min(1)
        bss_mplus = 1.0 - mean_brier / climatology_brier.clamp_min(1e-12)
        # GMGS evaluation constructs S from the observed marginal of the
        # contingency table (paper Eq. 12--16 and the official metric code).
        # This is intentionally distinct from the training loss, whose fixed S
        # is derived from train-only climatology.  A tiny pseudo-count matches
        # the official missing-class safeguard without changing populated
        # formal validation splits in practice.
        metric_probabilities = support + 1.0e-8
        metric_probabilities = metric_probabilities / metric_probabilities.sum()
        metric_score_matrix = gerrity_score_matrix(metric_probabilities).to(
            device=confusion.device, dtype=torch.float64
        )
        gmgs = (metric_score_matrix * confusion).sum() / total
        return {
            "accuracy": true_positive.sum() / total,
            "macro_f1": f1.mean(),
            "tss_mplus": tss_mplus,
            "bss_mplus": bss_mplus,
            "gmgs": gmgs,
        }

    def _log_epoch_metrics(self, split: str, accumulator: SplitAccumulator) -> None:
        confusion, mplus_brier_sum, sample_count = accumulator.synchronized(
            device=self.device
        )
        values = self._epoch_metric_values(
            confusion, mplus_brier_sum, sample_count
        )
        self.last_confusion_matrices[split] = confusion.cpu()
        if values:
            self.log_dict(
                {f"{split}_{name}": value for name, value in values.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                # Values were already computed from globally gathered
                # sufficient statistics. Synchronizing the identical scalars
                # keeps Lightning's callback_metrics contract explicit for
                # distributed ModelCheckpoint callbacks.
                sync_dist=True,
            )
        accumulator.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val", self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test", self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(float(value) for value in self.hparams.adam_betas),
        )


__all__ = [
    "DeepSWM",
    "SplitAccumulator",
    "gerrity_score_matrix",
    "inverse_frequency_weights",
]
