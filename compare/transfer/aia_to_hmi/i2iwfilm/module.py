"""Lightning implementation of the Sayez et al. non-adversarial I2IwFiLM baseline."""

from __future__ import annotations

import math
from typing import Any, Sequence

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .networks import GuidedUNet, PairGuidanceEncoder, SourceGuidancePredictor


_AIA_MODALITIES = {
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
}


def _inverse_hmi_preprocess(
    normalized: torch.Tensor,
    mean: float,
    std: float,
    max_log_value: float,
) -> torch.Tensor:
    """Invert SolarDataset's signed-log1p then z-score HMI transform."""

    signed_log = normalized.float() * std + mean
    magnitude = signed_log.abs().clamp(max=max_log_value)
    return signed_log.sign() * torch.expm1(magnitude)


def _imagewise_pcc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = torch.sqrt(
        prediction.square().sum(dim=1) * target.square().sum(dim=1)
    )
    correlations = torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )
    return correlations.mean()


def _imagewise_ccc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    prediction_mean = prediction.mean(dim=1)
    target_mean = target.mean(dim=1)
    prediction_centered = prediction - prediction_mean[:, None]
    target_centered = target - target_mean[:, None]
    covariance = (prediction_centered * target_centered).mean(dim=1)
    denominator = (
        prediction_centered.square().mean(dim=1)
        + target_centered.square().mean(dim=1)
        + (prediction_mean - target_mean).square()
    )
    concordance = torch.where(
        denominator > 1e-12,
        2.0 * covariance / denominator.clamp_min(1e-12),
        torch.zeros_like(covariance),
    )
    return concordance.mean()


def _strong_field_polarity_accuracy(
    prediction_gauss: torch.Tensor,
    target_gauss: torch.Tensor,
    threshold_gauss: float,
) -> torch.Tensor:
    strong = target_gauss.abs() >= threshold_gauss
    correct = (prediction_gauss.sign() == target_gauss.sign()).float()
    strong_flat = strong.flatten(start_dim=1).float()
    count = strong_flat.sum(dim=1)
    per_image = (correct.flatten(start_dim=1) * strong_flat).sum(
        dim=1
    ) / count.clamp_min(1.0)
    valid_images = (count > 0).float()
    return (per_image * valid_images).sum() / valid_images.sum().clamp_min(1.0)


def _ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Dependency-free local SSIM, averaged over pixels and images."""

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be an odd integer >= 3")
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    if min(prediction.shape[-2:]) <= window_size // 2:
        raise ValueError("SSIM inputs are too small for reflection padding")

    prediction = prediction.float()
    target = target.float()
    padding = window_size // 2

    def local_mean(value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(value, kernel_size=window_size, stride=1)

    prediction_mean = local_mean(prediction)
    target_mean = local_mean(target)
    prediction_variance = (
        local_mean(prediction.square()) - prediction_mean.square()
    ).clamp_min(0.0)
    target_variance = (local_mean(target.square()) - target_mean.square()).clamp_min(
        0.0
    )
    covariance = local_mean(prediction * target) - prediction_mean * target_mean
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * prediction_mean * target_mean + c1) * (2.0 * covariance + c2)
    denominator = (prediction_mean.square() + target_mean.square() + c1) * (
        prediction_variance + target_variance + c2
    )
    score = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
    return score.flatten(start_dim=1).mean(dim=1)


class SayezI2IwFiLM(pl.LightningModule):
    """Two-stage, non-adversarial AIA-to-HMI image translation.

    Stage 1 learns a paired source/target guidance encoder and guided U-Net.
    Stage 2 uses the stopped-gradient paired vector as supervision for a
    source-only guidance predictor, while continuing U-Net reconstruction.
    Validation, testing, ``forward``, and image logging always use only source.
    """

    def __init__(
        self,
        source_modal: str = "0304",
        target_modal: str = "hmi",
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 8),
        guidance_dim: int = 256,
        guidance_base_channels: int = 64,
        guidance_residual_blocks: int = 6,
        guidance_unshuffle_factor: int = 4,
        source_mlp_hidden_dims: Sequence[int] = (512, 512, 512),
        stage1_epochs: int = 100,
        max_epochs: int = 200,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_reconstruction: float = 1.0,
        lambda_guidance: float = 1.0,
        minimum_learning_rate: float = 1e-7,
        output_activation: str = "identity",
        hmi_mean: float = -0.0033644122878536808,
        hmi_std: float = 1.4462468177923982,
        metric_max_log_value: float = 20.0,
        hmi_ssim_clip_gauss: float = 1500.0,
        strong_field_threshold_gauss: float = 100.0,
        ssim_window_size: int = 11,
    ) -> None:
        super().__init__()
        if target_modal != "hmi":
            raise ValueError("SayezI2IwFiLM is an AIA-to-HMI comparison")
        if source_modal not in _AIA_MODALITIES:
            raise ValueError(
                f"source_modal must be one of {sorted(_AIA_MODALITIES)}; "
                f"received {source_modal!r}"
            )
        if output_activation != "identity":
            raise ValueError("SolarCHIP z-score targets require identity output")
        if not 0 < stage1_epochs < max_epochs:
            raise ValueError("stage1_epochs must lie strictly between 0 and max_epochs")
        if hmi_std <= 0 or hmi_ssim_clip_gauss <= 0:
            raise ValueError("HMI scale parameters must be positive")
        if lambda_reconstruction < 0 or lambda_guidance < 0:
            raise ValueError("loss weights cannot be negative")
        if learning_rate <= 0 or minimum_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and base LR positive")

        self.save_hyperparameters()
        self.source_modal = source_modal
        self.target_modal = target_modal
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.stage1_epochs = int(stage1_epochs)
        self.max_epochs_config = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lambda_reconstruction = float(lambda_reconstruction)
        self.lambda_guidance = float(lambda_guidance)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.hmi_mean = float(hmi_mean)
        self.hmi_std = float(hmi_std)
        self.metric_max_log_value = float(metric_max_log_value)
        self.hmi_ssim_clip_gauss = float(hmi_ssim_clip_gauss)
        self.strong_field_threshold_gauss = float(strong_field_threshold_gauss)
        self.ssim_window_size = int(ssim_window_size)

        self.pair_guidance_encoder = PairGuidanceEncoder(
            source_channels=input_channels,
            target_channels=output_channels,
            guidance_dim=guidance_dim,
            base_channels=guidance_base_channels,
            residual_blocks=guidance_residual_blocks,
            unshuffle_factor=guidance_unshuffle_factor,
        )
        self.source_guidance_predictor = SourceGuidancePredictor(
            source_channels=input_channels,
            guidance_dim=guidance_dim,
            base_channels=guidance_base_channels,
            residual_blocks=guidance_residual_blocks,
            unshuffle_factor=guidance_unshuffle_factor,
            mlp_hidden_dims=source_mlp_hidden_dims,
        )
        self.generator = GuidedUNet(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            guidance_dim=guidance_dim,
        )

    @property
    def in_stage_one(self) -> bool:
        return self.current_epoch < self.stage1_epochs

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Deployable inference: the target is never observed on this path."""

        source = source.float()
        guidance = self.source_guidance_predictor(source)
        return self.generator(source, guidance)

    def _paired_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            source = batch[self.source_modal].float()
            target = batch[self.target_modal].float()
        except KeyError as error:
            raise KeyError(
                f"Expected batch keys {self.source_modal!r} and {self.target_modal!r}; "
                f"received {sorted(batch)}"
            ) from error
        if source.ndim != 4 or target.ndim != 4:
            raise ValueError("source and target batches must have shape [B, C, H, W]")
        if source.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} source channels, got {source.shape[1]}"
            )
        if target.shape[1] != self.output_channels:
            raise ValueError(
                f"expected {self.output_channels} target channels, got {target.shape[1]}"
            )
        if source.shape[0] != target.shape[0] or source.shape[-2:] != target.shape[-2:]:
            raise ValueError("source and target must be aligned image pairs")
        return source, target

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        source, target = self._paired_batch(batch)

        if self.in_stage_one:
            pair_guidance = self.pair_guidance_encoder(source, target)
            prediction = self.generator(source, pair_guidance)
            guidance_loss = prediction.new_zeros(())
            stage = 1.0
        else:
            # The paired encoder is a fixed teacher during Stage 2.  Parameters
            # remain in the same optimizer; no dynamic requires_grad changes are
            # made, which keeps DDP reducer construction stable.
            with torch.no_grad():
                pair_guidance = self.pair_guidance_encoder(source, target).detach()
            source_guidance = self.source_guidance_predictor(source)
            prediction = self.generator(source, source_guidance)
            guidance_loss = F.l1_loss(source_guidance, pair_guidance)
            stage = 2.0

        reconstruction_loss = F.l1_loss(prediction, target)
        loss = (
            self.lambda_reconstruction * reconstruction_loss
            + self.lambda_guidance * guidance_loss
        )
        common = dict(
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        self.log("train/loss", loss, prog_bar=True, **common)
        self.log("train/reconstruction_l1", reconstruction_loss, **common)
        self.log("train/guidance_l1", guidance_loss, **common)
        self.log(
            "train/stage",
            loss.new_tensor(stage),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        return loss

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        split: str,
    ) -> torch.Tensor:
        source, target = self._paired_batch(batch)
        paired_teacher_loss = None
        if self.in_stage_one:
            # Stage-one validation diagnostic only. It deliberately is not the
            # checkpoint monitor because this path observes the target image.
            pair_guidance = self.pair_guidance_encoder(source, target)
            paired_prediction = self.generator(source, pair_guidance)
            paired_teacher_loss = F.l1_loss(paired_prediction, target)

        # Critical evaluation invariant: the checkpoint metric and deployable
        # prediction never use paired guidance.
        prediction = self(source)
        loss = F.l1_loss(prediction, target)

        prediction_gauss = _inverse_hmi_preprocess(
            prediction,
            self.hmi_mean,
            self.hmi_std,
            self.metric_max_log_value,
        )
        target_gauss = _inverse_hmi_preprocess(
            target,
            self.hmi_mean,
            self.hmi_std,
            self.metric_max_log_value,
        )
        error_gauss = prediction_gauss - target_gauss
        rmse_gauss = error_gauss.square().mean().sqrt()
        mae_gauss = error_gauss.abs().mean()
        pcc = _imagewise_pcc(prediction_gauss, target_gauss)
        ccc = _imagewise_ccc(prediction_gauss, target_gauss)
        polarity = _strong_field_polarity_accuracy(
            prediction_gauss,
            target_gauss,
            self.strong_field_threshold_gauss,
        )

        clip = self.hmi_ssim_clip_gauss
        prediction_ssim = prediction_gauss.clamp(-clip, clip) / clip
        target_ssim = target_gauss.clamp(-clip, clip) / clip
        generated_ssim = _ssim_index(
            prediction_ssim,
            target_ssim,
            window_size=self.ssim_window_size,
            data_range=2.0,
        )
        zero_ssim = _ssim_index(
            torch.zeros_like(target_ssim),
            target_ssim,
            window_size=self.ssim_window_size,
            data_range=2.0,
        )
        delta_ssim = (generated_ssim - zero_ssim).mean()

        common = dict(
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        self.log(f"{split}/loss", loss, prog_bar=True, **common)
        if paired_teacher_loss is not None:
            self.log(
                f"{split}/paired_teacher_l1",
                paired_teacher_loss,
                **common,
            )
        self.log(f"{split}/rmse_gauss", rmse_gauss, **common)
        self.log(f"{split}/pcc", pcc, **common)
        self.log(f"{split}/ccc", ccc, **common)
        self.log(f"{split}/physical_mae_gauss", mae_gauss, **common)
        self.log(f"{split}/strong_field_polarity", polarity, **common)
        self.log(f"{split}/delta_ssim", delta_ssim, prog_bar=True, **common)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        # All parameters are present from the first step through the stage
        # transition; Stage-specific graph usage is handled by Lightning's
        # ddp_find_unused_parameters_true strategy in the supplied configs.
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        minimum_factor = self.minimum_learning_rate / self.learning_rate

        def two_stage_cosine(epoch: int) -> float:
            # The source-only predictor first receives gradients at the Stage
            # 2 boundary. Restarting the cosine there gives that deployable
            # path a full learning-rate cycle without rebuilding the optimizer
            # or changing DDP parameter registration.
            if epoch < self.stage1_epochs:
                stage_epoch = epoch
                stage_length = self.stage1_epochs
            else:
                stage_epoch = epoch - self.stage1_epochs
                stage_length = self.max_epochs_config - self.stage1_epochs
            denominator = max(stage_length - 1, 1)
            progress = min(max(stage_epoch / denominator, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=two_stage_cosine,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @torch.no_grad()
    def log_images(
        self, batch: dict[str, torch.Tensor], **_: Any
    ) -> dict[str, torch.Tensor]:
        source, target = self._paired_batch(batch)
        generated = self(source)
        return {
            f"visualization/{self.source_modal}/condition": source,
            "visualization/hmi/target": target,
            "visualization/hmi/generated": generated,
        }
