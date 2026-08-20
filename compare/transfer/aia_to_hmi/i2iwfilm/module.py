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


def _mean_abs_spatial_gradient(value: torch.Tensor) -> torch.Tensor:
    """Mean first-order spatial variation for blockiness diagnostics."""

    value = value.float()
    horizontal = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
    vertical = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
    return 0.5 * (horizontal + vertical)


class SayezI2IwFiLM(pl.LightningModule):
    """Two-stage, non-adversarial AIA-to-HMI image translation.

    Stage 1 learns a paired source/target guidance encoder and guided U-Net,
    while the source-only predictor tracks the stopped-gradient paired vector.
    Stage 2 freezes the paired path and generator in the training graph and
    continues source-only guidance distillation.
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
        reconstruction_loss_type: str = "l1",
        strong_field_weight: float = 1.0,
        strong_field_loss_fraction: float = 0.5,
        smooth_l1_beta: float = 1.0,
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
        reconstruction_loss_type = reconstruction_loss_type.lower()
        if reconstruction_loss_type not in {
            "l1",
            "l2",
            "smooth_l1",
            "weighted_smooth_l1",
            "balanced_smooth_l1",
        }:
            raise ValueError(
                "reconstruction_loss_type must be one of "
                "'l1', 'l2', 'smooth_l1', 'weighted_smooth_l1', or "
                "'balanced_smooth_l1'"
            )
        if strong_field_weight < 1.0:
            raise ValueError("strong_field_weight must be at least 1")
        if not 0.0 < strong_field_loss_fraction < 1.0:
            raise ValueError("strong_field_loss_fraction must lie between 0 and 1")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if learning_rate <= 0 or minimum_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and base LR positive")
        if strong_field_threshold_gauss <= 0:
            raise ValueError("strong_field_threshold_gauss must be positive")

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
        self.reconstruction_loss_type = reconstruction_loss_type
        self.strong_field_weight = float(strong_field_weight)
        self.strong_field_loss_fraction = float(strong_field_loss_fraction)
        self.smooth_l1_beta = float(smooth_l1_beta)
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

    def _strong_field_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Return the HMI |B| threshold mask without an expensive expm1."""

        threshold_log = math.log1p(self.strong_field_threshold_gauss)
        positive_threshold = (threshold_log - self.hmi_mean) / self.hmi_std
        negative_threshold = (-threshold_log - self.hmi_mean) / self.hmi_std
        target = target.float()
        return (target >= positive_threshold) | (target <= negative_threshold)

    def _reconstruction_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the configured objective plus collapse-diagnostic terms."""

        prediction = prediction.float()
        target = target.float()
        absolute_error = (prediction - target).abs()
        raw_l1 = absolute_error.mean()
        strong_mask = self._strong_field_mask(target)
        strong_count = strong_mask.sum()
        strong_l1 = (
            absolute_error * strong_mask.to(dtype=absolute_error.dtype)
        ).sum() / strong_count.clamp_min(1).to(dtype=absolute_error.dtype)
        strong_fraction = strong_mask.float().mean()

        if self.reconstruction_loss_type == "l1":
            objective = raw_l1
        elif self.reconstruction_loss_type == "l2":
            objective = F.mse_loss(prediction, target)
        else:
            per_pixel = F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=self.smooth_l1_beta,
            )
            if self.reconstruction_loss_type == "weighted_smooth_l1":
                weights = 1.0 + (self.strong_field_weight - 1.0) * strong_mask.to(
                    dtype=per_pixel.dtype
                )
                objective = (weights * per_pixel).sum() / weights.sum().clamp_min(1.0)
            elif self.reconstruction_loss_type == "balanced_smooth_l1":
                # A fixed per-pixel multiplier still lets the quiet Sun dominate
                # when strong-field pixels are below one percent of the image.
                # Average the two regions separately so their contribution is
                # independent of the batch's strong-field pixel frequency.
                quiet_mask = ~strong_mask
                quiet_count = quiet_mask.sum()
                strong_objective = (
                    per_pixel * strong_mask.to(dtype=per_pixel.dtype)
                ).sum() / strong_count.clamp_min(1).to(dtype=per_pixel.dtype)
                quiet_objective = (
                    per_pixel * quiet_mask.to(dtype=per_pixel.dtype)
                ).sum() / quiet_count.clamp_min(1).to(dtype=per_pixel.dtype)
                balanced = (
                    1.0 - self.strong_field_loss_fraction
                ) * quiet_objective + self.strong_field_loss_fraction * strong_objective
                objective = torch.where(
                    strong_count > 0,
                    balanced,
                    quiet_objective,
                )
            else:
                objective = per_pixel.mean()

        return objective, raw_l1, strong_l1, strong_fraction

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        source, target = self._paired_batch(batch)

        if self.in_stage_one:
            pair_guidance = self.pair_guidance_encoder(source, target)
            prediction = self.generator(source, pair_guidance)
            # The teacher remains governed only by reconstruction: detaching its
            # vector lets the deployable predictor track it from the beginning
            # without pulling the paired representation toward an easier target.
            source_guidance = self.source_guidance_predictor(source)
            guidance_loss = F.l1_loss(source_guidance, pair_guidance.detach())
            stage = 1.0
        else:
            # The paired encoder is a fixed teacher during Stage 2.  Parameters
            # remain in the same optimizer; no dynamic requires_grad changes are
            # made, which keeps DDP reducer construction stable.
            with torch.no_grad():
                pair_guidance = self.pair_guidance_encoder(source, target).detach()
            source_guidance = self.source_guidance_predictor(source)
            guidance_loss = F.l1_loss(source_guidance, pair_guidance)
            # Match the authors' released Stage-2 configuration: distil the
            # source-only guidance predictor while keeping the Stage-1 teacher
            # and image generator fixed.  The detached prediction below exists
            # only for diagnostics and cannot update the generator.
            with torch.no_grad():
                prediction = self.generator(source, source_guidance.detach())
            stage = 2.0

        (
            reconstruction_loss,
            reconstruction_l1,
            strong_field_l1,
            strong_field_fraction,
        ) = self._reconstruction_components(prediction, target)
        if self.in_stage_one:
            loss = (
                self.lambda_reconstruction * reconstruction_loss
                + self.lambda_guidance * guidance_loss
            )
        else:
            loss = self.lambda_guidance * guidance_loss
        common = dict(
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        self.log("train/loss", loss, prog_bar=True, **common)
        self.log("train/reconstruction_objective", reconstruction_loss, **common)
        self.log("train/reconstruction_l1", reconstruction_l1, **common)
        self.log("train/strong_field_l1", strong_field_l1, **common)
        self.log("train/strong_field_fraction", strong_field_fraction, **common)
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
        # Diagnostic teacher path. It observes the target, so these values are
        # never used as the deployable checkpoint monitor. Keeping them after
        # the stage transition makes the guidance-distillation gap visible.
        pair_guidance = self.pair_guidance_encoder(source, target)
        paired_prediction = self.generator(source, pair_guidance)
        (
            paired_teacher_objective,
            paired_teacher_l1,
            _,
            _,
        ) = self._reconstruction_components(paired_prediction, target)

        # Critical evaluation invariant: the checkpoint metric and deployable
        # prediction never use paired guidance.
        prediction = self(source)
        (
            loss,
            reconstruction_l1,
            strong_field_l1,
            strong_field_fraction,
        ) = self._reconstruction_components(prediction, target)

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
        paired_prediction_gauss = _inverse_hmi_preprocess(
            paired_prediction,
            self.hmi_mean,
            self.hmi_std,
            self.metric_max_log_value,
        )
        paired_teacher_pcc = _imagewise_pcc(paired_prediction_gauss, target_gauss)
        paired_teacher_ccc = _imagewise_ccc(paired_prediction_gauss, target_gauss)
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
        self.log(f"{split}/reconstruction_l1", reconstruction_l1, **common)
        self.log(f"{split}/strong_field_l1", strong_field_l1, **common)
        self.log(f"{split}/strong_field_fraction", strong_field_fraction, **common)
        prediction_std = prediction.float().std()
        target_std = target.float().std()
        paired_teacher_std = paired_prediction.float().std()
        target_gradient = _mean_abs_spatial_gradient(target)
        prediction_gradient = _mean_abs_spatial_gradient(prediction)
        paired_teacher_gradient = _mean_abs_spatial_gradient(paired_prediction)
        prediction_strong_fraction = self._strong_field_mask(prediction).float().mean()
        paired_teacher_strong_fraction = (
            self._strong_field_mask(paired_prediction).float().mean()
        )
        self.log(f"{split}/prediction_std", prediction_std, **common)
        self.log(f"{split}/target_std", target_std, **common)
        self.log(
            f"{split}/amplitude_ratio",
            prediction_std / target_std.clamp_min(1e-8),
            **common,
        )
        self.log(
            f"{split}/prediction_abs_mean",
            prediction.float().abs().mean(),
            **common,
        )
        self.log(
            f"{split}/spatial_gradient_ratio",
            prediction_gradient / target_gradient.clamp_min(1e-8),
            **common,
        )
        self.log(
            f"{split}/prediction_strong_field_fraction",
            prediction_strong_fraction,
            **common,
        )
        self.log(f"{split}/paired_teacher_l1", paired_teacher_l1, **common)
        self.log(
            f"{split}/paired_teacher_objective",
            paired_teacher_objective,
            **common,
        )
        self.log(f"{split}/paired_teacher_std", paired_teacher_std, **common)
        self.log(
            f"{split}/paired_teacher_spatial_gradient_ratio",
            paired_teacher_gradient / target_gradient.clamp_min(1e-8),
            **common,
        )
        self.log(
            f"{split}/paired_teacher_strong_field_fraction",
            paired_teacher_strong_fraction,
            **common,
        )
        self.log(
            f"{split}/paired_teacher_amplitude_ratio",
            paired_teacher_std / target_std.clamp_min(1e-8),
            **common,
        )
        self.log(f"{split}/paired_teacher_pcc", paired_teacher_pcc, **common)
        self.log(f"{split}/paired_teacher_ccc", paired_teacher_ccc, **common)
        self.log(f"{split}/rmse_gauss", rmse_gauss, **common)
        self.log(f"{split}/pcc", pcc, **common)
        self.log(f"{split}/ccc", ccc, **common)
        if split == "val":
            # This is always the deployable source-only path. Stage 1 now trains
            # its guidance predictor, so a genuinely better early checkpoint is
            # valid and should not be discarded at the stage boundary.
            self.log("val/checkpoint_ccc", ccc, **common)
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
            # Restart at the Stage-2 boundary so the deployable predictor gets a
            # second full learning-rate cycle while tracking the fixed teacher.
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
        images = {
            f"visualization/{self.source_modal}/condition": source,
            "visualization/hmi/target": target,
            "visualization/hmi/generated": generated,
        }
        pair_guidance = self.pair_guidance_encoder(source, target)
        images["visualization/hmi/generated_paired_teacher"] = self.generator(
            source,
            pair_guidance,
        )
        return images
