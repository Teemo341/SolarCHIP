"""Lightning training adapter for the Dannehl et al. Pix2PixCC baseline."""

from __future__ import annotations

from typing import Any

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from .networks import PatchDiscriminator, Pix2PixCCGenerator, initialize_pix2pixcc


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


def _pearson_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-image Pearson correlation, computed in float32."""

    prediction = prediction.float().flatten(start_dim=1)
    target = target.float().flatten(start_dim=1)

    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (prediction * target).sum(dim=1)
    denominator = (
        prediction.square().sum(dim=1).sqrt()
        * target.square().sum(dim=1).sqrt()
    )

    return (numerator / denominator.clamp_min(1e-8)).mean()


def _concordance_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-4,
    min_target_variance: float = 1e-6,
) -> torch.Tensor:
    """Calculate a numerically stable mean per-image Lin CCC.

    The original implementation only clamped the denominator to 1e-8.
    Low-variance targets could therefore produce extremely large backward
    gradients even when the forward CCC remained finite.

    A positive epsilon is now added directly to the denominator. Images whose
    target variance is too small to define a meaningful correlation objective
    are excluded from the batch mean.
    """

    prediction = prediction.float().flatten(start_dim=1)
    target = target.float().flatten(start_dim=1)

    prediction_mean = prediction.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean

    covariance = (centered_prediction * centered_target).mean(dim=1)
    prediction_variance = centered_prediction.square().mean(dim=1)
    target_variance = centered_target.square().mean(dim=1)

    mean_difference = (
        prediction_mean.squeeze(1) - target_mean.squeeze(1)
    ).square()

    denominator = (
        prediction_variance
        + target_variance
        + mean_difference
    )

    per_image_ccc = 2.0 * covariance / (denominator + epsilon)
    per_image_ccc = per_image_ccc.clamp(min=-1.0, max=1.0)

    valid = target_variance >= min_target_variance
    masked_ccc = torch.where(
        valid,
        per_image_ccc,
        torch.zeros_like(per_image_ccc),
    )

    # If every image is invalid, masked_ccc remains connected to prediction
    # through torch.where and produces a differentiable zero.
    valid_count = valid.sum().clamp_min(1)
    return masked_ccc.sum() / valid_count


def _inverse_hmi_preprocess(
    normalized: torch.Tensor,
    mean: float,
    std: float,
    max_log_value: float,
) -> torch.Tensor:
    """Undo SolarDataset's z-score and signed-log1p HMI transforms."""

    signed_log = normalized.float() * std + mean
    magnitude = signed_log.abs().clamp(max=max_log_value)
    return signed_log.sign() * torch.expm1(magnitude)


class DannehlPix2PixCC(pl.LightningModule):
    """SolarCHIP-compatible Pix2PixCC for one AIA channel to HMI LOS.

    The generator objective is LSGAN + discriminator feature matching +
    multiscale CCC. Validation monitors normalized-space L1 rather than the
    adversarial objective.
    """

    def __init__(
        self,
        source_modal: str = "0304",
        target_modal: str = "hmi",
        input_channels: int = 1,
        output_channels: int = 1,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        n_downsample: int = 4,
        n_residual: int = 9,
        input_kernel_size: int = 7,
        downsample_kernel_size: int = 5,
        norm_type: str = "instance",
        padding_type: str = "replication",
        output_activation: str = "identity",
        learning_rate: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_lsgan: float = 2.0,
        lambda_feature_matching: float = 10.0,
        lambda_cc: float = 5.0,
        n_cc_scales: int = 4,
        ccc_epsilon: float = 1e-4,
        ccc_min_target_variance: float = 1e-6,
        gradient_clip_val: float = 1.0,
        target_mean: float = -0.0033644122878536808,
        target_std: float = 1.4462468177923982,
        metric_max_log_value: float = 20.0,
        strong_field_threshold: float = 100.0,
    ) -> None:
        super().__init__()

        if source_modal not in _AIA_MODALITIES:
            raise ValueError(
                f"source_modal must be one of {sorted(_AIA_MODALITIES)}; "
                f"received {source_modal!r}"
            )

        if target_modal != "hmi":
            raise ValueError(
                "Dannehl AIA-to-HMI comparisons require target_modal='hmi'"
            )

        if input_channels != 1 or output_channels != 1:
            raise ValueError(
                "The supplied SolarCHIP comparison is "
                "single-input/single-output"
            )

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")

        if min(lambda_lsgan, lambda_feature_matching, lambda_cc) < 0:
            raise ValueError("loss weights must be non-negative")

        if n_cc_scales < 1:
            raise ValueError("n_cc_scales must be positive")

        if ccc_epsilon <= 0:
            raise ValueError("ccc_epsilon must be positive")

        if ccc_min_target_variance < 0:
            raise ValueError(
                "ccc_min_target_variance must be non-negative"
            )

        if gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be non-negative")

        if target_std <= 0:
            raise ValueError("target_std must be positive")

        if metric_max_log_value <= 0:
            raise ValueError("metric_max_log_value must be positive")

        if strong_field_threshold <= 0:
            raise ValueError("strong_field_threshold must be positive")

        self.save_hyperparameters()

        self.source_modal = source_modal
        self.target_modal = target_modal

        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

        self.lambda_lsgan = float(lambda_lsgan)
        self.lambda_feature_matching = float(lambda_feature_matching)
        self.lambda_cc = float(lambda_cc)
        self.n_cc_scales = int(n_cc_scales)

        self.ccc_epsilon = float(ccc_epsilon)
        self.ccc_min_target_variance = float(
            ccc_min_target_variance
        )
        self.gradient_clip_val = float(gradient_clip_val)

        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.metric_max_log_value = float(metric_max_log_value)
        self.strong_field_threshold = float(strong_field_threshold)

        self.monitor = "val/loss"

        self.generator = Pix2PixCCGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=generator_channels,
            n_downsample=n_downsample,
            n_residual=n_residual,
            input_kernel_size=input_kernel_size,
            downsample_kernel_size=downsample_kernel_size,
            norm_type=norm_type,
            padding_type=padding_type,
            output_activation=output_activation,
        )

        self.discriminator = PatchDiscriminator(
            input_channels=input_channels + output_channels,
            base_channels=discriminator_channels,
        )

        initialize_pix2pixcc(self.generator)
        initialize_pix2pixcc(self.discriminator)

        self.automatic_optimization = False

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.generator(source.float())

    @staticmethod
    def _adversarial_loss(
        logits: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        labels = (
            torch.ones_like(logits)
            if is_real
            else torch.zeros_like(logits)
        )
        return 0.5 * F.mse_loss(logits, labels)

    @staticmethod
    def _feature_matching_loss(
        fake_features: list[torch.Tensor],
        real_features: list[torch.Tensor],
    ) -> torch.Tensor:
        # The paper defines T=4 discriminator feature layers. The final
        # discriminator logit is excluded from feature matching.
        layer_losses = [
            F.l1_loss(fake, real.detach())
            for fake, real in zip(
                fake_features[:-1],
                real_features[:-1],
            )
        ]

        if not layer_losses:
            return fake_features[-1].new_zeros(())

        return torch.stack(layer_losses).sum()

    def _multiscale_ccc_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        scale_losses = []

        scaled_prediction = prediction
        scaled_target = target

        for scale_index in range(self.n_cc_scales):
            ccc = _concordance_correlation(
                scaled_prediction,
                scaled_target,
                epsilon=self.ccc_epsilon,
                min_target_variance=self.ccc_min_target_variance,
            )
            scale_losses.append(1.0 - ccc)

            if scale_index + 1 < self.n_cc_scales:
                scaled_prediction = F.avg_pool2d(
                    scaled_prediction,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=False,
                )
                scaled_target = F.avg_pool2d(
                    scaled_target,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=False,
                )

        return torch.stack(scale_losses).mean()

    def _paired_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            source = batch[self.source_modal].float()
            target = batch[self.target_modal].float()
        except KeyError as error:
            raise KeyError(
                f"Expected batch keys {self.source_modal!r} and "
                f"{self.target_modal!r}; received {sorted(batch)}"
            ) from error

        if not torch.isfinite(source).all():
            raise FloatingPointError(
                f"Batch modality {self.source_modal!r} contains NaN or Inf"
            )

        if not torch.isfinite(target).all():
            raise FloatingPointError(
                f"Batch modality {self.target_modal!r} contains NaN or Inf"
            )

        return source, target

    @staticmethod
    def _require_finite_loss(
        name: str,
        value: torch.Tensor,
    ) -> None:
        """Fail before backward when a scalar loss is already non-finite."""

        if not torch.isfinite(value.detach()).all():
            raise FloatingPointError(
                f"{name} became non-finite before backward: "
                f"{value.detach().float().cpu().item()}"
            )

    def _clip_optimizer_gradients(
        self,
        optimizer: Any,
    ) -> None:
        """Clip manual-optimization gradients through Lightning."""

        if self.gradient_clip_val <= 0:
            return

        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch_idx

        source, target = self._paired_batch(batch)
        optimizer_g, optimizer_d = self.optimizers()

        # --------------------------------------------------------------
        # Discriminator update
        # --------------------------------------------------------------
        # Recompute G for the generator update rather than retaining the
        # 1024x1024 generator graph across the discriminator step.
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad(set_to_none=True)

        with torch.no_grad():
            generated_for_d = self(source)

        real_features_d = self.discriminator(
            torch.cat([source, target], dim=1)
        )
        fake_features_d = self.discriminator(
            torch.cat([source, generated_for_d], dim=1)
        )

        discriminator_real = self._adversarial_loss(
            real_features_d[-1],
            is_real=True,
        )
        discriminator_fake = self._adversarial_loss(
            fake_features_d[-1],
            is_real=False,
        )
        discriminator_loss = (
            discriminator_real + discriminator_fake
        )

        self._require_finite_loss(
            "discriminator_loss",
            discriminator_loss,
        )

        self.manual_backward(discriminator_loss)
        self._clip_optimizer_gradients(optimizer_d)
        optimizer_d.step()
        optimizer_d.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_d)

        discriminator_loss_log = discriminator_loss.detach()

        # Release both PatchGAN graphs before the second generator forward.
        # At 1024x1024 these feature lists otherwise consume several GiB.
        del (
            generated_for_d,
            real_features_d,
            fake_features_d,
            discriminator_real,
            discriminator_fake,
            discriminator_loss,
        )

        # --------------------------------------------------------------
        # Generator update
        # --------------------------------------------------------------
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad(set_to_none=True)

        generated = self(source)

        with torch.no_grad():
            real_features_g = self.discriminator(
                torch.cat([source, target], dim=1)
            )

        fake_features_g = self.discriminator(
            torch.cat([source, generated], dim=1)
        )

        generator_adversarial = self._adversarial_loss(
            fake_features_g[-1],
            is_real=True,
        )
        feature_matching = self._feature_matching_loss(
            fake_features_g,
            real_features_g,
        )

        # Completely skip CCC when lambda_cc is zero. Computing CCC and then
        # multiplying it by zero is unsafe because 0 * NaN remains NaN.
        if self.lambda_cc > 0:
            ccc_loss = self._multiscale_ccc_loss(
                generated,
                target,
            )
        else:
            ccc_loss = generated.sum() * 0.0

        generator_loss = (
            self.lambda_lsgan * generator_adversarial
            + self.lambda_feature_matching * feature_matching
            + self.lambda_cc * ccc_loss
        )

        self._require_finite_loss(
            "generator_adversarial",
            generator_adversarial,
        )
        self._require_finite_loss(
            "feature_matching",
            feature_matching,
        )
        self._require_finite_loss(
            "ccc_loss",
            ccc_loss,
        )
        self._require_finite_loss(
            "generator_loss",
            generator_loss,
        )

        self.manual_backward(generator_loss)
        self._clip_optimizer_gradients(optimizer_g)
        optimizer_g.step()
        optimizer_g.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optimizer_g)

        generator_loss_log = generator_loss.detach()
        generator_adversarial_log = generator_adversarial.detach()
        feature_matching_log = feature_matching.detach()
        ccc_loss_log = ccc_loss.detach()

        del (
            real_features_g,
            fake_features_g,
            generator_loss,
            generator_adversarial,
            feature_matching,
            ccc_loss,
        )

        normalized_l1 = F.l1_loss(
            generated.detach(),
            target,
        )

        batch_size = source.shape[0]
        common = dict(
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            "train/g_loss",
            generator_loss_log,
            prog_bar=True,
            **common,
        )
        self.log(
            "train/d_loss",
            discriminator_loss_log,
            prog_bar=True,
            **common,
        )
        self.log(
            "train/g_lsgan",
            generator_adversarial_log,
            **common,
        )
        self.log(
            "train/feature_matching",
            feature_matching_log,
            **common,
        )
        self.log(
            "train/ccc_loss",
            ccc_loss_log,
            **common,
        )
        self.log(
            "train/l1",
            normalized_l1,
            **common,
        )

    @staticmethod
    def _masked_image_average(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values.float().flatten(start_dim=1)
        mask = mask.flatten(start_dim=1)

        counts = mask.float().sum(dim=1)
        valid_images = counts > 0

        per_image = (
            values * mask.float()
        ).sum(dim=1) / counts.clamp_min(1.0)

        return (
            per_image * valid_images.float()
        ).sum() / valid_images.float().sum().clamp_min(1.0)

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        split: str,
    ) -> torch.Tensor:
        source, target = self._paired_batch(batch)
        generated = self(source)

        if not torch.isfinite(generated).all():
            raise FloatingPointError(
                f"{split} generator output contains NaN or Inf"
            )

        # Use a deterministic non-adversarial validation objective so that
        # checkpoints remain comparable across discriminator oscillations.
        normalized_l1 = F.l1_loss(generated, target)
        normalized_mse = F.mse_loss(generated, target)

        pcc = _pearson_correlation(generated, target)
        ccc = _concordance_correlation(
            generated,
            target,
            epsilon=self.ccc_epsilon,
            min_target_variance=self.ccc_min_target_variance,
        )

        generated_raw = _inverse_hmi_preprocess(
            generated,
            mean=self.target_mean,
            std=self.target_std,
            max_log_value=self.metric_max_log_value,
        )
        target_raw = _inverse_hmi_preprocess(
            target,
            mean=self.target_mean,
            std=self.target_std,
            max_log_value=self.metric_max_log_value,
        )

        absolute_error = (generated_raw - target_raw).abs()
        physical_mae = absolute_error.mean()
        mean_field_bias = (generated_raw - target_raw).mean()

        strong_mask = (
            target_raw.abs() >= self.strong_field_threshold
        )
        strong_field_mae = self._masked_image_average(
            absolute_error,
            strong_mask,
        )

        polarity_correct = (
            generated_raw.sign() == target_raw.sign()
        )
        strong_field_polarity_accuracy = (
            self._masked_image_average(
                polarity_correct.float(),
                strong_mask,
            )
        )

        strong_field_fraction = (
            strong_mask.float()
            .flatten(start_dim=1)
            .mean(dim=1)
            .mean()
        )

        batch_size = source.shape[0]
        common = dict(
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            f"{split}/loss",
            normalized_l1,
            prog_bar=True,
            **common,
        )
        self.log(
            f"{split}/l1",
            normalized_l1,
            **common,
        )
        self.log(
            f"{split}/mse",
            normalized_mse,
            **common,
        )
        self.log(
            f"{split}/pcc",
            pcc,
            prog_bar=True,
            **common,
        )
        self.log(
            f"{split}/ccc",
            ccc,
            prog_bar=True,
            **common,
        )
        self.log(
            f"{split}/physical_mae",
            physical_mae,
            **common,
        )
        self.log(
            f"{split}/mean_field_bias",
            mean_field_bias,
            **common,
        )
        self.log(
            f"{split}/strong_field_mae",
            strong_field_mae,
            **common,
        )
        self.log(
            f"{split}/strong_field_polarity_accuracy",
            strong_field_polarity_accuracy,
            **common,
        )
        self.log(
            f"{split}/strong_field_fraction",
            strong_field_fraction,
            **common,
        )

        return normalized_l1

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "test")

    def configure_optimizers(self) -> Any:
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        # The author release uses constant-rate Adam and no scheduler.
        return [optimizer_g, optimizer_d]

    @torch.no_grad()
    def log_images(
        self,
        batch: dict[str, torch.Tensor],
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        source, target = self._paired_batch(batch)
        generated = self(source)

        return {
            f"visualization/{self.source_modal}/condition": source,
            "visualization/hmi/target": target,
            "visualization/hmi/generated": generated,
        }