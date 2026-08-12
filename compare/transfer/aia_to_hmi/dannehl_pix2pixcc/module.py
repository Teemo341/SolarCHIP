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
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean per-image Pearson correlation, computed in float32."""

    prediction = prediction.float().flatten(start_dim=1)
    target = target.float().flatten(start_dim=1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = (
        prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def _concordance_correlation(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean per-image Lin concordance correlation coefficient (CCC)."""

    prediction = prediction.float().flatten(start_dim=1)
    target = target.float().flatten(start_dim=1)
    prediction_mean = prediction.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    covariance = (centered_prediction * centered_target).mean(dim=1)
    prediction_variance = centered_prediction.square().mean(dim=1)
    target_variance = centered_target.square().mean(dim=1)
    mean_difference = (prediction_mean - target_mean).squeeze(1).square()
    denominator = prediction_variance + target_variance + mean_difference
    return (2.0 * covariance / denominator.clamp_min(1e-8)).mean()


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

    The generator objective is the paper's LSGAN + discriminator feature
    matching + four-scale CCC inspector. Validation deliberately monitors a
    stable normalized-space L1 metric instead of the adversarial objective.
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
                "The supplied SolarCHIP comparison is single-input/single-output"
            )
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if min(lambda_lsgan, lambda_feature_matching, lambda_cc) < 0:
            raise ValueError("loss weights must be non-negative")
        if n_cc_scales < 1:
            raise ValueError("n_cc_scales must be positive")
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
    def _adversarial_loss(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
        labels = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
        return 0.5 * F.mse_loss(logits, labels)

    @staticmethod
    def _feature_matching_loss(
        fake_features: list[torch.Tensor],
        real_features: list[torch.Tensor],
    ) -> torch.Tensor:
        # The paper defines T=4 discriminator feature layers. The author's
        # archive also included the final logit in its loop; this adaptation
        # intentionally follows the paper semantic and excludes that logit.
        layer_losses = [
            F.l1_loss(fake, real.detach())
            for fake, real in zip(fake_features[:-1], real_features[:-1])
        ]
        if not layer_losses:
            return fake_features[-1].new_zeros(())
        return torch.stack(layer_losses).sum()

    def _multiscale_ccc_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        scale_losses = []
        scaled_prediction = prediction
        scaled_target = target
        for scale_index in range(self.n_cc_scales):
            scale_losses.append(
                1.0 - _concordance_correlation(scaled_prediction, scaled_target)
            )
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
        return source, target

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        source, target = self._paired_batch(batch)
        optimizer_g, optimizer_d = self.optimizers()

        # Recompute G for the generator step rather than retaining its full
        # 1024x1024 graph across the discriminator update. This trades compute
        # for a substantially lower peak-memory requirement.
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            generated_for_d = self(source)
        real_features = self.discriminator(torch.cat([source, target], dim=1))
        fake_features = self.discriminator(torch.cat([source, generated_for_d], dim=1))
        discriminator_real = self._adversarial_loss(real_features[-1], is_real=True)
        discriminator_fake = self._adversarial_loss(fake_features[-1], is_real=False)
        discriminator_loss = discriminator_real + discriminator_fake
        self.manual_backward(discriminator_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad(set_to_none=True)
        generated = self(source)
        with torch.no_grad():
            real_features = self.discriminator(torch.cat([source, target], dim=1))
        fake_features = self.discriminator(torch.cat([source, generated], dim=1))
        generator_adversarial = self._adversarial_loss(fake_features[-1], is_real=True)
        feature_matching = self._feature_matching_loss(fake_features, real_features)
        ccc_loss = self._multiscale_ccc_loss(generated, target)
        generator_loss = (
            self.lambda_lsgan * generator_adversarial
            + self.lambda_feature_matching * feature_matching
            + self.lambda_cc * ccc_loss
        )
        self.manual_backward(generator_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        normalized_l1 = F.l1_loss(generated.detach(), target)
        batch_size = source.shape[0]
        common = dict(
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log("train/g_loss", generator_loss.detach(), prog_bar=True, **common)
        self.log("train/d_loss", discriminator_loss.detach(), prog_bar=True, **common)
        self.log("train/g_lsgan", generator_adversarial.detach(), **common)
        self.log("train/feature_matching", feature_matching.detach(), **common)
        self.log("train/ccc_loss", ccc_loss.detach(), **common)
        self.log("train/l1", normalized_l1, **common)

    @staticmethod
    def _masked_image_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.float().flatten(start_dim=1)
        mask = mask.flatten(start_dim=1)
        counts = mask.float().sum(dim=1)
        valid_images = counts > 0
        per_image = (values * mask.float()).sum(dim=1) / counts.clamp_min(1.0)
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

        # val/loss is intentionally deterministic and non-adversarial so that
        # checkpoints remain comparable across discriminator oscillations.
        normalized_l1 = F.l1_loss(generated, target)
        normalized_mse = F.mse_loss(generated, target)
        pcc = _pearson_correlation(generated, target)
        ccc = _concordance_correlation(generated, target)

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

        strong_mask = target_raw.abs() >= self.strong_field_threshold
        strong_field_mae = self._masked_image_average(absolute_error, strong_mask)
        polarity_correct = generated_raw.sign() == target_raw.sign()
        strong_field_polarity_accuracy = self._masked_image_average(
            polarity_correct.float(), strong_mask
        )
        strong_field_fraction = (
            strong_mask.float().flatten(start_dim=1).mean(dim=1).mean()
        )

        batch_size = source.shape[0]
        common = dict(
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(f"{split}/loss", normalized_l1, prog_bar=True, **common)
        self.log(f"{split}/l1", normalized_l1, **common)
        self.log(f"{split}/mse", normalized_mse, **common)
        self.log(f"{split}/pcc", pcc, prog_bar=True, **common)
        self.log(f"{split}/ccc", ccc, prog_bar=True, **common)
        self.log(f"{split}/physical_mae", physical_mae, **common)
        self.log(f"{split}/mean_field_bias", mean_field_bias, **common)
        self.log(f"{split}/strong_field_mae", strong_field_mae, **common)
        self.log(
            f"{split}/strong_field_polarity_accuracy",
            strong_field_polarity_accuracy,
            **common,
        )
        self.log(f"{split}/strong_field_fraction", strong_field_fraction, **common)
        return normalized_l1

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        self, batch: dict[str, torch.Tensor], **_: Any
    ) -> dict[str, torch.Tensor]:
        source, target = self._paired_batch(batch)
        generated = self(source)
        return {
            f"visualization/{self.source_modal}/condition": source,
            "visualization/hmi/target": target,
            "visualization/hmi/generated": generated,
        }
