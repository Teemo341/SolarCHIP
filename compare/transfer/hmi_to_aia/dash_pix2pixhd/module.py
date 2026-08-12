"""PyTorch Lightning adapter for the Dash et al. Pix2PixHD comparison."""

from __future__ import annotations

from typing import Any

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .networks import (
    GlobalResNetGenerator,
    MultiscaleDiscriminator,
    initialize_pix2pixhd,
)


def _pearson_correlation(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.float().flatten(start_dim=1)
    target = target.float().flatten(start_dim=1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = (
        prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def _inverse_aia_preprocess(
    normalized: torch.Tensor,
    mean: float,
    std: float,
    max_log_value: float,
) -> torch.Tensor:
    """Undo SolarDataset's AIA z-score and log1p for paper-like metrics."""

    log_values = normalized.float() * std + mean
    magnitude = log_values.abs().clamp(max=max_log_value)
    return log_values.sign() * torch.expm1(magnitude)


class DashPix2PixHD(pl.LightningModule):
    """SolarCHIP-compatible, author-code-aligned Dash Pix2PixHD baseline.

    Project configs adapt the released RGB network to one-channel tensors and
    use an identity output because SolarDataset emits z-scores. The defining
    ingredients remain the conditional generator, two-scale PatchGAN, LSGAN,
    and discriminator feature matching.
    """

    def __init__(
        self,
        source_modal: str = "hmi",
        target_modal: str = "0304",
        input_channels: int = 1,
        output_channels: int = 1,
        generator_channels: int = 32,
        discriminator_channels: int = 64,
        n_downsample: int = 4,
        n_residual: int = 9,
        discriminator_layers: int = 3,
        n_discriminators: int = 2,
        norm_type: str = "instance",
        padding_type: str = "reflection",
        output_activation: str = "identity",
        learning_rate: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_feature_matching: float = 10.0,
        lambda_l1: float = 0.0,
        decay_start_epoch: int = 100,
        max_epochs: int = 200,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        metric_max_log_value: float = 20.0,
    ) -> None:
        super().__init__()
        if source_modal != "hmi":
            raise ValueError(
                "SolarCHIP HMI-to-AIA comparisons require source_modal='hmi'"
            )
        if target_modal == source_modal:
            raise ValueError("target_modal must be an AIA channel")
        if max_epochs <= decay_start_epoch:
            raise ValueError("max_epochs must be greater than decay_start_epoch")
        if target_std <= 0:
            raise ValueError("target_std must be positive")

        self.save_hyperparameters()
        self.source_modal = source_modal
        self.target_modal = target_modal
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_feature_matching = lambda_feature_matching
        self.lambda_l1 = lambda_l1
        self.decay_start_epoch = decay_start_epoch
        self.max_epochs_config = max_epochs
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.metric_max_log_value = float(metric_max_log_value)

        self.generator = GlobalResNetGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=generator_channels,
            n_downsample=n_downsample,
            n_residual=n_residual,
            norm_type=norm_type,
            padding_type=padding_type,
            output_activation=output_activation,
        )
        self.discriminator = MultiscaleDiscriminator(
            input_channels=input_channels + output_channels,
            base_channels=discriminator_channels,
            n_layers=discriminator_layers,
            n_discriminators=n_discriminators,
            norm_type=norm_type,
        )
        initialize_pix2pixhd(self.generator)
        initialize_pix2pixhd(self.discriminator)
        self.automatic_optimization = False

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.generator(source.float())

    @staticmethod
    def _adversarial_loss(
        predictions: list[list[torch.Tensor]],
        is_real: bool,
    ) -> torch.Tensor:
        losses = []
        target_value = 1.0 if is_real else 0.0
        for scale in predictions:
            logits = scale[-1]
            labels = torch.full_like(logits, target_value)
            losses.append(F.mse_loss(logits, labels))
        return torch.stack(losses).mean()

    @staticmethod
    def _feature_matching_loss(
        fake_predictions: list[list[torch.Tensor]],
        real_predictions: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        scale_losses = []
        for fake_scale, real_scale in zip(fake_predictions, real_predictions):
            layer_losses = []
            for fake_feature, real_feature in zip(fake_scale[:-1], real_scale[:-1]):
                # Each L1 is already normalized by the layer's element count.
                # Pix2PixHD then sums intermediate layers and averages scales.
                layer_losses.append(F.l1_loss(fake_feature, real_feature.detach()))
            if layer_losses:
                scale_losses.append(torch.stack(layer_losses).sum())
        if not scale_losses:
            return fake_predictions[0][-1].new_zeros(())
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
        generated = self(source)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad(set_to_none=True)
        real_predictions = self.discriminator(torch.cat([source, target], dim=1))
        fake_predictions = self.discriminator(
            torch.cat([source, generated.detach()], dim=1)
        )
        discriminator_real = self._adversarial_loss(real_predictions, is_real=True)
        discriminator_fake = self._adversarial_loss(fake_predictions, is_real=False)
        discriminator_loss = 0.5 * (discriminator_real + discriminator_fake)
        self.manual_backward(discriminator_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_predictions = self.discriminator(torch.cat([source, target], dim=1))
        fake_predictions = self.discriminator(torch.cat([source, generated], dim=1))
        generator_adversarial = self._adversarial_loss(fake_predictions, is_real=True)
        feature_matching = self._feature_matching_loss(
            fake_predictions, real_predictions
        )
        pixel_l1 = F.l1_loss(generated, target)
        generator_loss = (
            generator_adversarial
            + self.lambda_feature_matching * feature_matching
            + self.lambda_l1 * pixel_l1
        )
        self.manual_backward(generator_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

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
        self.log("train/g_adversarial", generator_adversarial.detach(), **common)
        self.log("train/feature_matching", feature_matching.detach(), **common)
        self.log("train/l1", pixel_l1.detach(), **common)

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        split: str,
    ) -> torch.Tensor:
        source, target = self._paired_batch(batch)
        generated = self(source)
        l1 = F.l1_loss(generated, target)
        mse = F.mse_loss(generated, target)
        pcc = _pearson_correlation(generated, target)

        generated_raw = _inverse_aia_preprocess(
            generated,
            self.target_mean,
            self.target_std,
            self.metric_max_log_value,
        )
        target_raw = _inverse_aia_preprocess(
            target,
            self.target_mean,
            self.target_std,
            self.metric_max_log_value,
        )
        dimensions = tuple(range(1, target_raw.ndim))
        target_flux = target_raw.sum(dim=dimensions).clamp_min(1e-8)
        per_image_relative_flux = (
            generated_raw.sum(dim=dimensions) - target_flux
        ) / target_flux
        signed_relative_flux = per_image_relative_flux.mean()
        absolute_relative_flux = per_image_relative_flux.abs().mean()
        relative_error = (
            generated_raw - target_raw
        ).abs() / target_raw.abs().clamp_min(1e-6)
        valid = target_raw > 1e-6
        valid_count = valid.flatten(start_dim=1).float().sum(dim=1)
        valid_images = (valid_count > 0).float()
        ppe10_per_image = ((relative_error < 0.10) & valid).flatten(
            start_dim=1
        ).float().sum(dim=1) / valid_count.clamp_min(1.0)
        ppe10 = (ppe10_per_image * valid_images).sum() / valid_images.sum().clamp_min(
            1.0
        )

        batch_size = source.shape[0]
        common = dict(
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(f"{split}/loss", l1, prog_bar=True, **common)
        self.log(f"{split}/l1", l1, **common)
        self.log(f"{split}/mse", mse, **common)
        self.log(f"{split}/pcc", pcc, prog_bar=True, **common)
        self.log(f"{split}/ppe10", ppe10, **common)
        self.log(f"{split}/signed_relative_flux", signed_relative_flux, **common)
        self.log(f"{split}/abs_relative_flux", absolute_relative_flux, **common)
        return l1

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

        def decay(epoch: int) -> float:
            if epoch <= self.decay_start_epoch:
                return 1.0
            remaining = self.max_epochs_config - epoch
            duration = self.max_epochs_config - self.decay_start_epoch
            return max(0.0, remaining / duration)

        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=decay)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=decay)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    @torch.no_grad()
    def log_images(
        self, batch: dict[str, torch.Tensor], **_: Any
    ) -> dict[str, torch.Tensor]:
        source, target = self._paired_batch(batch)
        generated = self(source)
        return {
            f"visualization/{self.source_modal}/condition": source,
            f"visualization/{self.target_modal}/target": target,
            f"visualization/{self.target_modal}/generated": generated,
        }
