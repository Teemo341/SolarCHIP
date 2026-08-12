"""PyTorch Lightning adapter for the Galvez et al. SDOML CNN baseline."""

from __future__ import annotations

from typing import Any

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .network import SDOMLCNN


def _inverse_aia_preprocess(
    normalized: torch.Tensor,
    mean: float,
    std: float,
    max_log_value: float,
) -> torch.Tensor:
    log_values = normalized.float() * std + mean
    magnitude = log_values.abs().clamp(max=max_log_value)
    return log_values.sign() * torch.expm1(magnitude)


def _masked_image_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average valid pixels per image, then average non-empty images."""

    valid_flat = valid.flatten(start_dim=1).float()
    valid_count = valid_flat.sum(dim=1)
    per_image = (values.flatten(start_dim=1) * valid_flat).sum(
        dim=1
    ) / valid_count.clamp_min(1.0)
    valid_images = (valid_count > 0).float()
    return (per_image * valid_images).sum() / valid_images.sum().clamp_min(1.0)


class GalvezSDOMLCNN(pl.LightningModule):
    """Single-band LOS adaptation of the Galvez SDOML HMI-to-AIA CNN.

    The paper used three HMI vector components and predicted nine AIA channels
    jointly at 256x256. SolarCHIP supplies one LOS channel and the requested
    experiments train one target at a time, so project configs instantiate a
    1-to-1 version while preserving topology, MSE, SGD/Nesterov, and schedule.
    """

    def __init__(
        self,
        source_modal: str = "hmi",
        target_modal: str = "0304",
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 11,
        learning_rate: float = 1e-3,
        momentum: float = 0.99,
        weight_decay: float = 1e-8,
        lr_step_size: int = 5,
        lr_gamma: float = 0.1,
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
        if target_std <= 0:
            raise ValueError("target_std must be positive")
        self.save_hyperparameters()
        self.source_modal = source_modal
        self.target_modal = target_modal
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.metric_max_log_value = float(metric_max_log_value)
        self.network = SDOMLCNN(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.network(source.float())

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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        source, target = self._paired_batch(batch)
        prediction = self(source)
        loss = F.mse_loss(prediction, target)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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
        prediction = self(source)
        mse = F.mse_loss(prediction, target)
        mae = F.l1_loss(prediction, target)

        prediction_raw = _inverse_aia_preprocess(
            prediction,
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
        relative_error = (
            prediction_raw - target_raw
        ).abs() / target_raw.abs().clamp_min(1e-6)
        valid = target_raw > 1e-6
        normalized_absolute_error = _masked_image_mean(relative_error, valid)
        good_10 = _masked_image_mean((relative_error < 0.10).float(), valid)
        good_20 = _masked_image_mean((relative_error < 0.20).float(), valid)
        good_50 = _masked_image_mean((relative_error < 0.50).float(), valid)

        common = dict(
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        self.log(f"{split}/loss", mse, prog_bar=True, **common)
        self.log(f"{split}/mse", mse, **common)
        self.log(f"{split}/mae", mae, **common)
        self.log(
            f"{split}/normalized_absolute_error", normalized_absolute_error, **common
        )
        self.log(f"{split}/good_pixels_10", good_10, **common)
        self.log(f"{split}/good_pixels_20", good_20, **common)
        self.log(f"{split}/good_pixels_50", good_50, prog_bar=True, **common)
        return mse

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
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
        prediction = self(source)
        return {
            f"visualization/{self.source_modal}/condition": source,
            f"visualization/{self.target_modal}/target": target,
            f"visualization/{self.target_modal}/generated": prediction,
        }
