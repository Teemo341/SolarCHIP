"""Self-contained convolutional I2IwFiLM networks for 1024-pixel solar images.

The paper describes a paired guidance-vector predictor, a source-only guidance
predictor, and additive feature modulation.  This project adaptation keeps
those mechanisms while replacing the authors' released I2IFormer internals
with a memory-conscious convolutional guided U-Net suitable for SolarCHIP's
1024 x 1024 comparisons.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class AdditiveFiLM(nn.Module):
    """Beta-only FiLM: add one learned guidance bias per feature channel."""

    def __init__(self, channels: int, guidance_dim: int) -> None:
        super().__init__()
        if channels <= 0 or guidance_dim <= 0:
            raise ValueError("channels and guidance_dim must be positive")
        self.channels = int(channels)
        self.guidance_dim = int(guidance_dim)
        self.to_beta = nn.Linear(self.guidance_dim, self.channels)
        nn.init.normal_(self.to_beta.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, features: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        if guidance.ndim != 2:
            raise ValueError(
                f"guidance must have shape [B, D], received {tuple(guidance.shape)}"
            )
        if guidance.shape[0] != features.shape[0]:
            raise ValueError("features and guidance must have the same batch size")
        if guidance.shape[1] != self.guidance_dim:
            raise ValueError(
                f"expected guidance dimension {self.guidance_dim}, "
                f"received {guidance.shape[1]}"
            )
        beta = self.to_beta(guidance).to(dtype=features.dtype)
        return features + beta[:, :, None, None]


class GuidedConvBlock(nn.Module):
    """Two residual convolution layers, each modulated by additive FiLM."""

    def __init__(self, in_channels: int, out_channels: int, guidance_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film1 = AdditiveFiLM(out_channels, guidance_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film2 = AdditiveFiLM(out_channels, guidance_dim)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, features: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        residual = self.residual(features)
        features = self.conv1(features)
        features = self.norm1(features)
        features = F.silu(self.film1(features, guidance), inplace=False)
        features = self.conv2(features)
        features = self.norm2(features)
        features = F.silu(self.film2(features, guidance), inplace=False)
        return (features + residual) * (2.0**-0.5)


class GuidedDownsample(nn.Module):
    """Stride-two convolution with beta-only guidance modulation."""

    def __init__(self, in_channels: int, out_channels: int, guidance_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film = AdditiveFiLM(out_channels, guidance_dim)

    def forward(self, features: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        features = self.norm(self.conv(features))
        return F.silu(self.film(features, guidance), inplace=False)


class GuidedUpsample(nn.Module):
    """Bilinear resize followed by a guided convolution."""

    def __init__(self, in_channels: int, out_channels: int, guidance_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film = AdditiveFiLM(out_channels, guidance_dim)

    def forward(
        self,
        features: torch.Tensor,
        guidance: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        features = F.interpolate(
            features, size=output_size, mode="bilinear", align_corners=False
        )
        features = self.norm(self.conv(features))
        return F.silu(self.film(features, guidance), inplace=False)


class GuidedUNet(nn.Module):
    """Convolutional U-Net whose learned feature stages receive one guidance vector."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 8),
        guidance_dim: int = 256,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")
        multipliers = tuple(int(value) for value in channel_multipliers)
        if len(multipliers) < 2 or any(value <= 0 for value in multipliers):
            raise ValueError("channel_multipliers must contain at least two positives")

        widths = [base_channels * multiplier for multiplier in multipliers]
        self.guidance_dim = int(guidance_dim)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        previous = input_channels
        for level, width in enumerate(widths):
            self.encoder_blocks.append(GuidedConvBlock(previous, width, guidance_dim))
            if level < len(widths) - 1:
                self.downsamples.append(
                    GuidedDownsample(width, widths[level + 1], guidance_dim)
                )
                previous = widths[level + 1]

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.upsamples.append(GuidedUpsample(current, skip_width, guidance_dim))
            self.decoder_blocks.append(
                GuidedConvBlock(skip_width * 2, skip_width, guidance_dim)
            )
            current = skip_width

        # Deliberately unbounded: SolarCHIP targets are z-scores, not [-1, 1].
        self.output = nn.Conv2d(widths[0], output_channels, 1)

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        features = source
        for level, block in enumerate(self.encoder_blocks):
            features = block(features, guidance)
            skips.append(features)
            if level < len(self.downsamples):
                features = self.downsamples[level](features, guidance)

        # The last encoder feature is the bottleneck, not a decoder skip.
        skips.pop()
        for upsample, block, skip in zip(
            self.upsamples, self.decoder_blocks, reversed(skips)
        ):
            features = upsample(features, guidance, skip.shape[-2:])
            features = block(torch.cat((features, skip), dim=1), guidance)
        return self.output(features)


class GuidanceResidualBlock(nn.Module):
    """Residual block used by the paired and source guidance encoders."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = F.silu(self.norm1(self.conv1(features)), inplace=False)
        features = self.norm2(self.conv2(features))
        return F.silu((features + residual) * (2.0**-0.5), inplace=False)


class _GuidanceEncoder(nn.Module):
    """Pixel-unshuffle encoder shared in topology, but not weights, by both paths."""

    def __init__(
        self,
        input_channels: int,
        guidance_dim: int,
        base_channels: int,
        residual_blocks: int,
        unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()
        if residual_blocks < 0:
            raise ValueError("residual_blocks cannot be negative")
        if unshuffle_factor <= 0:
            raise ValueError("unshuffle_factor must be positive")
        self.unshift_factor = int(unshuffle_factor)
        shuffled_channels = input_channels * self.unshift_factor**2
        self.stem = nn.Sequential(
            nn.Conv2d(shuffled_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            *(GuidanceResidualBlock(base_channels) for _ in range(residual_blocks))
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(base_channels * 4, guidance_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must have shape [B, C, H, W], got {image.shape}")
        factor = self.unshift_factor
        if image.shape[-2] % factor or image.shape[-1] % factor:
            raise ValueError(
                f"guidance input spatial dimensions must be divisible by {factor}"
            )
        features = F.pixel_unshuffle(image, factor)
        features = self.residual(self.stem(features))
        features = self.down2(self.down1(features))
        return self.projection(self.pool(features).flatten(1))


class PairGuidanceEncoder(nn.Module):
    """Stage-one teacher that extracts guidance from aligned source/target pairs."""

    def __init__(
        self,
        source_channels: int = 1,
        target_channels: int = 1,
        guidance_dim: int = 256,
        base_channels: int = 64,
        residual_blocks: int = 6,
        unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = _GuidanceEncoder(
            input_channels=source_channels + target_channels,
            guidance_dim=guidance_dim,
            base_channels=base_channels,
            residual_blocks=residual_blocks,
            unshuffle_factor=unshuffle_factor,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[0] != target.shape[0] or source.shape[-2:] != target.shape[-2:]:
            raise ValueError("source and target must be spatially aligned pairs")
        return self.encoder(torch.cat((source, target), dim=1))


class SourceGuidancePredictor(nn.Module):
    """Deployable source-only guidance encoder followed by the paper's MLP idea."""

    def __init__(
        self,
        source_channels: int = 1,
        guidance_dim: int = 256,
        base_channels: int = 64,
        residual_blocks: int = 6,
        unshuffle_factor: int = 4,
        mlp_hidden_dims: Sequence[int] = (512, 512, 512),
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(value) for value in mlp_hidden_dims)
        if any(value <= 0 for value in hidden_dims):
            raise ValueError("mlp_hidden_dims must be positive")
        self.encoder = _GuidanceEncoder(
            input_channels=source_channels,
            guidance_dim=guidance_dim,
            base_channels=base_channels,
            residual_blocks=residual_blocks,
            unshuffle_factor=unshuffle_factor,
        )
        layers: list[nn.Module] = []
        previous = guidance_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, guidance_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(source))
