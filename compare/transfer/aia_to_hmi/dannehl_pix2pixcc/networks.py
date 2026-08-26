"""Networks for the Dannehl, Delouille & Barra Pix2PixCC baseline.

The implementation is self-contained, but follows the architecture in the
authors' archived ``pix2pixCC2`` release: a 4-down/9-residual/4-up generator
and one 70x70 PatchGAN discriminator that exposes intermediate features.

InstanceNorm is used with ``eps = 1e-2`` (instead of the PyTorch default
1e-5) so that the per-layer backward amplification stays bounded on the
smooth solar images; see ``_stable_instance_norm``.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


def _stable_instance_norm(channels: int) -> nn.Module:
    """Return an InstanceNorm2d layer with a deliberately large ``eps``.

    ``eps`` is much larger than the PyTorch default (1e-5). The backward
    pass of instance normalization scales the gradient by
    ``1 / sqrt(var + eps)``. Solar images contain large nearly constant
    regions, so after a few training steps many generator channels have
    spatial variance far below 1e-4; with a small ``eps`` every one of the
    ~26 normalization layers would amplify the backward gradient by up to
    100x, and the product over the residual blocks overflows float32
    (~1e38), producing Inf/NaN gradients. ``eps = 1e-2`` caps the per-layer
    amplification at 10x, which keeps the total amplification comfortably
    inside float32 while only suppressing channels that are already nearly
    constant.
    """

    return nn.InstanceNorm2d(
        num_features=channels,
        eps=1e-2,
        affine=False,
    )


def _normalization(name: str):
    name = name.lower()

    if name in {
        "instance",
        "instancenorm",
        "instancenorm2d",
        "stable_instance",
        "group_instance",
    }:
        return _stable_instance_norm

    if name in {
        "batch",
        "batchnorm",
        "batchnorm2d",
    }:
        return partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True,
            eps=1e-4,
        )

    if name in {"identity", "none"}:
        return lambda _channels: nn.Identity()

    raise ValueError(f"Unsupported normalization: {name}")


def _padding(name: str):
    name = name.lower()

    if name in {"replication", "replicate"}:
        return nn.ReplicationPad2d

    if name in {"reflection", "reflect"}:
        return nn.ReflectionPad2d

    if name in {"zero", "zeros"}:
        return nn.ZeroPad2d

    raise ValueError(f"Unsupported padding: {name}")


def _output_activation(name: str) -> nn.Module:
    name = name.lower()

    if name in {"identity", "linear", "none"}:
        return nn.Identity()

    if name == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unsupported output activation: {name}")


class Mish(nn.Module):
    """Mish activation used by the authors' released generator."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.tanh(F.softplus(inputs))


class ResidualBlock(nn.Module):
    """Two-convolution residual block from ``pix2pixCC2``."""

    def __init__(
        self,
        channels: int,
        norm,
        pad,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            pad(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            norm(channels),
            Mish(),
            pad(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            norm(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class Pix2PixCCGenerator(nn.Module):
    """Author-code-aligned 4-down/9-residual/4-up ResNet generator."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        n_downsample: int = 4,
        n_residual: int = 9,
        input_kernel_size: int = 7,
        downsample_kernel_size: int = 5,
        norm_type: str = "instance",
        padding_type: str = "replication",
        output_activation: str = "identity",
    ) -> None:
        super().__init__()

        if input_channels < 1 or output_channels < 1:
            raise ValueError(
                "input_channels and output_channels must be positive"
            )

        if base_channels < 1:
            raise ValueError("base_channels must be positive")

        if n_downsample < 1:
            raise ValueError("n_downsample must be positive")

        if n_residual < 0:
            raise ValueError("n_residual must be non-negative")

        if input_kernel_size < 1 or input_kernel_size % 2 == 0:
            raise ValueError(
                "input_kernel_size must be a positive odd integer"
            )

        if (
            downsample_kernel_size < 1
            or downsample_kernel_size % 2 == 0
        ):
            raise ValueError(
                "downsample_kernel_size must be a positive odd integer"
            )

        norm = _normalization(norm_type)
        pad = _padding(padding_type)

        channels = base_channels

        layers: list[nn.Module] = [
            pad(input_kernel_size // 2),
            nn.Conv2d(
                input_channels,
                channels,
                kernel_size=input_kernel_size,
                padding=0,
            ),
            norm(channels),
            Mish(),
        ]

        for _ in range(n_downsample):
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=downsample_kernel_size,
                        stride=2,
                        padding=downsample_kernel_size // 2,
                    ),
                    norm(channels * 2),
                    Mish(),
                ]
            )
            channels *= 2

        layers.extend(
            ResidualBlock(
                channels,
                norm,
                pad,
            )
            for _ in range(n_residual)
        )

        for _ in range(n_downsample):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        channels // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    norm(channels // 2),
                    Mish(),
                ]
            )
            channels //= 2

        # The released implementation fixes the final convolution at 7x7 and
        # has no output activation. SolarCHIP exposes identity/tanh explicitly.
        layers.extend(
            [
                pad(3),
                nn.Conv2d(
                    channels,
                    output_channels,
                    kernel_size=7,
                    padding=0,
                ),
                _output_activation(output_activation),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class PatchDiscriminator(nn.Module):
    """Single-scale 70x70 PatchGAN with four feature-matching layers."""

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        if input_channels < 1 or base_channels < 1:
            raise ValueError(
                "input_channels and base_channels must be positive"
            )

        def activation() -> nn.Module:
            return nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        base_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    activation(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        base_channels,
                        base_channels * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    _stable_instance_norm(base_channels * 2),
                    activation(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        base_channels * 2,
                        base_channels * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    _stable_instance_norm(base_channels * 4),
                    activation(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        base_channels * 4,
                        base_channels * 8,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                    ),
                    _stable_instance_norm(base_channels * 8),
                    activation(),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        base_channels * 8,
                        1,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                    )
                ),
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> list[torch.Tensor]:
        features = []
        output = inputs

        for block in self.blocks:
            output = block(output)
            features.append(output)

        return features


def initialize_pix2pixcc(module: nn.Module) -> None:
    """Apply the released model's normal-0.02 convolution initialization."""

    for layer in module.modules():
        if isinstance(
            layer,
            (nn.Conv2d, nn.ConvTranspose2d),
        ):
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=0.02,
            )

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        elif isinstance(layer, nn.BatchNorm2d):
            if layer.weight is not None:
                nn.init.normal_(
                    layer.weight,
                    mean=1.0,
                    std=0.02,
                )

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)