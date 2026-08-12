"""Self-contained Pix2PixHD networks used by the Dash comparison.

This follows the concrete architecture in the authors' released ``pix2pixHD``
code: one global ResNet generator and scale-separated PatchGAN discriminators.
The paper prose additionally describes a local enhancer, but that component is
absent from the released implementation; see README.md.
"""

from functools import partial

import torch
from torch import nn


def _normalization(name: str):
    name = name.lower()
    if name in {"instance", "instancenorm", "instancenorm2d"}:
        return partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if name in {"batch", "batchnorm", "batchnorm2d"}:
        return partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if name in {"none", "identity"}:
        return lambda _channels: nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


def _padding(name: str):
    name = name.lower()
    if name == "reflection":
        return nn.ReflectionPad2d
    if name == "replication":
        return nn.ReplicationPad2d
    if name == "zero":
        return nn.ZeroPad2d
    raise ValueError(f"Unsupported padding: {name}")


def _output_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "none", "linear"}:
        return nn.Identity()
    raise ValueError(f"Unsupported output activation: {name}")


class ResidualBlock(nn.Module):
    """Two-convolution residual block from the released Dash implementation."""

    def __init__(self, channels: int, norm, pad) -> None:
        super().__init__()
        self.block = nn.Sequential(
            pad(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            norm(channels),
            nn.ReLU(inplace=True),
            pad(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            norm(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class GlobalResNetGenerator(nn.Module):
    """4-down/9-residual/4-up Pix2PixHD global generator.

    ``output_activation='identity'`` is the SolarCHIP default because the
    existing dataloader supplies log1p + z-score values, which are not bounded
    to [-1, 1]. Set it to ``tanh`` only with matching range-normalized data.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        n_downsample: int = 4,
        n_residual: int = 9,
        norm_type: str = "instance",
        padding_type: str = "reflection",
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        if n_downsample < 1:
            raise ValueError("n_downsample must be positive")
        if n_residual < 0:
            raise ValueError("n_residual must be non-negative")

        norm = _normalization(norm_type)
        pad = _padding(padding_type)
        channels = base_channels
        layers = [
            pad(3),
            nn.Conv2d(input_channels, channels, kernel_size=7, padding=0),
            norm(channels),
            nn.ReLU(inplace=True),
        ]

        for _ in range(n_downsample):
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    norm(channels * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2

        layers.extend(ResidualBlock(channels, norm, pad) for _ in range(n_residual))

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
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2

        layers.extend(
            [
                pad(3),
                nn.Conv2d(channels, output_channels, kernel_size=7, padding=0),
                _output_activation(output_activation),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that exposes intermediate feature maps."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
        max_channels: int = 512,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        norm = _normalization(norm_type)

        blocks = [
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]
        previous = base_channels
        for layer_idx in range(1, n_layers):
            current = min(base_channels * (2**layer_idx), max_channels)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, current, 4, stride=2, padding=1),
                    norm(current),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            previous = current

        current = min(base_channels * (2**n_layers), max_channels)
        blocks.append(
            nn.Sequential(
                nn.Conv2d(previous, current, 4, stride=1, padding=1),
                norm(current),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        blocks.append(nn.Sequential(nn.Conv2d(current, 1, 4, stride=1, padding=1)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = []
        output = inputs
        for block in self.blocks:
            output = block(output)
            features.append(output)
        return features


class MultiscaleDiscriminator(nn.Module):
    """Two-scale PatchGAN by default, as used in Dash et al."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 2,
        norm_type: str = "instance",
    ) -> None:
        super().__init__()
        if n_discriminators < 1:
            raise ValueError("n_discriminators must be positive")
        self.discriminators = nn.ModuleList(
            PatchDiscriminator(
                input_channels=input_channels,
                base_channels=base_channels,
                n_layers=n_layers,
                norm_type=norm_type,
            )
            for _ in range(n_discriminators)
        )
        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(self, inputs: torch.Tensor) -> list[list[torch.Tensor]]:
        outputs = []
        scale_input = inputs
        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(scale_input))
            if index + 1 < len(self.discriminators):
                scale_input = self.downsample(scale_input)
        return outputs


def initialize_pix2pixhd(module: nn.Module) -> None:
    """Initialize convolutions with the normal scheme used by Pix2PixHD."""

    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            if layer.weight is not None:
                nn.init.normal_(layer.weight, mean=1.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
