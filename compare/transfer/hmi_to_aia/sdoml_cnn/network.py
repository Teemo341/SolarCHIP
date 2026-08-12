"""The deterministic HMI-to-AIA CNN baseline described by Galvez et al."""

import torch
import torch.nn.functional as F
from torch import nn


class ConvReluBatchNorm(nn.Module):
    """128-filter intermediate layer in the order stated by the paper."""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SDOMLCNN(nn.Module):
    """3/7/11-layer Galvez SDOML baseline, with 11 layers by default.

    Total convolution count includes the initial 7x7 convolution and the final
    output convolution. Therefore an 11-layer model contains nine 3x3 hidden
    body convolutions, matching the paper's ten hidden convolutions in total.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 11,
    ) -> None:
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be at least 3")
        self.stem = ConvReluBatchNorm(
            input_channels,
            hidden_channels,
            kernel_size=7,
            stride=2,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.body = nn.Sequential(
            *(
                ConvReluBatchNorm(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                )
                for _ in range(num_layers - 2)
            )
        )
        self.head = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_size = inputs.shape[-2:]
        features = self.pool(self.stem(inputs))
        features = self.body(features)
        prediction = self.head(features)
        return F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
