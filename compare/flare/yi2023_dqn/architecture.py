"""Paper-specific DenseNet used by the Yi et al. (2023) comparison."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


CLASS_NAMES = ("0AB", "C", "M", "X")
HEAD_NAMES = ("Cplus", "Mplus", "Xplus")


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    """Map grouped 0AB/C/M/X labels to cumulative C+/M+/X+ targets."""

    labels = labels.long().reshape(-1)
    if labels.numel() and ((labels < 0).any() or (labels > 3).any()):
        raise ValueError("Yi2023 labels must be grouped 0AB/C/M/X IDs 0..3")
    thresholds = torch.arange(1, 4, device=labels.device)
    return labels.unsqueeze(1).ge(thresholds).long()


def decode_cumulative_actions(actions: torch.Tensor) -> torch.Tensor:
    """Decode by the highest positive head without monotone projection."""

    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"Expected actions shaped [B,3], got {tuple(actions.shape)}")
    levels = torch.tensor((1, 2, 3), device=actions.device, dtype=torch.long)
    return (actions.long() * levels).amax(dim=1)


def cumulative_inconsistency(actions: torch.Tensor) -> torch.Tensor:
    """Flag predictions that violate C+ >= M+ >= X+."""

    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"Expected actions shaped [B,3], got {tuple(actions.shape)}")
    c_plus, m_plus, x_plus = actions.bool().unbind(dim=1)
    return (m_plus & ~c_plus) | (x_plus & ~m_plus)


class YiDenseBlock(nn.Module):
    """BN-ReLU-1x1-BN-ReLU-3x3-concat-average-pool block."""

    def __init__(self, in_channels: int, block_index: int) -> None:
        super().__init__()
        bottleneck_channels = 13 * block_index
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels, 39, kernel_size=3, padding=1
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = in_channels + 39

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(F.relu(self.norm1(inputs), inplace=False))
        new_features = self.conv3(
            F.relu(self.norm2(new_features), inplace=False)
        )
        return self.pool(torch.cat((inputs, new_features), dim=1))


class YiDenseNet(nn.Module):
    """The exact 512-pixel topology illustrated in Yi et al. (2023).

    Its five pooled blocks produce 65/104/143/182/221 channels.  The final
    BN and 2x2 average pool produce a 221x4x4 = 3536 feature vector.  The
    SolarCHIP adaptation replaces the original one binary head with three
    binary heads while retaining a shared trunk.
    """

    stage_channels = (65, 104, 143, 182, 221)
    feature_dim = 221 * 4 * 4

    def __init__(self, num_heads: int = 3) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = int(num_heads)
        self.stem_conv = nn.Conv2d(1, 26, kernel_size=3, padding=1)
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks: list[YiDenseBlock] = []
        channels = 26
        for block_index in range(1, 6):
            block = YiDenseBlock(channels, block_index)
            blocks.append(block)
            channels = block.out_channels
        self.blocks = nn.ModuleList(blocks)
        if channels != 221:
            raise AssertionError(f"Yi trunk must end with 221 channels, got {channels}")

        self.final_norm = nn.BatchNorm2d(channels)
        self.final_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.heads = nn.ModuleList(
            nn.Linear(self.feature_dim, 2) for _ in range(self.num_heads)
        )

    def forward_features(
        self,
        inputs: torch.Tensor,
        *,
        return_stage_shapes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                "YiDenseNet requires [B,1,512,512], got " f"{tuple(inputs.shape)}"
            )
        if inputs.shape[-2:] != (512, 512):
            raise ValueError(
                "YiDenseNet's 3536-dimensional head requires 512x512 input, got "
                f"{tuple(inputs.shape[-2:])}"
            )

        # The paper diagram specifies Conv3x3 followed directly by MaxPool2.
        features = self.stem_pool(self.stem_conv(inputs))
        stage_shapes: list[tuple[int, ...]] = [tuple(features.shape[1:])]
        for block in self.blocks:
            features = block(features)
            stage_shapes.append(tuple(features.shape[1:]))
        features = self.final_pool(self.final_norm(features)).flatten(start_dim=1)
        if features.shape[1] != self.feature_dim:
            raise AssertionError(
                f"Yi trunk must flatten to {self.feature_dim}, got {features.shape[1]}"
            )
        if return_stage_shapes:
            return features, tuple(stage_shapes)
        return features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        return torch.stack([head(features) for head in self.heads], dim=1)


__all__ = [
    "CLASS_NAMES",
    "HEAD_NAMES",
    "YiDenseBlock",
    "YiDenseNet",
    "cumulative_inconsistency",
    "cumulative_targets",
    "decode_cumulative_actions",
]
