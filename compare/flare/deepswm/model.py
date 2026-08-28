"""DeepSWM-derived HMI-only network components.

This preserves the published SSE/DCSM/ST-SSM, LT-SSM and mixing-SSM topology,
but parameterizes it for one HMI modality and a configurable number of daily
frames.  The long-history image encoder keeps the published SparseMAE encoder
shape (patch embedding, 8 Transformer blocks, width 128) while intentionally
removing masking and the decoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from s5 import S5
except ImportError as error:  # pragma: no cover - exercised only in bad envs
    raise ImportError(
        "DeepSWM requires s5-pytorch==0.2.1. Install the pinned dependency "
        "without changing the existing torch/torchvision versions."
    ) from error

try:
    from timm.models.vision_transformer import Block, PatchEmbed
except ImportError as error:  # pragma: no cover - exercised only in bad envs
    raise ImportError(
        "DeepSWM requires timm==1.0.15 for its SparseMAE encoder blocks."
    ) from error


def _sincos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2:
        raise ValueError("one-dimensional sine/cosine width must be even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    phase = positions.reshape(-1, 1).to(torch.float64) * omega.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=1).to(torch.float32)


def build_2d_sincos_position_embedding(
    embed_dim: int, grid_size: int, include_cls_token: bool = True
) -> torch.Tensor:
    """Create the fixed 2-D sine/cosine embedding used by MAE."""

    if embed_dim % 4:
        raise ValueError("two-dimensional sine/cosine width must divide by four")
    grid_h, grid_w = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    embedding = torch.cat(
        (
            _sincos_1d(embed_dim // 2, grid_w.reshape(-1)),
            _sincos_1d(embed_dim // 2, grid_h.reshape(-1)),
        ),
        dim=1,
    )
    if include_cls_token:
        embedding = torch.cat((torch.zeros(1, embed_dim), embedding), dim=0)
    return embedding.unsqueeze(0)


class SparseMAEEncoder(nn.Module):
    """Unmasked encoder-only branch of the paper's 128-d SparseMAE."""

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        position = build_2d_sincos_position_embedding(
            embed_dim, image_size // patch_size, include_cls_token=True
        )
        self.register_buffer("pos_embed", position, persistent=True)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=lambda width: nn.LayerNorm(width, eps=1e-6),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.reshape(weight.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        tokens = tokens + self.pos_embed[:, 1:].to(dtype=tokens.dtype)
        cls_token = self.cls_token + self.pos_embed[:, :1].to(dtype=tokens.dtype)
        tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # DeepSWM's frozen Sparse-MAE feature-extraction script discards the
        # cls token and averages the encoded patch tokens.  Keep that pooling
        # contract even though this HMI-only adaptation removes masking and
        # trains the encoder jointly with the classifier.
        return tokens[:, 1:].mean(dim=1)


class S5Layer(nn.Module):
    """The residual S5+MLP layer used in the official DeepSWM code."""

    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s5 = S5(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.s5(self.norm1(tokens)))
        return tokens + self.dropout(self.mlp(self.norm2(tokens)))


class S5Block(nn.Module):
    def __init__(self, dim: int, depth: int, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [S5Layer(dim, dropout_rate=dropout_rate) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class DepthwiseChannelSelectiveModule(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                padding_mode="replicate",
            ),
            nn.InstanceNorm3d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.conv2d = nn.Sequential(
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=dim,
                padding_mode="replicate",
            ),
            nn.InstanceNorm3d(dim),
            nn.ReLU(),
        )
        self.image_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fused = self.conv3d(features) + self.conv2d(features)
        return features + self.refine(fused * self.image_attention(fused))


class SpatioTemporalSSM(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.s5_block = S5Block(dim, depth=1, dropout_rate=dropout_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels = features.shape[:2]
        spatial_shape = features.shape[2:]
        tokens = features.reshape(batch_size, channels, -1).transpose(1, 2)
        tokens = self.s5_block(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)


class SolarSpatialEncoder(nn.Module):
    """Three-level SSE parameterized by daily time length and one modality."""

    def __init__(
        self,
        time_length: int,
        dim: int = 64,
        sequence_length: int = 128,
        num_modalities: int = 1,
        levels: int = 3,
        sse_dropout: float = 0.6,
        dcsm_dropout: float = 0.6,
        stssm_dropout: float = 0.6,
    ) -> None:
        super().__init__()
        if levels != 3:
            raise ValueError("this implementation preserves the paper's three SSE levels")
        self.dim = dim
        self.sequence_length = sequence_length
        self.num_modalities = num_modalities
        self.stem = nn.Sequential(
            nn.Conv3d(
                time_length,
                dim,
                kernel_size=(3, 5, 5),
                stride=(1, 4, 4),
                padding=(1, 2, 2),
            ),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.Dropout(sse_dropout),
        )
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm3d(dim),
                    nn.Conv3d(
                        dim,
                        dim,
                        kernel_size=3,
                        stride=(1, 2, 2),
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Dropout(sse_dropout),
                )
                for _ in range(levels)
            ]
        )
        self.dcsm_modules = nn.ModuleList(
            [
                DepthwiseChannelSelectiveModule(dim, dcsm_dropout)
                for _ in range(levels)
            ]
        )
        self.stssm_modules = nn.ModuleList(
            [SpatioTemporalSSM(dim, stssm_dropout) for _ in range(levels)]
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(dim * num_modalities, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(sse_dropout),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Dropout(sse_dropout),
            nn.Conv2d(dim // 2, sequence_length, 3, padding=1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B, T, one HMI modality, H, W].  As in the official
        # implementation, T is Conv3d's channel axis and modality is depth.
        features = self.stem(sequence)
        for downsample, dcsm, stssm in zip(
            self.downsample_layers, self.dcsm_modules, self.stssm_modules
        ):
            features = stssm(dcsm(downsample(features)))

        if features.shape[2] != self.num_modalities:
            raise RuntimeError(
                "SSE modality depth changed unexpectedly: "
                f"expected {self.num_modalities}, got {features.shape[2]}"
            )
        features = features.reshape(
            features.shape[0], self.dim * self.num_modalities, *features.shape[-2:]
        )
        features = self.output_projection(features)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (8, 8))
        if self.dim != 64:
            raise RuntimeError(
                "DeepSWM's [L,D] reshape requires dim=64 with an 8x8 grid"
            )
        return features.flatten(2)


class LongRangeTemporalSSM(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 64,
        depth: int = 1,
        dropout_rate: float = 0.6,
    ) -> None:
        super().__init__()
        self.ssm_block = S5Block(feature_dim, depth, dropout_rate)
        self.temporal_convs = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_pool = nn.AdaptiveAvgPool1d(output_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        history = self.ssm_block(history)
        history = self.temporal_convs(history.transpose(1, 2))
        return self.output_pool(self.dropout(history))


class ClassificationHead(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        dim: int,
        num_classes: int = 4,
        dropout_rate: float = 0.7,
    ) -> None:
        super().__init__()
        hidden = max(dim // 16, 1)
        self.layers = nn.Sequential(
            nn.Linear(sequence_length * dim, hidden),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens.reshape(tokens.shape[0], -1))


class HMIOnlyDeepSWM(nn.Module):
    """Raw-logit HMI-only adaptation of DeepSWM."""

    def __init__(
        self,
        window_length: int = 1,
        image_size: int = 256,
        dim: int = 64,
        sequence_length: int = 128,
        sparse_embed_dim: int = 128,
        sparse_depth: int = 8,
        sparse_patch_size: int = 8,
        lt_depth: int = 1,
        mixing_depth: int = 1,
        sse_dropout: float = 0.6,
        dcsm_dropout: float = 0.6,
        stssm_dropout: float = 0.6,
        ltssm_dropout: float = 0.6,
        mixing_dropout: float = 0.6,
        head_dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if window_length < 1:
            raise ValueError("window_length must be at least one")
        if image_size != 256:
            raise ValueError("the audited DeepSWM geometry requires image_size=256")
        if dim != 64 or sequence_length != 128 or sparse_embed_dim != 128:
            raise ValueError(
                "the audited architecture uses dim=64, sequence_length=128, "
                "and sparse_embed_dim=128"
            )
        self.window_length = window_length
        self.image_size = image_size
        self.sse = SolarSpatialEncoder(
            time_length=window_length,
            dim=dim,
            sequence_length=sequence_length,
            num_modalities=1,
            levels=3,
            sse_dropout=sse_dropout,
            dcsm_dropout=dcsm_dropout,
            stssm_dropout=stssm_dropout,
        )
        self.history_encoder = SparseMAEEncoder(
            image_size=image_size,
            patch_size=sparse_patch_size,
            embed_dim=sparse_embed_dim,
            depth=sparse_depth,
            num_heads=8,
        )
        self.lt_ssm = LongRangeTemporalSSM(
            feature_dim=sparse_embed_dim,
            output_dim=dim,
            depth=lt_depth,
            dropout_rate=ltssm_dropout,
        )
        self.mixing_ssm = S5Block(
            dim=dim, depth=mixing_depth, dropout_rate=mixing_dropout
        )
        self.classification_head = ClassificationHead(
            sequence_length=2 * sequence_length,
            dim=dim,
            num_classes=4,
            dropout_rate=head_dropout,
        )

    @property
    def feature_extractors(self) -> tuple[nn.Module, ...]:
        return self.sse, self.history_encoder, self.lt_ssm, self.mixing_ssm

    def freeze_feature_extractor(self) -> None:
        for module in self.feature_extractors:
            module.requires_grad_(False)
            module.eval()
        self.classification_head.requires_grad_(True)

    def unfreeze_feature_extractor(self) -> None:
        for module in self.feature_extractors:
            module.requires_grad_(True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError(
                "DeepSWM expects [B,T,1,H,W], got " f"{tuple(sequence.shape)}"
            )
        batch_size, time_length, channels, _, _ = sequence.shape
        if time_length != self.window_length or channels != 1:
            raise ValueError(
                f"expected [B,{self.window_length},1,H,W], got "
                f"{tuple(sequence.shape)}"
            )
        sse_features = self.sse(sequence)
        history = self.history_encoder(sequence.flatten(0, 1))
        history = history.reshape(batch_size, time_length, -1)
        lt_features = self.lt_ssm(history)
        mixed = self.mixing_ssm(torch.cat((sse_features, lt_features), dim=1))
        # Return raw logits.  Softmax is intentionally applied only by losses
        # and probabilistic metrics, fixing the official double-softmax path.
        return self.classification_head(mixed)


__all__ = [
    "HMIOnlyDeepSWM",
    "SparseMAEEncoder",
    "SolarSpatialEncoder",
    "LongRangeTemporalSSM",
]
