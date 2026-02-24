from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


def load_torchvision_vit_b16(
	weights: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1,
) -> nn.Module:
	"""Load torchvision ViT-B/16 with optional pretrained weights."""
	return vit_b_16(weights=weights)


def _init_by_interpolate(
	dst_weight: torch.Tensor,
	src_weight: torch.Tensor,
	new_kh: int,
	new_kw: int,
) -> None:
	"""Interpolate src_weight to match dst_weight spatial size (in-place)."""
	interpolated = F.interpolate(
		src_weight,
		size=(new_kh, new_kw),
		mode="bicubic",
		align_corners=False,
	)
	dst_weight.copy_(interpolated)

def replace_patchify(
	model: nn.Module,
	in_channels: int,
	patch_h: int,
	patch_w: int,
) -> nn.Module:
	"""
	Replace torchvision ViT patchify layer (conv_proj) with custom CxHxW patchify.

	Args:
		model: torchvision ViT model.
		in_channels: input channel number C.
		patch_h: patch height H.
		patch_w: patch width W.
	"""
	old_conv = model.conv_proj
	if not isinstance(old_conv, nn.Conv2d):
		raise TypeError("model.conv_proj is expected to be nn.Conv2d")

	new_conv = nn.Conv2d(
		in_channels=in_channels,
		out_channels=old_conv.out_channels,
		kernel_size=(patch_h, patch_w),
		stride=(patch_h, patch_w),
		padding=0,
		bias=old_conv.bias is not None,
	)

	with torch.no_grad():
		old_weight = old_conv.weight
		old_c = old_weight.shape[1]
		old_kh, old_kw = old_weight.shape[-2:]

		if in_channels == old_c:
			if (old_kh, old_kw) == (patch_h, patch_w):
				new_conv.weight.copy_(old_weight)
			else:
				_init_by_interpolate(new_conv.weight, old_weight, patch_h, patch_w)
		elif in_channels > old_c:
			if (old_kh, old_kw) == (patch_h, patch_w):
				new_conv.weight[:, :old_c] = old_weight
			else:
				_init_by_interpolate(new_conv.weight[:, :old_c], old_weight, patch_h, patch_w)
			extra = in_channels - old_c
			mean_channel = old_weight.mean(dim=1, keepdim=True)
			_init_by_interpolate(new_conv.weight[:, old_c:], mean_channel.expand(-1, extra, -1, -1), patch_h, patch_w)
		else:
			if (old_kh, old_kw) == (patch_h, patch_w):
				new_conv.weight.copy_(old_weight[:, :in_channels])
			else:
				_init_by_interpolate(new_conv.weight, old_weight[:, :in_channels], patch_h, patch_w)

		if old_conv.bias is not None:
			new_conv.bias.copy_(old_conv.bias)

	model.conv_proj = new_conv
	return model


def interpolate_pos_embedding(
	model: nn.Module,
	num_patches_w: int,
	num_patches_h: int,
) -> torch.Tensor:
	"""
	Interpolate ViT positional embedding to a new patch grid size.

	Args:
		model: torchvision ViT model.
		num_patches_w: number of patches along width.
		num_patches_h: number of patches along height.

	Returns:
		Interpolated positional embedding tensor with shape
		[1, 1 + num_patches_h * num_patches_w, hidden_dim].
	"""
	pos_embed = model.encoder.pos_embedding
	if pos_embed.ndim != 3:
		raise ValueError("Expected model.encoder.pos_embedding to be [1, N, D]")

	cls_token = pos_embed[:, :1, :]
	patch_pos = pos_embed[:, 1:, :]

	old_num_patches = patch_pos.shape[1]
	old_grid_h = int(old_num_patches**0.5)
	old_grid_w = old_grid_h
	if old_grid_h * old_grid_w != old_num_patches:
		raise ValueError("Original patch positional embedding is not square")

	hidden_dim = patch_pos.shape[-1]
	patch_pos = patch_pos.reshape(1, old_grid_h, old_grid_w, hidden_dim)
	patch_pos = patch_pos.permute(0, 3, 1, 2)
	patch_pos = F.interpolate(
		patch_pos,
		size=(num_patches_h, num_patches_w),
		mode="bicubic",
		align_corners=False,
	)
	patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
		1,
		num_patches_h * num_patches_w,
		hidden_dim,
	)

	return torch.cat([cls_token, patch_pos], dim=1)


def build_custom_vit_b64(
	in_channels: int=1,
	patch_h: int=64,
	patch_w: int=64,
	num_patches_w: int=16,
	num_patches_h: int=16,
	weights: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1,
) -> nn.Module:
	"""
	Build a pretrained torchvision ViT-B/16 adapted for custom patchify,
	with position embeddings already interpolated and replaced in-place.
	
	Args:
		in_channels: Input channel count (default 1 for single modality).
		patch_h, patch_w: Patch size in pixels.
		num_patches_h, num_patches_w: Target grid size in patches.
		weights: Pretrained weights enum.
	
	Returns:
		Model with replaced patchify and position embeddings.
	"""
	model = load_torchvision_vit_b16(weights=weights)
	model = replace_patchify(
		model=model,
		in_channels=in_channels,
		patch_h=patch_h,
		patch_w=patch_w,
	)
	new_pos_embed = interpolate_pos_embedding(
		model,
		num_patches_w=num_patches_w,
		num_patches_h=num_patches_h,
	)
	model.encoder.pos_embedding = nn.Parameter(new_pos_embed)
	return model