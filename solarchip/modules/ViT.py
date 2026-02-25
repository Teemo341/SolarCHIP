from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16
import pytorch_lightning as pl


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


def build_custom_vit_b(
	in_channels: int=1,
	patch_h: int=64,
	patch_w: int=64,
	num_patches_w: int=16,
	num_patches_h: int=16,
	weights: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1,
	cache_dir: Optional[str] = './checkpoints/opensource',
	**ignore_kwargs
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
	if cache_dir is not None:
		torch.hub.set_dir(cache_dir)
	if patch_h != patch_w:
		raise ValueError("torchvision ViT requires square patch size; expected patch_h == patch_w")
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
	model.patch_size = patch_h
	model.image_size = patch_h * num_patches_h
	return model

class vit_decoder(nn.Module):
	"""
	Symmetric decoder for ViT token features.

	Input token shape: [B, N, D] where N can be patch count or patch count + 1 (with cls token).
	Output image shape: [B, out_channels, num_patches_h * patch_h, num_patches_w * patch_w]
	"""

	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		mlp_dim: int,
		num_layers: int,
		num_patches_h: int,
		num_patches_w: int,
		patch_h: int,
		patch_w: int,
		out_channels: int = 1,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_patches_h = num_patches_h
		self.num_patches_w = num_patches_w
		self.patch_h = patch_h
		self.patch_w = patch_w
		self.out_channels = out_channels
		self.seq_len = num_patches_h * num_patches_w

		self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, hidden_dim))
		nn.init.trunc_normal_(self.pos_embedding, std=0.02)

		self.blocks = nn.ModuleList(
			[
				nn.TransformerEncoderLayer(
					d_model=hidden_dim,
					nhead=num_heads,
					dim_feedforward=mlp_dim,
					dropout=dropout,
					activation="gelu",
					batch_first=True,
					norm_first=True,
				)
				for _ in range(num_layers)
			]
		)
		self.norm = nn.LayerNorm(hidden_dim)
		self.to_patch_pixels = nn.Linear(hidden_dim, out_channels * patch_h * patch_w)

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		if tokens.ndim != 3:
			raise ValueError(f"Expected tokens as [B, N, D], got shape {tokens.shape}")

		if tokens.shape[-1] != self.hidden_dim:
			raise ValueError(
				f"Hidden dim mismatch: decoder expects {self.hidden_dim}, got {tokens.shape[-1]}"
			)

		if tokens.shape[1] == self.seq_len + 1:
			tokens = tokens[:, 1:, :]
		elif tokens.shape[1] != self.seq_len:
			raise ValueError(
				f"Token length mismatch: expected {self.seq_len} or {self.seq_len + 1}, got {tokens.shape[1]}"
			)

		x = tokens + self.pos_embedding
		for block in self.blocks:
			x = block(x)
		x = self.norm(x)

		x = self.to_patch_pixels(x)
		batch_size = x.shape[0]
		x = x.view(
			batch_size,
			self.num_patches_h,
			self.num_patches_w,
			self.out_channels,
			self.patch_h,
			self.patch_w,
		)
		x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
		x = x.view(
			batch_size,
			self.out_channels,
			self.num_patches_h * self.patch_h,
			self.num_patches_w * self.patch_w,
		)
		return x


def build_symmetric_vit_decoder(
	encoder_model: nn.Module,
	num_patches_h: int,
	num_patches_w: int,
	patch_h: int,
	patch_w: int,
	out_channels: int = 1,
	**ignore_kwargs
) -> vit_decoder:
	"""Build a decoder symmetric to a torchvision ViT encoder."""
	if not hasattr(encoder_model, "encoder") or not hasattr(encoder_model.encoder, "layers"):
		raise TypeError("encoder_model must be a torchvision ViT model with encoder.layers")

	if len(encoder_model.encoder.layers) == 0:
		raise ValueError("encoder_model.encoder.layers is empty")

	first_layer = encoder_model.encoder.layers[0]
	hidden_dim = getattr(encoder_model, "hidden_dim", first_layer.ln_1.normalized_shape[0])
	num_heads = first_layer.self_attention.num_heads
	mlp_dim = first_layer.mlp[0].out_features
	dropout = first_layer.dropout.p
	num_layers = len(encoder_model.encoder.layers)

	decoder = vit_decoder(
		hidden_dim=hidden_dim,
		num_heads=num_heads,
		mlp_dim=mlp_dim,
		num_layers=num_layers,
		num_patches_h=num_patches_h,
		num_patches_w=num_patches_w,
		patch_h=patch_h,
		patch_w=patch_w,
		out_channels=out_channels,
		dropout=dropout,
	)

	with torch.no_grad():
		encoder_pos = interpolate_pos_embedding(
			encoder_model,
			num_patches_w=num_patches_w,
			num_patches_h=num_patches_h,
		)
		decoder.pos_embedding.copy_(encoder_pos[:, 1:, :])

	return decoder


class AutoencoderViT(pl.LightningModule):
	def __init__(self,
				 ddconfig,
				 ckpt_path=None,
				 ignore_keys=[],
				 **ignore_kwargs
				 ):
		super().__init__()
		self.encoder = build_custom_vit_b(**ddconfig)
		self.decoder = build_symmetric_vit_decoder(encoder_model=self.encoder, **ddconfig)
		if ckpt_path is not None:
			self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

	def init_from_ckpt(self, path, ignore_keys=list()):
		sd = torch.load(path, map_location="cpu")["state_dict"]
		keys = list(sd.keys())
		for k in keys:
			for ik in ignore_keys:
				if k.startswith(ik):
					print("Deleting key {} from state_dict.".format(k))
					del sd[k]
		self.load_state_dict(sd, strict=False)
		print(f"Restored from {path}")

	def encode(self, x): # return
		# x: [B, C, H, W]
		tokens = self.encoder._process_input(x)
		batch_size = tokens.shape[0]
		cls_tokens = self.encoder.class_token.expand(batch_size, -1, -1)
		tokens = torch.cat([cls_tokens, tokens], dim=1)
		enc = self.encoder.encoder(tokens)
		# enc: [B, N, D] where N can be patch count or patch count + 1 (with cls token)
		return enc

	def decode(self, z):
		dec = self.decoder(z)
		return dec

	def forward(self, input):
		z = self.encode(input)
		dec = self.decode(z)
		return dec


class AutoencoderViT_B_64(AutoencoderViT):
	def __init__(self, ckpt_path=None, ignore_keys=[]):
		ddconfig = dict(
			in_channels=1,
			patch_h=64,
			patch_w=64,
			num_patches_h=16,
			num_patches_w=16,
			out_channels=1,
			weights=ViT_B_16_Weights.IMAGENET1K_V1,
			cache_dir='./checkpoints/opensource',
		)
		super().__init__(ddconfig, ckpt_path=ckpt_path, ignore_keys=ignore_keys)

if __name__ == "__main__":
	# vit_model = build_custom_vit_b().to("cuda")
	# random_input = torch.randn(1,1,1024,1024).to("cuda")
	# logits = vit_model(random_input)
	# print(logits.shape)

	# with torch.no_grad():
	# 	tokens = vit_model._process_input(random_input)
	# 	batch_size = tokens.shape[0]
	# 	cls_tokens = vit_model.class_token.expand(batch_size, -1, -1)
	# 	tokens = torch.cat([cls_tokens, tokens], dim=1)
	# 	output = vit_model.encoder(tokens)
	# print(output.shape)
	# vit_decoder_model = build_symmetric_vit_decoder(vit_model, num_patches_h=16, num_patches_w=16, patch_h=64, patch_w=64).to("cuda")
	# decoded_image = vit_decoder_model(output)
	# print(decoded_image.shape)

	ae_vit = AutoencoderViT_B_64().to("cuda")
	random_input = torch.randn(22,1,1024,1024).to("cuda")
	decoded = ae_vit(random_input)
	print(decoded.shape)