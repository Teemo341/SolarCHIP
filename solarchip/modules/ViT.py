from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from auxiliary.clip.model import VisionTransformer as clipvit
from auxiliary.clip.model import LayerNorm, Transformer


class Encoder(clipvit):
    """
    Modified ViT encoder that outputs token features without the classification head.
    """

    def __init__(self, input_dim: int, input_resolution: int, patch_size: int, hidden_dim: int, layers: int, heads: int):
        super().__init__(input_resolution, patch_size, hidden_dim, layers, heads, 1)
        self.proj = None  # remove the projection head
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, hidden_dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, hidden_dim, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, hidden_dim]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, hidden_dim]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LBD -> BLD

        # x = self.ln_post(x[:, 0, :]) # originally only return the cls token, but we want to return all tokens
        x = self.ln_post(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_dim:int, input_resolution: int, patch_size: int, hidden_dim: int, layers: int, heads: int):
        super().__init__()
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.input_dim = input_dim
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.de_proj = nn.Linear(hidden_dim, input_dim * patch_size * patch_size)

        self.ln_pre = LayerNorm(hidden_dim)
        self.transformer = Transformer(hidden_dim, layers, heads)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, "Input to decoder should be of shape [B, L, D]"
        assert x.shape[1] == (self.input_resolution // self.patch_size) ** 2 + 1, f"Input to decoder should have {self.input_resolution // self.patch_size ** 2 + 1} tokens, but got {x.shape[1]}"

        x = x[:, 1:, :]  # remove the cls token
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LBD -> BLD

        x = self.de_proj(x)  # shape = [B, L, input_dim * patch_size * patch_size]
        x = x.reshape(x.shape[0], self.input_dim, self.input_resolution, self.input_resolution)  # shape = [B, input_dim, input_resolution, input_resolution]

        return x


class AE_ViT(pl.LightningModule):
    def __init__(self,
                 contrastive_dim,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 **ignore_kwargs
                 ):
        super().__init__()
        print(f"Initializing AutoencoderViT with input_shape : Bx{ddconfig['input_dim']}x{ddconfig['input_resolution']}x{ddconfig['input_resolution']}, patch_size: {ddconfig['patch_size']}, hidden_dim: {ddconfig['hidden_dim']}")
        print(f"The corresponding latent shape is Bx{(ddconfig['input_resolution'] // ddconfig['patch_size']) ** 2 + 1}x{ddconfig['hidden_dim']} (with cls token)")
        hidden_dim = ddconfig['hidden_dim']
        scale = hidden_dim ** -0.5
        self.contrasive_porject = nn.Parameter(scale * torch.randn(hidden_dim, contrastive_dim))
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
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
        enc = self.encoder(x)
        # enc: [B, N, D] where N can be patch count or patch count + 1 (with cls token)
        return enc

    def decode(self, z):
        dec = self.decoder(z)
        return dec
    
    def contrastive_projection(self, z):
        # z: [B, N, D]
        # return: [B, N, contrastive_dim]
        return z @ self.contrasive_porject

    def forward(self, input):
        z = self.encode(input)
        dec = self.decode(z)
        return dec, z


if __name__ == "__main__":

    ae_vit = AE_ViT(
        contrastive_dim=512,
        ddconfig={
            "input_dim": 1,
            "input_resolution": 1024,
            "patch_size": 64,
            "hidden_dim": 768,
            "layers": 6,
            "heads": 8,
        },
    ).to("cuda")
    x = torch.randn(22, 1, 1024, 1024).to("cuda")
    z = ae_vit.encode(x)
    print(z.shape)
    contra = ae_vit.contrastive_projection(z)
    print(contra.shape)
    rec = ae_vit.decode(z)
    print(rec.shape)
        