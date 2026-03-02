import torch
import torch.nn as nn
import pytorch_lightning as pl

from auxiliary.ldm.modules.diffusionmodules.model import Encoder, Decoder
from auxiliary.clip.model import AttentionPool2d
from auxiliary.ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class VAE_CNN(pl.LightningModule):
    def __init__(self,
                 contrastive_dim,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 **ignore_kwargs
                 ):
        super().__init__()
        assert ddconfig['double_z'] == True, "VAE_CNN only supports double_z=True"
        self.image_size = ddconfig['resolution']
        self.downsample_factor = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.feature_size = self.image_size // self.downsample_factor
        self.feature_dim = ddconfig['z_channels']
        print(f"Initializing AutoencoderCNN with input_shape : Bx{ddconfig['in_channels']}x{self.image_size}x{self.image_size}, downsample_factor: {self.downsample_factor}, hidden_dim: {self.feature_dim}")
        print(f"The corresponding latent shape is Bx{self.feature_dim}x{self.feature_size}x{self.feature_size}")

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(2*self.feature_dim, 2*self.feature_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.feature_dim, self.feature_dim, 1)

        # self.cls_proj = AttentionPool2d(spacial_dim=self.feature_size, embed_dim=ddconfig['z_channels'], num_heads=8, output_dim=contrastive_dim)
        self.cls_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        scale = self.feature_dim ** -0.5
        self.contrasive_porject = nn.Parameter(scale * torch.randn(self.feature_dim, contrastive_dim))
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

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def contrastive_projection(self, z):
        # z shape: [B, feature_dim, feature_size, feature_size]
        cls = self.cls_proj(z)  # shape = [B, feature_dim]
        z = z.reshape(z.shape[0], self.feature_dim, -1).transpose(1, 2)  # shape = [B, feature_size*feature_size, feature_dim]
        z = torch.cat([cls.unsqueeze(1), z], dim=1)  # shape = [B, L+1, feature_dim]
        return z @ self.contrasive_porject
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    

class AE_CNN(VAE_CNN):
    def __init__(self,
                 contrastive_dim,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 **ignore_kwargs
                 ):
        pl.LightningModule.__init__(self)
        assert ddconfig['double_z'] == False, "AE_CNN only supports double_z=False"
        self.image_size = ddconfig['resolution']
        self.downsample_factor = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.feature_size = self.image_size // self.downsample_factor
        self.feature_dim = ddconfig['z_channels']
        print(f"Initializing AutoencoderCNN with input_shape : Bx{ddconfig['in_channels']}x{self.image_size}x{self.image_size}, downsample_factor: {self.downsample_factor}, hidden_dim: {self.feature_dim}")
        print(f"The corresponding latent shape is Bx{self.feature_dim}x{self.feature_size}x{self.feature_size}")

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # self.cls_proj = AttentionPool2d(spacial_dim=self.feature_size, embed_dim=ddconfig['z_channels'], num_heads=8, output_dim=contrastive_dim)
        self.cls_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        scale = self.feature_dim ** -0.5
        self.contrasive_porject = nn.Parameter(scale * torch.randn(self.feature_dim, contrastive_dim))
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        dec = self.decoder(z)
        return dec
    
    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        dec = self.decode(z)
        return dec, z
    

if __name__ == "__main__":
    dd_config = {
        "double_z": True,
        "z_channels": 32,
        "resolution": 1024,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 8,
        "ch_mult": [1,1,1,1,1,2,2], # spatial downsample 64, channel upsample 4
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "use_linear_attn": False,
        "dropout": 0.0,
    }

    ae_cnn = VAE_CNN(
        contrastive_dim=32,
        ddconfig=dd_config,
    ).to("cuda")
    x = torch.randn(22, 1, 1024, 1024).to("cuda")
    z = ae_cnn.encode(x)
    z = z.mode()
    print(z.shape)
    contra = ae_cnn.contrastive_projection(z)
    print(contra.shape)
    rec = ae_cnn.decode(z)
    print(rec.shape)

    dd_config = {
        "double_z": False,
        "z_channels": 32,
        "resolution": 1024,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 8,
        "ch_mult": [1,1,1,1,1,2,2], # spatial downsample 64, channel upsample 4
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "use_linear_attn": False,
        "dropout": 0.0,
    }

    ae_cnn = AE_CNN(
        contrastive_dim=32,
        ddconfig=dd_config,
    ).to("cuda")
    x = torch.randn(22, 1, 1024, 1024).to("cuda")
    z = ae_cnn.encode(x)
    print(z.shape)
    contra = ae_cnn.contrastive_projection(z)
    print(contra.shape)
    rec = ae_cnn.decode(z)
    print(rec.shape)  
