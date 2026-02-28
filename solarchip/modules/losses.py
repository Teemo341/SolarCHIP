import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import numpy as np

from auxiliary.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

class LPIPS(nn.Module):
    # Learned perceptual image patch similarity (LPIPS) loss
    # see https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/contperceptual.py
    def __init__(self, 
                 rec_loss_type='l2',
                 log_var_init=0.0,
                 kl_weight=0.1,
                 perceptual_weight=0):
        super().__init__()
        self.rec_loss_type = rec_loss_type
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

        # self.logvar = nn.Parameter(torch.ones(size=())*log_var_init)
        if perceptual_weight > 0:
            self.perceptual_loss = lpips.LPIPS(net='vgg').eval()

        
    def forward(self, inputs, recons, posteriors, weights=None):

        if self.rec_loss_type == 'l1':
            rec_loss = torch.abs(inputs.contiguous() - recons.contiguous())
        elif self.rec_loss_type == 'l2':
            rec_loss = F.mse_loss(inputs.contiguous(), recons.contiguous(), reduction='none')
        else:
            raise ValueError(f"Invalid rec_loss_type: {self.rec_loss_type}")
        
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(inputs.repeat(1, 3, 1, 1).contiguous(), recons.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * perceptual_loss

        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = rec_loss 
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        if isinstance(posteriors, None):
            kl_loss = torch.tensor(0.0)
        elif isinstance(posteriors, tuple):
            post_mu, post_logvar = posteriors
            kl_loss = -0.5 * torch.sum(1 + post_logvar - post_mu.pow(2) - post_logvar.exp())
            kl_loss = kl_loss / post_mu.shape[0]
        elif isinstance(posteriors, DiagonalGaussianDistribution):
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            raise ValueError(f"Invalid type for posteriors: {type(posteriors)}")
        
        loss = weighted_nll_loss + self.kl_weight * kl_loss

        loss_dict = {
            "total_loss": loss.item(),
            "rec_loss": rec_loss.mean().item(),
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return loss, loss_dict
    

class Contrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pat_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.int_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def cls_contrastive_loss(self, z_1, z_2):
        # z shape: [B, L, D]
        cls_1 = z_1[:, 0, :]  # shape = [B, D]
        cls_2 = z_2[:, 0, :]  # shape = [B, D]
        # normalized features
        feature_1 = cls_1 / cls_1.norm(dim=1, keepdim=True)
        feature_2 = cls_2 / cls_2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.cls_logit_scale.exp()
        logits = logit_scale * feature_1 @ feature_2.t()  # shape = [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.t(), labels)
        loss = (loss_1 + loss_2) / 2
        return loss
    
    def pat_contrastive_loss(self, z_1, z_2):    
        # z shape: [B, L, D]
        pat_1 = z_1[:, 1:, :]  # shape = [B, L-1, D]
        pat_2 = z_2[:, 1:, :]  # shape = [B, L-1, D]

        loss = 0
        labels = torch.arange(pat_1.shape[0], device=pat_1.device)
        logit_scale = self.pat_logit_scale.exp()
        for i in range(pat_1.shape[1]):
            feature_1 = pat_1[:, i, :] / pat_1[:, i, :].norm(dim=1, keepdim=True) # shape = [B, D]
            feature_2 = pat_2[:, i, :] / pat_2[:, i, :].norm(dim=1, keepdim=True)

            logits = logit_scale * feature_1 @ feature_2.t()  # shape = [B, B]
            loss_1 = F.cross_entropy(logits, labels)
            loss_2 = F.cross_entropy(logits.t(), labels)
            loss += (loss_1 + loss_2) / 2
        loss = loss / pat_1.shape[1]
        return loss
    
    def int_contrastive_loss(self, z_1, z_2):
        # z shape: [B, L, D]
        int_1 = z_1[:, 1:, :]  # shape = [B, L-1, D]
        int_2 = z_2[:, 1:, :]  # shape = [B, L-1, D]

        loss = 0
        labels = torch.arange(int_1.shape[1], device=int_1.device)
        for i in range(int_1.shape[0]):
            feature_1 = int_1[i, :, :] / int_1[i, :, :].norm(dim=1, keepdim=True) # shape = [L-1, D]
            feature_2 = int_2[i, :, :] / int_2[i, :, :].norm(dim=1, keepdim=True)

            logits = self.int_logit_scale.exp() * feature_1 @ feature_2.t()  # shape = [L-1, L-1]
            loss_1 = F.cross_entropy(logits, labels)
            loss_2 = F.cross_entropy(logits.t(), labels)
            loss += (loss_1 + loss_2) / 2
        loss = loss / int_1.shape[0]
        return loss