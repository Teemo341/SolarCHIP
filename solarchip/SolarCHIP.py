import math
import random
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np

from .utils.util import instantiate_from_config
from .modules.losses import LPIPS, Contrastive


class solarchip_base(pl.LightningModule):
    """
    A wrapper class for multiple models to be used in a single training loop.
    """
    def __init__(self, modal_list, base_model, loss_config=None, save_memory=False, ckpt_path=None, ignore_keys=list()):
        super().__init__()
        self.id_to_modal = modal_list
        self.modal_to_id = {modal: i for i, modal in enumerate(modal_list)}
        self.save_memory = save_memory # whether to save memory by not optimizing all models at the same time

        self.init_models(base_model)
        self.init_loss(loss_config)
        self.automatic_optimization = False # use manual optimization to control the optimization of each model for memory saving

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # funcitons for initialization
    def init_models(self, base_model_config):
        """
        Instantiate the models specified in the config.
        """
        assert self.id_to_modal[0] == 'hmi', "The first modal must be hmi for the current implementation."
        self.model_dict = nn.ModuleDict()
        for modal in self.id_to_modal:
            model = instantiate_from_config(base_model_config)
            self.model_dict[modal] = model

    def get_model(self, modal):
        return self.model_dict[modal]

    def init_loss(self, loss_config):
        self.loss_config = loss_config
        if loss_config is not None:
            self.rec_loss_fn = LPIPS(**loss_config)
            self.contrastive_loss_fn = Contrastive()
            self.cls_ctr_weight = loss_config['cls_contrastive_weight']
            self.pat_ctr_weight = loss_config['pat_contrastive_weight']
            self.int_ctr_weight = loss_config['int_contrastive_weight']
        

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.loss_config is None: # if no loss config is provided,delete possible loss parameters from the state dict to avoid loading error
            if 'rec_loss_fn' in sd:
                del sd['rec_loss_fn']
            if 'contrastive_loss_fn' in sd:
                del sd['contrastive_loss_fn']
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def configure_optimizers(self):
        assert self.model_dict is not None, "Models must be initialized before configuring optimizer."
        if self.loss_config is None:
            return # no optimizer if no loss config is provided since there is no training objective
        
        param_list = []
        for _, model in self.model_dict.items():
            param_list.append(model.parameters())
        if self.loss_config is not None:
            param_list.append(self.rec_loss_fn.parameters())
            param_list.append(self.contrastive_loss_fn.parameters())
        if self.loss_config["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(param_list, lr = self.loss_config["lr"])
        elif self.loss_config[optimizer] == 'AdamW':
            optimizer = torch.optim.AdamW(param_list, lr = self.loss_config["lr"])
        else:
            raise ValueError(f'Optimizer {self.loss_config["optimizer"]} is not supported')

        if self.loss_config["scheduler"] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.loss_config["epochs"])
        elif self.loss_config["scheduler"] == None:
            scheduler = None
        else:
            raise ValueError(f'Scheduler {self.loss_config["scheduler"]} is not supported')
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def forward_save_memory(self, batch, optimize=True):
        loss_dict = {}

        # first optimize hmi
        hmi = batch['hmi'].to(self.device)
        rec_hmi, z_hmi = self.model_dict['hmi'](hmi)
        rec_loss, loss_dict_tmp = self.rec_loss_fn(hmi, rec_hmi, posteriors=z_hmi)
        for k, v in loss_dict_tmp.items():
            loss_dict[f"hmi/{k}"] = v
        cls_ctr_loss, pat_ctr_loss, int_ctr_loss = 0, 0, 0
        for i in range(1, len(self.id_to_modal)):
            modal = self.id_to_modal[i]
            aia = batch[modal].to(self.device)
            with torch.no_grad():
                rec_aia, z_aia = self.get_model(modal)(aia)
            if self.cls_ctr_weight >0:
                cls_ctr_loss_tmp = self.contrastive_loss_fn.cls_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"cls_ctr_loss/hmi_{modal}"] = cls_ctr_loss_tmp.item()
                cls_ctr_loss += cls_ctr_loss_tmp
            if self.pat_ctr_weight > 0:
                pat_ctr_loss_tmp = self.contrastive_loss_fn.pat_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"pat_ctr_loss/hmi_{modal}"] = pat_ctr_loss_tmp.item()
                pat_ctr_loss += pat_ctr_loss_tmp
            if self.int_ctr_weight > 0:
                    int_ctr_loss_tmp = self.contrastive_loss_fn.int_contrastive_loss(z_hmi, z_aia)
                    loss_dict[f"int_ctr_loss/hmi_{modal}"] = int_ctr_loss_tmp.item()
                    int_ctr_loss += int_ctr_loss_tmp
        cls_ctr_loss = cls_ctr_loss / (len(self.id_to_modal)-1)
        pat_ctr_loss = pat_ctr_loss / (len(self.id_to_modal)-1)
        int_ctr_loss = int_ctr_loss / (len(self.id_to_modal)-1)
        total_loss = rec_loss + self.cls_ctr_weight * cls_ctr_loss + self.pat_ctr_weight * pat_ctr_loss + self.int_ctr_weight * int_ctr_loss
        loss_dict['hmi/total_loss'] = total_loss.item()
        if optimize:
            optimizer = self.optimizers()
            self.manual_backward(total_loss)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True) # remove the computational graph for hmi to save memory
        # remove the computational graph for hmi to save memory
        del rec_hmi, rec_loss, cls_ctr_loss, pat_ctr_loss, int_ctr_loss, total_loss
        z_hmi = z_hmi.detach() # detach z_hmi to save memory
        z_hmi.requires_grad = False

        # then optimize other modals one by one
        for i in range(1, len(self.id_to_modal)):
            modal = self.id_to_modal[i]
            aia = batch[modal].to(self.device)
            rec_aia, z_aia = self.get_model(modal)(aia)
            rec_loss, loss_dict_tmp = self.rec_loss_fn(aia, rec_aia, posteriors=z_aia)
            for k, v in loss_dict_tmp.items():
                loss_dict[f"{modal}/{k}"] = v
            cls_ctr_loss, pat_ctr_loss, int_ctr_loss = 0, 0, 0
            if self.cls_ctr_weight > 0:
                cls_ctr_loss = self.contrastive_loss_fn.cls_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"cls_ctr_loss/hmi_{modal}"] = (cls_ctr_loss.item()+loss_dict[f"cls_ctr_loss/hmi_{modal}"]) / 2 # average contrastive loss between hmi and the modal
            if self.pat_ctr_weight > 0:
                pat_ctr_loss = self.contrastive_loss_fn.pat_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"pat_ctr_loss/hmi_{modal}"] = (pat_ctr_loss.item()+loss_dict[f"pat_ctr_loss/hmi_{modal}"]) / 2
            if self.int_ctr_weight > 0:
                int_ctr_loss = self.contrastive_loss_fn.int_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"int_ctr_loss/hmi_{modal}"] = (int_ctr_loss.item()+loss_dict[f"int_ctr_loss/hmi_{modal}"]) / 2
            total_loss = rec_loss + self.cls_ctr_weight * cls_ctr_loss + self.pat_ctr_weight * pat_ctr_loss + self.int_ctr_weight * int_ctr_loss
            loss_dict[f'{modal}/total_loss'] = total_loss.item()
            if optimize:
                optimizer = self.optimizers()
                self.manual_backward(total_loss)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True) # remove the computational graph for the modal to save memory
            # remove the computational graph for the modal to save memory
            del rec_aia, rec_loss, cls_ctr_loss, pat_ctr_loss, int_ctr_loss, total_loss
            del z_aia

        return loss_dict
    
    def forward_full_memory(self, batch, optimize=True):
        # optimize all models together
        loss_dict = {}
        total_loss = 0
        hmi = batch['hmi'].to(self.device)
        rec_hmi, z_hmi = self.model_dict['hmi'](hmi)
        rec_loss_tmp, loss_dict_tmp = self.rec_loss_fn(hmi, rec_hmi, posteriors=z_hmi)
        for k, v in loss_dict_tmp.items():
            loss_dict[f"hmi/{k}"] = v
        loss_dict['hmi/total_loss'] = rec_loss_tmp.item()
        total_loss += rec_loss_tmp

        # calculate contrastive loss between hmi and other modals
        for i in range(1, len(self.id_to_modal)):
            modal = self.id_to_modal[i]
            aia = batch[modal].to(self.device)
            rec_aia, z_aia = self.get_model(modal)(aia)
            rec_loss_tmp, loss_dict_tmp = self.rec_loss_fn(aia, rec_aia, posteriors=z_aia)
            total_loss += rec_loss_tmp
            for k, v in loss_dict_tmp.items():
                loss_dict[f"{modal}/{k}"] = v
            loss_dict[f'{modal}/total_loss'] = rec_loss_tmp.item()
            if self.cls_ctr_weight>0:
                cls_ctr_loss_tmp = self.contrastive_loss_fn.cls_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"cls_ctr_loss/hmi_{modal}"] = cls_ctr_loss_tmp.item()
                cls_ctr_loss_tmp = cls_ctr_loss_tmp/ (len(self.id_to_modal)-1) * self.cls_ctr_weight
                loss_dict['hmi/total_loss'] += cls_ctr_loss_tmp.item()
                loss_dict[f'{modal}/total_loss'] += cls_ctr_loss_tmp.item()
                total_loss += cls_ctr_loss_tmp
            if self.pat_ctr_weight > 0:
                pat_ctr_loss_tmp = self.contrastive_loss_fn.pat_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"pat_ctr_loss/hmi_{modal}"] = pat_ctr_loss_tmp.item()
                pat_ctr_loss_tmp = pat_ctr_loss_tmp/ (len(self.id_to_modal)-1) * self.pat_ctr_weight
                loss_dict['hmi/total_loss'] += pat_ctr_loss_tmp.item()
                loss_dict[f'{modal}/total_loss'] += pat_ctr_loss_tmp.item()
                total_loss += pat_ctr_loss_tmp
            if self.int_ctr_weight > 0:
                int_ctr_loss_tmp = self.contrastive_loss_fn.int_contrastive_loss(z_hmi, z_aia)
                loss_dict[f"int_ctr_loss/hmi_{modal}"] = int_ctr_loss_tmp.item()
                int_ctr_loss_tmp = int_ctr_loss_tmp/ (len(self.id_to_modal)-1) * self.int_ctr_weight
                loss_dict['hmi/total_loss'] += int_ctr_loss_tmp.item()
                loss_dict[f'{modal}/total_loss'] += int_ctr_loss_tmp.item()
                total_loss += int_ctr_loss_tmp
        if optimize:
            optimizer = self.optimizers()
            self.manual_backward(total_loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # remove the computational graph to save memory

        return loss_dict

    # functions for training
    def training_step(self, batch, batch_idx):
        if self.save_memory:
            loss_dict = self.forward_save_memory(batch, optimize=True)
        else:
            loss_dict = self.forward_full_memory(batch, optimize=True)

        # log losses
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, logger=True, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            schedulers.step()

    def validation_step(self, batch, batch_idx):
        if self.save_memory:
            loss_dict = self.forward_save_memory(batch, optimize=False)
        else:
            loss_dict = self.forward_full_memory(batch, optimize=False)

        # log losses
        for k, v in loss_dict.items():
            self.log(f'val/{k}', v, logger=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        if self.save_memory:
            loss_dict = self.forward_save_memory(batch, optimize=False)
        else:
            loss_dict = self.forward_full_memory(batch, optimize=False)

        # log losses
        for k, v in loss_dict.items():
            self.log(f'test/{k}', v, logger=True, on_epoch=True, sync_dist=True)
         

def solar_painting(image_array, modal, title = None):
    assert len(image_array.shape) == 4 # (b, c, h, w)
    if modal == "hmi":
        cmap = "RdBu_r"
        vmax = np.max(np.abs(image_array))
        vmin = -vmax
    else:
        cmap = "Reds"
        vmin = 0
        vmax = np.max(image_array)
    num_images = min(image_array.shape[0], 4)
    fig = plt.figure(figsize=(num_images*16, 16))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(image_array[i, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def latent_painting(image_array, modal, title = None):
    assert len(image_array.shape) == 4 # (b, c, h, w)
    num_images = min(image_array.shape[0], 4)
    fig = plt.figure(figsize=(num_images*16, 16))
    c = image_array.shape[1]
    visual_channels = np.random.choice(c, 3, replace=(c < 3))
    image_array = image_array[:, visual_channels, :, :]
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(image_array[i,: :, :].transpose(1, 2, 0))
        plt.title(title)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig