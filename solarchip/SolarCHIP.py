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


class solarchip_base(pl.LightningModule):
    """
    A wrapper class for multiple models to be used in a single training loop.
    """
    def __init__(self, modal_list, save_memory=False, ckpt_path=None, ignore_keys=list()):
        super().__init__()
        self.id_to_modal = modal_list
        self.save_memory = save_memory # whether to save memory by not optimizing all models at the same time

        assert self.data_id_to_modal == self.config.data.params.validation.params.modal_list # train and val modal list should be the same
        assert len(self.data_id_to_modal) == len(self.models), "train and val modal list should be the same as the model list"
        self.data_modal_to_id = { modal: i for i, modal in enumerate(self.data_id_to_modal) }

        self.automatic_optimization = False # to use manual optimization
        self.get_models()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # funcitons for initialization
    def get_models(self):
        """
        Instantiate the models specified in the config.
        """
        self.models = nn.ModuleDict()
        self.modal_to_id = {}
        self.id_to_modal = {}
        for i, (modal_name, model_config) in enumerate(self.config.model.items()):
            self.modal_to_id[modal_name] = i
            self.id_to_modal[i] = modal_name
            model = instantiate_from_config(model_config)
            self.models[modal_name] = model
            print(f"Model {modal_name} loaded from {model_config.params.ckpt_path}")

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

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        for i in range(len(self.models)):
            modal_name = self.id_to_modal[i]
            if getattr(self.config.model, modal_name).base_learning_optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.models[modal_name].parameters(), lr = getattr(self.config.model, modal_name).base_learning_rate)
            elif getattr(self.config.model, modal_name).base_learning_optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(self.models[modal_name].parameters(), lr = getattr(self.config.model, modal_name).base_learning_rate)
            else:
                raise ValueError(f'Optimizer {getattr(self.config.model, modal_name).base_learning_optimizer} is not supported')
            optimizers.append(optimizer)

            if getattr(self.config.model, modal_name).base_learning_schedule == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs)
            else:
                raise ValueError(f'Scheduler {getattr(self.config.model, modal_name).base_learning_schedule} is not supported')
            schedulers.append(scheduler)
        return optimizers, schedulers
    
    # functions for training
    def training_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max
        label = torch.arange(batch.shape[0]).to(batch.device)  # (b,) label for contrastive loss

        if self.full_model_train:
            rec_loss_ = {}
            kld_loss_ = {}
            logits_ = {}
            contrast_loss_ = {}
            for name, model in self.models.items():
                # get rec_loss, kld_loss and logits for each model
                model.train()
                rec_loss, kld_loss, mu, _, _ = model.calculate_loss(batch[:, self.data_modal_to_id[name], :, :, :], return_moment=True)
                rec_loss_[name] = rec_loss
                kld_loss_[name] = kld_loss
                logit = model.get_logit(mu)
                # logit = model.class_block(mu)  # (b, c)
                logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                logits_[name] = logit

            if self.mean_logit:
                mean_logit = torch.mean(torch.stack(list(logits_.values())), dim=0)  # (b, c)
                mean_logit = mean_logit/(mean_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            
            # calculate contrast loss
            for name, model in self.models.items():
                contrast_loss = 0
                if self.mean_logit:
                    cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                    contrast_loss = F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss
                else:
                    for name2, model2 in self.models.items():
                        if name != name2:
                            cor_matrix = torch.matmul(logits_[name], logits_[name2].T)
                            contrast_loss += F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss/ (len(self.models)-1) # average contrast loss for each model

            # optimize
            rec_loss = sum(rec_loss_.values())
            kld_loss = sum(kld_loss_.values())
            contrast_loss = sum(contrast_loss_.values())
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss
            optimizers = self.optimizers()
            for optimizer in optimizers:
                optimizer.zero_grad()
            self.manual_backward(loss)
            # check gradients
            # if self.global_rank == 0:
            #     for name, model in self.models.items():
            #         print(f"Model {name} parameters:")
            #         for param_name, param in model.class_block.named_parameters():
            #             print(f"Parameter name: {param_name}")
            #             if param.grad is not None:
            #                 print("Gradient:", param.grad)
            #             else:
            #                 print("Gradient: None")
            #         break     
            for optimizer in optimizers:
                optimizer.step()

            # log
            for name, model in self.models.items():
                self.log(f"{name}/train/loss", loss, logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/rec_loss/", rec_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/kld_loss", kld_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/contrast_loss", contrast_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"contrast_weight", contrast_weight, logger=True, on_epoch=True, sync_dist=True)
                self.log(f'{name}/scheduler', self.lr_schedulers()[self.modal_to_id[name]].get_last_lr()[0], logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/loss", loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/rec_loss", rec_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/kld_loss", kld_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/contrast_loss", contrast_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)

        else:
            training_id = random.randint(0, len(self.models) - 1)  # randomly select a model to train
            training_modal = self.id_to_modal[training_id]  # get the training modal name
            self.models[training_modal].train()  # set the selected model to train mode

            # loss for the selected model
            data_id = self.data_modal_to_id[training_modal]  # get the data id for the selected model
            rec_loss, kld_loss, mu, _, _ = self.models[training_modal].calculate_loss(batch[:, data_id, :, :, :], return_moment=True)

            # contrastive loss
            contrast_loss = 0
            logit = self.models[training_modal].get_logit(mu)  # (b, c)
            logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            other_logits = {}
            for i in range(len(self.models)):
                if i != training_id:
                    compare_modal = self.id_to_modal[i]  # get the compare modal name
                    self.models[compare_modal].eval()  # set the other models to eval mode
                    data_id = self.data_modal_to_id[compare_modal]  # get the data id for the other model
                    with torch.no_grad():
                        other_logit = self.models[compare_modal].get_logit(self.models[compare_modal].encode(batch[:, data_id, :, :, :])[0])  # (b, c)
                        other_logit = other_logit/(other_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                        other_logits[compare_modal] = other_logit
            if self.mean_logit:
                other_logits[training_modal] = logit
                mean_logit = torch.mean(torch.stack(list(other_logits.values())), dim=0)  # (b, c)
                cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                contrast_loss = F.cross_entropy(cor_matrix, label)
            else:
                for other_logit in other_logits.values():
                    cor_matrix = torch.matmul(logit, other_logit.T)  # (b, b)
                    contrast_loss += F.cross_entropy(cor_matrix, label)
                contrast_loss = contrast_loss / (len(self.models) - 1)  # average contrast loss for the selected model
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

            # optimize
            optimizers = self.optimizers()
            optimizers[training_id].zero_grad()
            self.manual_backward(loss)
            optimizers[training_id].step()

            # log
            self.log(f"{training_modal}/train/loss", loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/rec_loss", rec_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/kld_loss", kld_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/contrast_loss", contrast_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"contrast_weight", contrast_weight, logger=True, on_epoch=True, sync_dist=True)
            self.log(f'{training_modal}/scheduler', self.lr_schedulers()[training_id].get_last_lr()[0], logger=True, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    def validation_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max
        label = torch.arange(batch.shape[0]).to(batch.device)  # (b,) label for contrastive loss

        with torch.no_grad():
            rec_loss_ = {}
            kld_loss_ = {}
            logits_ = {}
            contrast_loss_ = {}
            for name, model in self.models.items():
                # get rec_loss, kld_loss and logits for each model
                model.train()
                rec_loss, kld_loss, mu, _, _ = model.calculate_loss(batch[:, self.data_modal_to_id[name], :, :, :], return_moment=True)
                rec_loss_[name] = rec_loss
                kld_loss_[name] = kld_loss
                logit = model.get_logit(mu)
                logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                logits_[name] = logit

            if self.mean_logit:
                mean_logit = torch.mean(torch.stack(list(logits_.values())), dim=0)  # (b, c)
                mean_logit = mean_logit/(mean_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            
            # calculate contrast loss
            for name, model in self.models.items():
                contrast_loss = 0
                if self.mean_logit:
                    cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                    contrast_loss = F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss
                else:
                    for name2, model2 in self.models.items():
                        if name != name2:
                            cor_matrix = torch.matmul(logits_[name], logits_[name2].T)
                            contrast_loss += F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss/ (len(self.models)-1) # average contrast loss for each model

            # optimize
            rec_loss = sum(rec_loss_.values())
            kld_loss = sum(kld_loss_.values())
            contrast_loss = sum(contrast_loss_.values())
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

            # log
            for name, model in self.models.items():
                self.log(f"{name}/val/loss", loss, logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/rec_loss/", rec_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/kld_loss", kld_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/contrast_loss", contrast_loss_[name], logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/loss", loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/rec_loss", rec_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/kld_loss", kld_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/contrast_loss", contrast_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
         

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