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
            cls_ctr_loss_tmp = self.contrastive_loss_fn.cls_contrastive_loss(z_hmi, z_aia) if self.cls_ctr_weight > 0 else 0
            pat_ctr_loss_tmp = self.contrastive_loss_fn.pat_contrastive_loss(z_hmi, z_aia) if self.pat_ctr_weight > 0 else 0
            int_ctr_loss_tmp = self.contrastive_loss_fn.int_contrastive_loss(z_hmi, z_aia) if self.int_ctr_weight > 0 else 0
            cls_ctr_loss += cls_ctr_loss_tmp
            pat_ctr_loss += pat_ctr_loss_tmp
            int_ctr_loss += int_ctr_loss_tmp
            loss_dict[f"cls_ctr_loss/hmi_{modal}"] = cls_ctr_loss_tmp.item() if self.cls_ctr_weight > 0 else 0
            loss_dict[f"pat_ctr_loss/hmi_{modal}"] = pat_ctr_loss_tmp.item() if self.pat_ctr_weight > 0 else 0
            loss_dict[f"int_ctr_loss/hmi_{modal}"] = int_ctr_loss_tmp.item() if self.int_ctr_weight > 0 else 0
        cls_ctr_loss = cls_ctr_loss / (len(self.id_to_modal)-1) if self.cls_ctr_weight > 0 else 0
        pat_ctr_loss = pat_ctr_loss / (len(self.id_to_modal)-1) if self.pat_ctr_weight > 0 else 0
        int_ctr_loss = int_ctr_loss / (len(self.id_to_modal)-1) if self.int_ctr_weight > 0 else 0
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
        hmi = batch['hmi'].to(self.device)
        rec_hmi, z_hmi = self.model_dict['hmi'](hmi)
        total_loss, loss_dict_tmp = self.rec_loss_fn(hmi, rec_hmi, posteriors=z_hmi)
        for k, v in loss_dict_tmp.items():
            loss_dict[f"hmi/{k}"] = v
        loss_dict['hmi/total_loss'] = total_loss.item()
        cls_ctr_loss, pat_ctr_loss, int_ctr_loss = 0, 0, 0
        for i in range(1, len(self.id_to_modal)):
            modal = self.id_to_modal[i]
            aia = batch[modal].to(self.device)
            rec_aia, z_aia = self.get_model(modal)(aia)
            if self.cls_ctr_weight>0:
                
            cls_ctr_loss_tmp = self.contrastive_loss_fn.cls_contrastive_loss(z_hmi, z_aia) if self.cls_ctr_weight > 0 else 0
            pat_ctr_loss_tmp = self.contrastive_loss_fn.pat_contrastive_loss(z_hmi, z_aia) if self.pat_ctr_weight > 0 else 0
            int_ctr_loss_tmp = self.contrastive_loss_fn.int_contrastive_loss(z_hmi, z_aia) if self.int_ctr_weight > 0 else 0
            cls_ctr_loss += cls_ctr_loss_tmp
            pat_ctr_loss += pat_ctr_loss_tmp
            int_ctr_loss += int_ctr_loss_tmp
            loss_dict[f"cls_ctr_loss/hmi_{modal}"] = cls_ctr_loss_tmp.item() if self.cls_ctr_weight > 0 else 0
            loss_dict[f"pat_ctr_loss/hmi_{modal}"] = pat_ctr_loss_tmp.item() if self.pat_ctr_weight > 0 else 0
            loss_dict[f"int_ctr_loss/hmi_{modal}"] = int_ctr_loss_tmp.item() if self.int_ctr_weight > 0 else 0
        cls_ctr_loss = cls_ctr_loss / (len(self.id_to_modal)-1) if self.cls_ctr_weight > 0 else 0
        pat_ctr_loss = pat_ctr_loss / (len(self.id_to_modal)-1) if self.pat_ctr_weight > 0 else 0
        int_ctr_loss = int_ctr_loss / (len(self.id_to_modal)-1) if self.int_ctr_weight > 0 else 0
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

            



    # functions for training
    def training_step(self, batch, batch_idx):

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