"""
SolarControl: 在 SolarLDM 之上叠一个 ControlNet 风格的控制分支。

设计选择(由用户确认):
  1. hint = 另一模态经 SolarCHIP AE 编码后的 latent,shape = (B, 128, 16, 16),
     已经在 latent 空间,无需再做 8x 下采样 —— 因此把原版 ControlNet 的
     `input_hint_block`(走 3 次 stride-2 conv)替换为一个仅做通道投影的轻量结构。
  2. backbone(主 UNet)从零开始训,与 control 分支**联合训练**。
     —— 原版 ControlNet 的 `ControlledUnetModel.forward` 用 `with torch.no_grad():`
        把 input_blocks / middle_block 锁住,我们要写一个解锁版。
  3. backbone 不锁定。
     `configure_optimizers` 同时优化 `self.model.parameters()`(整个 UNet)
     + `self.control_model.parameters()`(ControlNet 分支)。
     `self.first_stage_model` / `self.cond_stage_model`(SolarCHIP AE)
     仍然由 SolarLDM 设置为 eval + requires_grad=False,保持冻结。

继承关系:
  SolarControl  →  SolarLDM  →  auxiliary.ldm.models.diffusion.ddpm.LatentDiffusion
                                                ↑
                            (SolarCHIP-style 编解码器、log_images、get_input)

  网络组件(只是 nn.Module,不和 LatentDiffusion 发生继承耦合):
    - SolarControlledUnetModel:auxiliary.ControlNet.cldm.cldm.ControlledUnetModel
      的子类,移除 torch.no_grad,使主 UNet 可训练。
    - SolarControlNet:auxiliary.ControlNet.cldm.cldm.ControlNet 的子类,
      `input_hint_block` 改为单层 zero-conv 投影(latent 形状直接对齐)。
"""

import os
import sys

import torch
import torch.nn as nn

from auxiliary.ControlNet.cldm.cldm import ControlNet, ControlledUnetModel
from auxiliary.ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module,
    timestep_embedding,
)
from auxiliary.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential

from downstream.ldm.SolarLDM import SolarLDM
from solarchip.utils.util import instantiate_from_config


# ======================================================================
# 网络组件
# ======================================================================
class SolarControlledUnetModel(ControlledUnetModel):
    """
    去掉 `torch.no_grad()` 的 ControlledUnetModel —— input_blocks / middle_block
    也参与训练,适配从零联训。其余 forward 路径与原版完全一致。
    """

    def forward(self, x, timesteps=None, context=None, control=None,
                only_mid_control=False, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h = h + control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class SolarControlNet(ControlNet):
    """
    ControlNet 的子类。hint 已经是 SolarCHIP AE 的 latent(形状与 z_t 一致),
    所以把原版需要做 8x 空间下采样的 input_hint_block 换成一个轻量的
    "通道投影 + zero-conv" —— 保持 ControlNet 在初始化时控制信号为 0 的特性。
    """

    def __init__(self, *args, **kwargs):
        # 走父类把所有 zero_convs / input_blocks / middle_block 都正常建好,
        # 然后只覆盖 input_hint_block。
        super().__init__(*args, **kwargs)

        # 父类没有把 hint_channels 存成属性,需要从 kwargs 里捞;
        # 也可能是位置参数,这里两种都兼容。
        hint_channels = kwargs.get("hint_channels", None)
        if hint_channels is None:
            # 从原 input_hint_block 的第一层 conv 推断
            first_conv = self.input_hint_block[0]
            hint_channels = first_conv.in_channels

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(self.dims, hint_channels, self.model_channels, 3, padding=1),
            nn.SiLU(),
            zero_module(
                conv_nd(self.dims, self.model_channels, self.model_channels, 3, padding=1)
            ),
        )


# ======================================================================
# Lightning 模型
# ======================================================================
class SolarControl(SolarLDM):
    """
    在 SolarLDM 之上挂一个 ControlNet 分支。

    yaml 里需要的新增字段:
        control_stage_config: 实例化 SolarControlNet 的 config(target+params)。
        only_mid_control:     bool;True 时只用 middle_block 的 control,
                              其余 skip 连接不加 control(用于消融)。
        sd_locked:            bool;True 时只训 ControlNet 分支(原版做法),
                              False 时把整个主 UNet 也一起训(用户选这个)。
        cond_drop_prob:       float;训练时随机把 hint 置零的概率,用于做
                              classifier-free guidance(可选,默认 0)。
    """

    def __init__(
        self,
        control_stage_config,
        only_mid_control: bool = False,
        sd_locked: bool = False,
        cond_drop_prob: float = 0.0,
        control_scales=None,
        *args,
        **kwargs,
    ):
        # 强制 conditioning_key=None:cond 不走 DiffusionWrapper 的 concat/crossattn,
        # 全部从 ControlNet 分支注入(本类自己重写的 apply_model 接管)。
        kwargs["conditioning_key"] = None
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control
        self.sd_locked = sd_locked
        self.cond_drop_prob = float(cond_drop_prob)

        # ControlNet 默认输出 = len(zero_convs) + 1 (middle_block_out)
        n_outs = len(self.control_model.zero_convs) + 1
        if control_scales is None:
            self.control_scales = [1.0] * n_outs
        else:
            assert len(control_scales) == n_outs, (
                f"control_scales 长度应为 {n_outs}(等于 zero_convs 数量+1),"
                f"实际 {len(control_scales)}"
            )
            self.control_scales = list(control_scales)

    # ------------------------------------------------------------------
    # 数据接口:第一阶段得到 z,第二阶段单独编码 cond modality 得到 hint
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_input(self, batch, k=None, bs=None, *args, **kwargs):
        """
        SolarLDM 的 get_input 在 conditioning_key=None 时不会编码 cond;
        这里手工补上,把 cond modality 也过一次 SolarCHIP AE 拿到 hint。

        返回:
            z:    (B, 128, 16, 16) —— first stage 编码后的 target latent
            cond: dict(c_concat=[hint])  —— hint 是 cond stage 编码后的 latent
        """
        first_key = k if k is not None else self.first_stage_key

        # ---- target latent z ----
        x = self._solar_get_raw(batch, first_key)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        # ---- control hint = cond modality 经 cond_stage AE 编码 ----
        xc = self._solar_get_raw(batch, self.cond_stage_key)
        if bs is not None:
            xc = xc[:bs]
        xc = xc.to(self.device)
        hint = self.get_learned_conditioning(xc).detach()

        # 训练期可选的 hint dropout(用于 classifier-free guidance)
        if self.training and self.cond_drop_prob > 0.0:
            keep = (
                torch.rand(hint.shape[0], device=hint.device) > self.cond_drop_prob
            ).float().view(-1, 1, 1, 1)
            hint = hint * keep

        return z, dict(c_concat=[hint])

    # ------------------------------------------------------------------
    # 前向:ControlNet 分支 + 主 UNet 注入控制信号
    # ------------------------------------------------------------------
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict) and "c_concat" in cond, (
            f"SolarControl.apply_model 期望 cond 是 {{c_concat: [hint]}},实际:{type(cond)}"
        )
        diffusion_model = self.model.diffusion_model  # SolarControlledUnetModel

        hint = torch.cat(cond["c_concat"], dim=1)
        # 没有 text/crossattn 上下文 —— 太阳图本身已经够强,控制信号走 ControlNet 分支
        context = None

        control = self.control_model(
            x=x_noisy, hint=hint, timesteps=t, context=context
        )
        control = [c * s for c, s in zip(control, self.control_scales)]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=context,
            control=control,
            only_mid_control=self.only_mid_control,
        )
        return eps

    # ------------------------------------------------------------------
    # 优化器:UNet + ControlNet 联合训练(SolarCHIP AE 仍然冻结)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            # 全部主 UNet 一起练(用户选项:from scratch + 联合微调)
            params += list(self.model.parameters())
        else:
            # 兼容原版 ControlNet 锁定模式:只放开 output_blocks + out
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        if self.learn_logvar:
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    # ------------------------------------------------------------------
    # 可视化:在 SolarLDM 的基础上加 hint 和受控采样结果
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_images(
        self,
        batch,
        N: int = 4,
        sample: bool = True,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        return_latent: bool = True,
        unconditional_guidance_scale: float = 1.0,
        **kwargs,
    ):
        log = {}
        use_ddim = ddim_steps is not None

        # 原始图(用于展示输入/重建)
        x = self._solar_get_raw(batch, self.first_stage_key)[:N].to(self.device)
        xc = self._solar_get_raw(batch, self.cond_stage_key)[:N].to(self.device)
        N = min(x.shape[0], N)

        # 编码
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
        hint = self.get_learned_conditioning(xc).detach()
        xrec = self.decode_first_stage(z)

        log[f"visualization/{self.first_stage_key}/input"] = x.detach().cpu()
        log[f"visualization/{self.first_stage_key}/rec"] = xrec.detach().cpu()
        log[f"visualization/{self.cond_stage_key}/cond_input"] = xc.detach().cpu()
        if return_latent:
            log[f"visualization/{self.first_stage_key}/latent"] = z.detach().cpu()
            log[f"visualization/{self.cond_stage_key}/cond_latent"] = hint.detach().cpu()

        if sample:
            cond = {"c_concat": [hint]}
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(
                    cond=cond, batch_size=N, ddim=use_ddim,
                    ddim_steps=ddim_steps, eta=ddim_eta,
                )
            x_samples = self.decode_first_stage(samples)
            log[f"visualization/{self.first_stage_key}/samples"] = x_samples.detach().cpu()

            # classifier-free guidance(若 hint dropout 训练 + 指定 scale>1)
            if unconditional_guidance_scale > 1.0:
                uc = {"c_concat": [torch.zeros_like(hint)]}
                with self.ema_scope("Plotting CFG"):
                    samples_cfg, _ = self.sample_log(
                        cond=cond, batch_size=N, ddim=use_ddim,
                        ddim_steps=ddim_steps, eta=ddim_eta,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc,
                    )
                x_cfg = self.decode_first_stage(samples_cfg)
                log[f"visualization/{self.first_stage_key}/samples_cfg{unconditional_guidance_scale:.1f}"] = x_cfg.detach().cpu()

        return log


if __name__ == "__main__":
    print("SolarControl MRO:", [c.__name__ for c in SolarControl.__mro__])
    print("SolarControlNet base:", SolarControlNet.__bases__)
    print("SolarControlledUnetModel base:", SolarControlledUnetModel.__bases__)
