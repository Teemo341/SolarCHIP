"""
SolarControl: 在 SolarLDM 之上叠一个 ControlNet 风格的控制分支。

设计选择:
  1. hint = 另一模态经 SolarCHIP AE 编码后的 latent,shape = (B, 128, 16, 16),
     已经在 latent 空间,无需再做 8x 下采样 —— 因此把原版 ControlNet 的
     `input_hint_block`(走 3 次 stride-2 conv)替换为一个仅做通道投影的轻量结构。
  2. `sd_locked=True` 时完整冻结 backbone，只训练 ControlNet；
     `sd_locked=False` 时可选择 backbone 与 control 分支联合训练。
  3. 即使 backbone 冻结，forward 也不能用 `torch.no_grad()` 包住 decoder，
     因为 control residual 仍需穿过冻结层反传到 ControlNet。
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

from pathlib import Path

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
                              False 时把整个主 UNet 也一起训练。
        sd_backbone_ckpt:     HMI unconditional SD checkpoint；锁定训练时必填。
        sd_backbone_use_ema:  是否优先用源 SD 的 EMA 参数，默认 True。
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
        sd_backbone_ckpt=None,
        sd_backbone_use_ema: bool = True,
        *args,
        **kwargs,
    ):
        if sd_locked and kwargs.get("use_ema", False):
            raise ValueError(
                "sd_locked=True 时主 UNet 完全冻结，不能启用只跟踪主 UNet 的 "
                "use_ema；请设 use_ema=False"
            )
        if sd_locked and sd_backbone_ckpt is None:
            raise ValueError(
                "sd_locked=True 必须提供 sd_backbone_ckpt，不能冻结随机初始化的主 UNet"
            )
        if sd_backbone_ckpt is not None and not Path(sd_backbone_ckpt).is_file():
            raise FileNotFoundError(
                f"找不到 SD backbone checkpoint: {sd_backbone_ckpt}"
            )
        # 强制 conditioning_key=None:cond 不走 DiffusionWrapper 的 concat/crossattn,
        # 全部从 ControlNet 分支注入(本类自己重写的 apply_model 接管)。
        kwargs["conditioning_key"] = None

        # 保存 ckpt_path: super().__init__ 中 DDPM 会消费它去加载主 UNet/AE,
        # 但此时 control_model 尚未创建, ControlNet 权重会丢失。
        # 因此在 control_model 创建后需要手动补加载。
        _ckpt_path = kwargs.get("ckpt_path", None)
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control
        self.sd_locked = bool(sd_locked)
        self.cond_drop_prob = float(cond_drop_prob)
        self.sd_backbone_ckpt = sd_backbone_ckpt
        self.sd_backbone_use_ema = bool(sd_backbone_use_ema)

        # 补加载: 从 ckpt_path 中恢复 ControlNet 分支权重
        _controlnet_from_ckpt = (
            _ckpt_path is not None and Path(_ckpt_path).is_file()
        )
        if _controlnet_from_ckpt:
            self._load_controlnet_from_ckpt(_ckpt_path)

        if self.sd_backbone_ckpt is not None:
            if _controlnet_from_ckpt:
                # ControlNet 已从 ckpt_path 加载训练好的权重 (含 zero-conv),
                # 此时只加载 SD backbone 的 UNet + latent stats,
                # 跳过 encoder 拷贝和 zero-conv 清零, 避免覆盖已训练的 ControlNet。
                state_dict = self._load_checkpoint_state_dict(
                    self.sd_backbone_ckpt
                )
                unet_tensors, ema_parameters = (
                    self._load_main_unet_from_sd_state(
                        state_dict, use_ema=self.sd_backbone_use_ema
                    )
                )
                latent_stats = self._load_first_stage_latent_stats(state_dict)
                print(
                    f"[SolarControl] 从 {self.sd_backbone_ckpt} 加载 SD backbone: "
                    f"UNet tensors={unet_tensors}, EMA params={ema_parameters}, "
                    f"latent stats={latent_stats} "
                    f"(ControlNet 保留自 ckpt_path, 不做重新初始化)"
                )
            else:
                # 首次训练: 完整初始化 (encoder 拷贝 + zero-conv 清零)
                self.initialize_from_sd_checkpoint(
                    self.sd_backbone_ckpt,
                    use_ema=self.sd_backbone_use_ema,
                )
        self._set_sd_backbone_trainable(not self.sd_locked)

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

    @staticmethod
    def _load_checkpoint_state_dict(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 SD backbone checkpoint: {path}")
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        except TypeError:
            try:
                checkpoint = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"checkpoint state_dict 类型错误: {type(state_dict)}")
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _copy_module_state_strict(source, target, label):
        try:
            target.load_state_dict(source.state_dict(), strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"ControlNet {label} 与主 UNet 结构不一致") from error

    def _load_controlnet_from_ckpt(self, path):
        """从完整的 SolarControl checkpoint 中提取并加载 ControlNet 分支权重。

        由于 __init__ 中 control_model 在 super().__init__ (含 DDPM.init_from_ckpt)
        之后才创建, ControlNet 权重不会在首次加载时恢复。此方法在 control_model
        创建后被调用, 补加载 ControlNet 分支的参数。
        """
        state_dict = self._load_checkpoint_state_dict(path)
        prefix = "control_model."
        control_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not control_state:
            print(f"[SolarControl] 警告: checkpoint {path} 中未找到 control_model.* 权重,"
                  f" ControlNet 分支将保持随机初始化。")
            return
        missing, unexpected = self.control_model.load_state_dict(
            control_state, strict=False
        )
        print(f"[SolarControl] 从 {path} 加载 ControlNet 权重: "
              f"{len(control_state)} 个参数, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    def _load_main_unet_from_sd_state(self, state_dict, use_ema):
        diffusion_model = self.model.diffusion_model
        target_state = diffusion_model.state_dict()
        prefixes = ("model.diffusion_model.", "diffusion_model.")
        source_state = None
        for prefix in prefixes:
            candidate = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if candidate:
                source_state = candidate
                break
        if source_state is None:
            raise KeyError(
                "SD checkpoint 中找不到 model.diffusion_model.* 权重"
            )

        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        if missing or unexpected:
            raise RuntimeError(
                "SD backbone 与 ControlNet 主 UNet 结构不完全一致："
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        diffusion_model.load_state_dict(source_state, strict=True)

        ema_parameters_loaded = 0
        if use_ema:
            missing_ema = []
            with torch.no_grad():
                for name, parameter in diffusion_model.named_parameters():
                    shadow_name = (f"diffusion_model.{name}").replace(".", "")
                    key = f"model_ema.{shadow_name}"
                    if key not in state_dict:
                        missing_ema.append(key)
                        continue
                    source = state_dict[key]
                    if source.shape != parameter.shape:
                        raise RuntimeError(
                            f"EMA 参数 shape 不匹配: {key}, "
                            f"source={tuple(source.shape)}, target={tuple(parameter.shape)}"
                        )
                    parameter.copy_(source)
                    ema_parameters_loaded += 1
            if missing_ema:
                raise RuntimeError(
                    "要求使用 SD EMA 权重，但 checkpoint 缺少 EMA 参数："
                    f"{missing_ema[:5]}"
                )
        return len(source_state), ema_parameters_loaded

    def _load_first_stage_latent_stats(self, state_dict):
        if not self.normalize_latent_per_channel:
            return 0
        required = (
            "latent_mean",
            "latent_std",
            "latent_stats_initialized",
        )
        missing = [key for key in required if key not in state_dict]
        if missing:
            raise RuntimeError(
                "SD checkpoint 缺少逐通道 HMI latent stats，不能与当前归一化协议兼容："
                f"{missing}。请使用第 2 包之后重新训练的 HMI unconditional checkpoint"
            )
        if not bool(state_dict["latent_stats_initialized"].item()):
            raise RuntimeError("SD checkpoint 中的 HMI latent stats 尚未初始化")
        with torch.no_grad():
            for key in required:
                target = getattr(self, key)
                source = state_dict[key]
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"{key} shape 不匹配: source={tuple(source.shape)}, "
                        f"target={tuple(target.shape)}"
                    )
                target.copy_(source)
        return len(required)

    def _initialize_control_encoder_from_main_unet(self):
        backbone = self.model.diffusion_model
        control = self.control_model
        self._copy_module_state_strict(
            backbone.time_embed, control.time_embed, "time_embed"
        )
        self._copy_module_state_strict(
            backbone.input_blocks, control.input_blocks, "input_blocks"
        )
        self._copy_module_state_strict(
            backbone.middle_block, control.middle_block, "middle_block"
        )
        if hasattr(backbone, "label_emb") or hasattr(control, "label_emb"):
            if not (hasattr(backbone, "label_emb") and hasattr(control, "label_emb")):
                raise RuntimeError("主 UNet 与 ControlNet 的 label_emb 配置不一致")
            self._copy_module_state_strict(
                backbone.label_emb, control.label_emb, "label_emb"
            )

    @staticmethod
    def _zero_module_parameters(module):
        count = 0
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
                count += 1
        return count

    def _reset_and_validate_control_outputs(self):
        modules = list(self.control_model.zero_convs)
        modules.append(self.control_model.middle_block_out)
        if hasattr(self.control_model, "input_hint_block"):
            hint_children = list(self.control_model.input_hint_block.children())
            if hint_children:
                modules.append(hint_children[-1])

        parameter_count = sum(
            self._zero_module_parameters(module) for module in modules
        )
        nonzero = []
        for module_index, module in enumerate(modules):
            for name, parameter in module.named_parameters():
                if torch.count_nonzero(parameter).item() != 0:
                    nonzero.append(f"module[{module_index}].{name}")
        if nonzero:
            raise RuntimeError(f"ControlNet zero-conv 初始化失败: {nonzero[:5]}")
        return parameter_count

    def initialize_from_sd_checkpoint(self, path, use_ema=True):
        """严格加载 SD backbone，并用其 encoder 初始化 ControlNet。"""
        state_dict = self._load_checkpoint_state_dict(path)
        unet_tensors, ema_parameters = self._load_main_unet_from_sd_state(
            state_dict, use_ema=use_ema
        )
        latent_stats = self._load_first_stage_latent_stats(state_dict)
        self._initialize_control_encoder_from_main_unet()
        zero_parameters = self._reset_and_validate_control_outputs()
        print(
            f"[SolarControl] initialized from {path}: "
            f"UNet tensors={unet_tensors}, EMA params={ema_parameters}, "
            f"latent stats={latent_stats}, zeroed control params={zero_parameters}"
        )

    def _set_sd_backbone_trainable(self, trainable):
        """统一设置主 UNet 的梯度与模式，避免 optimizer 外仍计算参数梯度。"""
        self.model.requires_grad_(trainable)
        if trainable and self.training:
            self.model.train()
        else:
            self.model.eval()
        if not trainable:
            for parameter in self.model.parameters():
                parameter.grad = None

    def train(self, mode=True):
        """Lightning 切回 train 模式时，锁定的 backbone 仍保持 eval。"""
        super().train(mode)
        if getattr(self, "sd_locked", False):
            self.model.eval()
        return self

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
    # 优化器:锁定时只训练 ControlNet；解锁时才加入整个主 UNet
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from torch.optim.lr_scheduler import LambdaLR

        lr = self.learning_rate
        self._set_sd_backbone_trainable(not self.sd_locked)
        params = [p for p in self.control_model.parameters() if p.requires_grad]
        if not self.sd_locked:
            params += [p for p in self.model.parameters() if p.requires_grad]

        if self.learn_logvar:
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert "target" in self.scheduler_config, (
                "scheduler_config 需要 target 字段"
            )
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
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
