"""
SolarLDM: 在 SolarCHIP 多模态编解码器之上做 Latent Diffusion 训练。

继承自 auxiliary/ldm/models/diffusion/ddpm.py 的 LatentDiffusion
(对应原版 Stable Diffusion 的主类)。

相比父类有三处定制:
1. 用 SolarCHIP 整个 multi-modal wrapper 来构造 first/cond stage,
   只保留 first_stage_key 和 cond_stage_key 真正用到的 AE_CNN,
   其余删除;并且能正确处理 solarchip_mergeaia / solarchip_mergeall
   这种多个 modal 共享一个 AE_CNN 的场景,不会误删。
2. first_stage_key / cond_stage_key 是 SolarCHIP 自定义的模态名,
   例如 'hmi'、'0193' 等;batch 是 {modal: tensor[B, 1, H, W]} 的 dict,
   所以重写 `get_input` 走 SolarCHIP 的 (B, C, H, W) 数据约定,
   不再走 DDPM 默认的 (B, H, W, C) -> rearrange 路径。
3. log_images 改为 SolarCHIP 风格的字典 (`visualization/<modal>/<kind>`),
   与 `solarchip.utils.callback.SolarImageLogger` 配合使用。
"""

import torch
import torch.distributed as dist

from auxiliary.ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from auxiliary.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from solarchip.utils.util import instantiate_from_config


UNCONDITIONAL_CONFIG = "__is_unconditional__"


class SolarLDM(LatentDiffusion):
    """
    SolarCHIP downstream Latent Diffusion Model.
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(
        self,
        solarchip_config,
        first_stage_key: str,
        cond_stage_key: str,
        learning_rate: float = 1e-4,
        normalize_latent_per_channel: bool = False,
        latent_stats_eps: float = 1e-6,
        latent_mean=None,
        latent_std=None,
        cond_latent_mean=None,
        cond_latent_std=None,
        *args,
        **kwargs,
    ):
        # 父类初始化时会调用 self.instantiate_first_stage(first_stage_config)
        # 和 self.instantiate_cond_stage(cond_stage_config)。条件模型仍用占位符
        # 从 SolarCHIP wrapper 中抽取 cond AE；无条件模型则必须把
        # "__is_unconditional__" 原样传给 LatentDiffusion，否则父类会根据
        # concat_mode 把它误判成 concat/cross-attention 模型。
        kwargs.pop("first_stage_config", None)
        requested_cond_stage_config = kwargs.pop("cond_stage_config", None)
        self._solar_is_unconditional = (
            requested_cond_stage_config == UNCONDITIONAL_CONFIG
        )
        self.normalize_latent_per_channel = bool(normalize_latent_per_channel)
        self.latent_stats_eps = float(latent_stats_eps)
        if self.latent_stats_eps <= 0:
            raise ValueError("latent_stats_eps 必须大于 0")
        if self.normalize_latent_per_channel:
            if kwargs.get("scale_by_std", False):
                raise ValueError(
                    "normalize_latent_per_channel 与旧的 scale_by_std 不能同时启用"
                )
            if float(kwargs.get("scale_factor", 1.0)) != 1.0:
                raise ValueError(
                    "逐通道 latent normalization 要求 scale_factor=1.0"
                )
        self._latent_mean_config = latent_mean
        self._latent_std_config = latent_std
        self._cond_latent_mean_config = cond_latent_mean
        self._cond_latent_std_config = cond_latent_std
        parent_cond_stage_config = (
            UNCONDITIONAL_CONFIG
            if self._solar_is_unconditional
            else {"_solarldm_extract_from_wrapper": True}
        )

        self._solarchip_config = solarchip_config  # 临时挂在 self 上,
        # 由于此时 nn.Module 还未完成 __init__,_modules 不存在,
        # 直接赋普通属性不会触发自动 submodule 注册。

        super().__init__(
            first_stage_config=solarchip_config,
            cond_stage_config=parent_cond_stage_config,
            first_stage_key=first_stage_key,
            cond_stage_key=cond_stage_key,
            *args,
            **kwargs,
        )

        # 用户层级常用的训练率(LatentDiffusion.configure_optimizers 会读它)
        self.learning_rate = learning_rate

        # 释放掉只在 init 期间需要的临时引用
        if hasattr(self, "_solarchip_config"):
            try:
                del self._solarchip_config
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Latent normalization
    # ------------------------------------------------------------------
    def _register_latent_stats(self, prefix, channels, mean_config, std_config):
        """注册可随 checkpoint 保存的逐通道统计量。"""
        if not self.normalize_latent_per_channel:
            return

        mean_name = f"{prefix}latent_mean"
        std_name = f"{prefix}latent_std"
        initialized_name = f"{prefix}latent_stats_initialized"
        if hasattr(self, mean_name):
            return

        if (mean_config is None) != (std_config is None):
            raise ValueError(
                f"{mean_name} 与 {std_name} 必须同时提供或同时省略"
            )

        initialized = mean_config is not None
        if initialized:
            mean = torch.as_tensor(mean_config, dtype=torch.float32).flatten()
            std = torch.as_tensor(std_config, dtype=torch.float32).flatten()
            if mean.numel() != channels or std.numel() != channels:
                raise ValueError(
                    f"{prefix or 'first_'}latent stats 应有 {channels} 个通道，"
                    f"实际 mean={mean.numel()}, std={std.numel()}"
                )
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError("latent mean/std 必须全部为有限值")
            if (std <= self.latent_stats_eps).any():
                raise ValueError(
                    f"latent std 必须全部大于 latent_stats_eps={self.latent_stats_eps}"
                )
        else:
            mean = torch.zeros(channels, dtype=torch.float32)
            std = torch.ones(channels, dtype=torch.float32)

        self.register_buffer(mean_name, mean.view(1, channels, 1, 1))
        self.register_buffer(std_name, std.view(1, channels, 1, 1))
        self.register_buffer(
            initialized_name, torch.tensor(initialized, dtype=torch.bool)
        )

    @staticmethod
    def _posterior_to_latent(encoder_posterior, sample=True):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            return encoder_posterior.sample() if sample else encoder_posterior.mode()
        if isinstance(encoder_posterior, torch.Tensor):
            return encoder_posterior
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )

    def _uses_first_stats_for_condition(self):
        return (
            self.cond_stage_model is not None
            and self.cond_stage_model is self.first_stage_model
        )

    def _latent_stats_prefix(self, condition=False):
        if condition and not self._uses_first_stats_for_condition():
            return "cond_"
        return ""

    def _allow_uninitialized_stats(self):
        trainer = getattr(self, "_trainer", None)
        return trainer is not None and getattr(trainer, "sanity_checking", False)

    def _normalize_latent(self, z, condition=False):
        if not self.normalize_latent_per_channel:
            return z
        if not isinstance(z, torch.Tensor) or z.dim() != 4:
            stage = "condition" if condition else "first-stage"
            shape = getattr(z, "shape", None)
            raise ValueError(
                f"{stage} 逐通道 latent normalization 期望 BCHW tensor，"
                f"实际 type={type(z)}, shape={shape}"
            )
        prefix = self._latent_stats_prefix(condition)
        initialized = getattr(self, f"{prefix}latent_stats_initialized")
        if not bool(initialized.item()):
            if self._allow_uninitialized_stats():
                return z
            stage = "condition" if condition else "first-stage"
            raise RuntimeError(
                f"{stage} 逐通道 latent stats 尚未初始化；训练时应先完成 "
                "on_train_start 的完整训练集预计算，推理时必须加载包含统计量的 checkpoint"
            )
        mean = getattr(self, f"{prefix}latent_mean").to(dtype=z.dtype)
        std = getattr(self, f"{prefix}latent_std").to(dtype=z.dtype)
        if z.shape[1] != mean.shape[1]:
            raise ValueError(
                f"latent 通道数不匹配：tensor={z.shape[1]}, stats={mean.shape[1]}"
            )
        return (z - mean) / std

    def _denormalize_latent(self, z):
        if not self.normalize_latent_per_channel:
            return z
        if not isinstance(z, torch.Tensor) or z.dim() != 4:
            shape = getattr(z, "shape", None)
            raise ValueError(
                "first-stage latent denormalization 期望 BCHW tensor，"
                f"实际 type={type(z)}, shape={shape}"
            )
        initialized = self.latent_stats_initialized
        if not bool(initialized.item()):
            if self._allow_uninitialized_stats():
                return z
            raise RuntimeError(
                "first-stage 逐通道 latent stats 尚未初始化，无法安全解码"
            )
        mean = self.latent_mean.to(dtype=z.dtype)
        std = self.latent_std.to(dtype=z.dtype)
        if z.shape[1] != mean.shape[1]:
            raise ValueError(
                f"latent 通道数不匹配：tensor={z.shape[1]}, stats={mean.shape[1]}"
            )
        return z * std + mean

    def _new_channel_moments(self, channels, device):
        return {
            "sum": torch.zeros(channels, dtype=torch.float64, device=device),
            "sum_sq": torch.zeros(channels, dtype=torch.float64, device=device),
            "count": torch.zeros((), dtype=torch.float64, device=device),
        }

    def _accumulate_channel_moments(self, moments, z):
        if z.dim() != 4:
            raise ValueError(
                f"逐通道 latent normalization 期望 BCHW tensor，实际 shape={tuple(z.shape)}"
            )
        if z.shape[1] != moments["sum"].numel():
            raise ValueError(
                f"latent 通道数不匹配：tensor={z.shape[1]}, "
                f"stats={moments['sum'].numel()}"
            )
        values = z.detach().to(dtype=torch.float64)
        moments["sum"].add_(values.sum(dim=(0, 2, 3)))
        moments["sum_sq"].add_(values.square().sum(dim=(0, 2, 3)))
        moments["count"].add_(
            torch.tensor(
                values.shape[0] * values.shape[2] * values.shape[3],
                dtype=torch.float64,
                device=values.device,
            )
        )

    def _finalize_global_channel_moments(self, moments):
        if dist.is_available() and dist.is_initialized():
            for value in moments.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if moments["count"].item() == 0:
            raise RuntimeError("训练集为空，无法预计算 latent stats")
        mean = moments["sum"] / moments["count"]
        variance = moments["sum_sq"] / moments["count"] - mean.square()
        std = variance.clamp_min(self.latent_stats_eps ** 2).sqrt()
        return mean.float(), std.float(), int(moments["count"].item())

    def _store_latent_stats(self, mean, std, condition=False):
        prefix = self._latent_stats_prefix(condition)
        getattr(self, f"{prefix}latent_mean").copy_(
            mean.view(1, -1, 1, 1)
        )
        getattr(self, f"{prefix}latent_std").copy_(
            std.view(1, -1, 1, 1)
        )
        getattr(self, f"{prefix}latent_stats_initialized").fill_(True)

    @torch.no_grad()
    def initialize_latent_stats_from_dataloader(self, dataloader):
        """遍历完整训练 dataloader，流式预计算目标/条件逐通道统计量。"""
        need_first = not bool(self.latent_stats_initialized.item())
        has_separate_condition = (
            self.cond_stage_model is not None
            and not self._uses_first_stats_for_condition()
        )
        need_condition = (
            has_separate_condition
            and not bool(self.cond_latent_stats_initialized.item())
        )
        if not need_first and not need_condition:
            return

        first_moments = None
        cond_moments = None
        num_batches = 0
        for batch in dataloader:
            num_batches += 1
            if need_first:
                x = self._solar_get_raw(batch, self.first_stage_key).to(self.device)
                raw_z = self._posterior_to_latent(
                    self.encode_first_stage(x), sample=True
                )
                if first_moments is None:
                    first_moments = self._new_channel_moments(
                        raw_z.shape[1], raw_z.device
                    )
                self._accumulate_channel_moments(first_moments, raw_z)
                del x, raw_z

            if need_condition:
                xc = self._solar_get_raw(batch, self.cond_stage_key).to(self.device)
                raw_c = LatentDiffusion.get_learned_conditioning(self, xc)
                if cond_moments is None:
                    cond_moments = self._new_channel_moments(
                        raw_c.shape[1], raw_c.device
                    )
                self._accumulate_channel_moments(cond_moments, raw_c)
                del xc, raw_c

            if num_batches % 100 == 0 and self._is_global_zero():
                print(
                    f"[SolarLDM] latent stats precompute: {num_batches} batches"
                )

        initialized = []
        if need_first:
            if first_moments is None:
                raise RuntimeError(
                    "训练 dataloader 未产生 batch，无法统计 first-stage latent"
                )
            mean, std, count = self._finalize_global_channel_moments(first_moments)
            self._store_latent_stats(mean, std)
            initialized.append(("first-stage", mean, std, count))
        if need_condition:
            if cond_moments is None:
                raise RuntimeError(
                    "训练 dataloader 未产生 batch，无法统计 condition latent"
                )
            mean, std, count = self._finalize_global_channel_moments(cond_moments)
            self._store_latent_stats(mean, std, condition=True)
            initialized.append(("condition", mean, std, count))

        if self._is_global_zero():
            for stage, mean, std, count in initialized:
                print(
                    f"[SolarLDM] initialized {stage} full-train-set "
                    f"per-channel latent stats ({count} values/channel): "
                    f"mean=[{mean.min().item():.6g}, {mean.max().item():.6g}], "
                    f"std=[{std.min().item():.6g}, {std.max().item():.6g}]"
                )

    @staticmethod
    def _is_global_zero():
        return (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_rank() == 0
        )

    @torch.no_grad()
    def on_train_start(self):
        super().on_train_start()
        if self.normalize_latent_per_channel:
            self.initialize_latent_stats_from_dataloader(
                self.trainer.train_dataloader
            )

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.normalize_latent_per_channel:
            return

        # 旧的全局标量 scale_by_std 兼容路径。SolarLDM 的 batch 已经是
        # BCHW，不能调用父类假设 BHWC 的 get_input。
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
            and self._is_global_zero()
        ):
            assert self.scale_factor == 1.0, (
                "Don't set both scale_factor and scale_by_std"
            )
            print("### USING STD-RESCALING ###")
            x = self._solar_get_raw(batch, self.first_stage_key).to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self._posterior_to_latent(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    # ------------------------------------------------------------------
    # First / Cond stage:从 SolarCHIP wrapper 抽出需要的部分,其它丢弃
    # ------------------------------------------------------------------
    def instantiate_first_stage(self, config):
        """
        config 实际上就是 solarchip_config(在 __init__ 里塞进来的)。
        步骤:
            1. 用 solarchip_config 实例化整个 SolarCHIP wrapper(11 个模态)。
            2. 通过 wrapper.get_model(first_stage_key) 拿到 first 的 AE_CNN;
               通过 wrapper.get_model(cond_stage_key)  拿到 cond  的 AE_CNN。
               (mergeaia / mergeall 的情况下,两者可能指向同一个 AE_CNN。)
            3. 把这两个 AE_CNN 从 wrapper.model_dict 里 pop 出来,
               这样 wrapper 走出本函数作用域被 GC 时,只会带走
               没用上的 AE_CNN —— 用上的因为已经被 self.first_stage_model /
               self.cond_stage_model 持有,不会被释放。
            4. 注意:这种做法对 solarchip_base(每个 modal 一个 AE_CNN)、
               solarchip_mergeaia('hmi' + 'aia' 两个)、
               solarchip_mergeall(只有 'all' 一个)三种情况都安全:
               所有共享同一个 AE_CNN 的 modal 会通过 `m is needed_model`
               一次性识别出来,不会重复删,也不会漏删。
        """
        # config 在 __init__ 里被原样透传,这里再做一次容错
        if config is None:
            config = getattr(self, "_solarchip_config", None)
        assert config is not None, "SolarLDM: solarchip_config 不能为空"

        wrapper = instantiate_from_config(config)
        assert hasattr(wrapper, "model_dict") and hasattr(wrapper, "get_model"), (
            "SolarLDM 要求 solarchip_config 实例化出的对象具备 .model_dict 和 .get_model(),"
            "目前实现的 solarchip_base / solarchip_mergeaia / solarchip_mergeall 都满足。"
        )

        # 找到 first / cond 真正对应的 AE_CNN 对象。无条件模型只保留
        # first-stage AE，不实例化 cond_stage_model，也不会执行条件编码。
        first_model = wrapper.get_model(self.first_stage_key)
        if self._solar_is_unconditional:
            cond_model = None
        elif self.cond_stage_key == self.first_stage_key:
            cond_model = first_model
        else:
            cond_model = wrapper.get_model(self.cond_stage_key)

        first_channels = int(getattr(first_model, "feature_dim", self.channels))
        self._register_latent_stats(
            "",
            first_channels,
            self._latent_mean_config,
            self._latent_std_config,
        )
        if cond_model is not None and cond_model is not first_model:
            cond_channels = int(getattr(cond_model, "feature_dim", self.channels))
            self._register_latent_stats(
                "cond_",
                cond_channels,
                self._cond_latent_mean_config,
                self._cond_latent_std_config,
            )

        # 把需要保留的 AE_CNN 从 wrapper 中摘出来,避免 wrapper 被 GC 时连带它们
        keep_ids = {id(first_model)}
        if cond_model is not None:
            keep_ids.add(id(cond_model))
        keys_to_pop = [
            k for k, m in wrapper.model_dict.items() if id(m) in keep_ids
        ]
        for k in keys_to_pop:
            # nn.ModuleDict 支持 del / pop
            del wrapper.model_dict[k]

        # 同时清空 wrapper 上其它会阻止 GC 的 nn.Module 属性(可选,
        # 主要是为了让显存释放更立即;wrapper 离开作用域 Python 会自动处理。)
        for attr_name in ("rec_loss_fn", "contrastive_loss_fn"):
            if hasattr(wrapper, attr_name):
                try:
                    delattr(wrapper, attr_name)
                except AttributeError:
                    pass

        # 真正挂到 self 上(此时 nn.Module 已注册为 submodule)
        self.first_stage_model = first_model.eval()
        self.first_stage_model.train = disabled_train
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

        if not self._solar_is_unconditional:
            # 把 cond_model 暂存到 self.__dict__ —— 这样不会触发 nn.Module 的
            # 自动注册(否则 cond_model 会先以 _pending 名义被注册一次,
            # 等会儿 instantiate_cond_stage 再赋值到 cond_stage_model 时
            # 又会注册一次,出现重复)。
            self.__dict__["_pending_cond_model"] = cond_model
            self.__dict__["_pending_cond_is_first"] = (cond_model is first_model)

        print(
            f"[SolarLDM] first_stage_key='{self.first_stage_key}', "
            f"cond_stage_key='{self.cond_stage_key}', "
            f"剪枝后 SolarCHIP wrapper 剩余 keys={list(wrapper.model_dict.keys())} "
            f"(这些会被释放)"
        )

    def instantiate_cond_stage(self, config):
        """
        cond model 已经在 instantiate_first_stage 里准备好,
        这里只负责把它挂到 self.cond_stage_model 上,并设置 train/eval、grad。
        """
        if self._solar_is_unconditional or config == UNCONDITIONAL_CONFIG:
            self.__dict__.pop("_pending_cond_model", None)
            self.__dict__.pop("_pending_cond_is_first", None)
            self.cond_stage_model = None
            print(f"[SolarLDM] Training {self.__class__.__name__} unconditionally")
            return

        cond_model = self.__dict__.pop("_pending_cond_model", None)
        cond_is_first = self.__dict__.pop("_pending_cond_is_first", False)
        if cond_model is None:
            raise RuntimeError(
                "SolarLDM.instantiate_cond_stage 在 instantiate_first_stage 之前被调用,"
                "或 _pending_cond_model 已被清理。"
            )

        if cond_is_first:
            print("[SolarLDM] cond_stage_key == first_stage_key:复用 first_stage_model 作为 cond_stage_model")
            self.cond_stage_model = self.first_stage_model
            return

        if not self.cond_stage_trainable:
            self.cond_stage_model = cond_model.eval()
            self.cond_stage_model.train = disabled_train
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False
        else:
            self.cond_stage_model = cond_model  # 保持训练模式 + 需要梯度

    # ------------------------------------------------------------------
    # 数据接口:SolarCHIP 的 batch 已经是 {modal: tensor[B, 1, H, W]}
    # ------------------------------------------------------------------
    def _solar_get_raw(self, batch, k):
        """从 SolarCHIP 风格的 batch 中取出 modal=k 对应的原图张量。"""
        x = batch[k]
        # 兼容偶尔出现的 (B, H, W) 形式
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x.to(memory_format=torch.contiguous_format).float()

    def get_first_stage_encoding(self, encoder_posterior):
        if not self.normalize_latent_per_channel:
            return super().get_first_stage_encoding(encoder_posterior)
        z = self._posterior_to_latent(encoder_posterior, sample=True)
        return self._normalize_latent(z)

    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs: bool = False,
        force_c_encode: bool = False,
        cond_key=None,
        return_original_cond: bool = False,
        bs=None,
    ):
        """
        与 LatentDiffusion.get_input 同语义,但 raw-x 走 SolarCHIP 数据约定:
        batch[k] 已经是 (B, C, H, W),不需要 rearrange。
        """
        x = self._solar_get_raw(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                xc = self._solar_get_raw(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
        else:
            c = None
            xc = None

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    # ------------------------------------------------------------------
    # 条件编码:cross-attention 模式下将 cond latent flatten 为序列
    # ------------------------------------------------------------------
    def get_learned_conditioning(self, c):
        """重写父类方法，cross-attention 模式下将 [B, C, H, W] flatten 为 [B, H*W, C]。
        
        SpatialTransformer 的 cross-attention 需要 context 形如 (B, N, context_dim)，
        其中 N 是序列长度。SolarCHIP 的 AE 编码器输出 latent [B, 32, 32, 32]，
        需将 H, W 合并为序列维度。
        """
        c = super().get_learned_conditioning(c)
        if self.normalize_latent_per_channel:
            c = self._normalize_latent(c, condition=True)
        if self.model.conditioning_key == 'crossattn' and c.dim()==4:
            # [B, C, H, W] -> [B, H*W, C]
            c = c.flatten(2).transpose(1, 2)
        return c

    @torch.no_grad()
    def decode_first_stage(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if self.normalize_latent_per_channel:
            z = self._denormalize_latent(z)
        return super().decode_first_stage(
            z,
            predict_cids=predict_cids,
            force_not_quantize=force_not_quantize,
        )

    def differentiable_decode_first_stage(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if self.normalize_latent_per_channel:
            z = self._denormalize_latent(z)
        return super().differentiable_decode_first_stage(
            z,
            predict_cids=predict_cids,
            force_not_quantize=force_not_quantize,
        )

    # ------------------------------------------------------------------
    # log_images:沿用 SolarCHIP 的 'visualization/<modal>/<kind>' 命名
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
        **kwargs,
    ):
        """
        与 SolarImageLogger 约定的输出格式对齐:返回 dict,key 是 'visualization/<modal>/<kind>',
        value 是 CPU tensor,形状统一为 (B, C, H, W),由 SolarImageLogger
        自行根据 channel 数(<3 用 cmap,>=3 取前 3 个通道做 RGB 归一化)处理可视化。

        生成的 key 包括:
            visualization/<first_stage_key>/input         —— 原图
            visualization/<first_stage_key>/rec           —— first stage AE 重建
            visualization/<first_stage_key>/latent        —— first stage 隐空间(若 return_latent)
            visualization/<cond_stage_key>/cond_input     —— 条件模态原图(若 cond≠first)
            visualization/<cond_stage_key>/cond_latent    —— 条件隐空间(若 return_latent 且为 tensor)
            visualization/<first_stage_key>/samples       —— DDIM 采样后再 first stage 解码的图
        """
        log = {}
        use_ddim = ddim_steps is not None

        z, c, x, xrec, xc = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        N = min(x.shape[0], N)

        log[f"visualization/{self.first_stage_key}/input"] = x.detach().cpu()
        log[f"visualization/{self.first_stage_key}/rec"] = xrec.detach().cpu()

        if return_latent:
            log[f"visualization/{self.first_stage_key}/latent"] = self._to_loggable_latent(z).detach().cpu()

        # cond 模态:仅当 cond ≠ first 且 xc 是图像 tensor 时展示原图
        if xc is not None and isinstance(xc, torch.Tensor) and self.cond_stage_key != self.first_stage_key:
            log[f"visualization/{self.cond_stage_key}/cond_input"] = xc.detach().cpu()
            if return_latent and isinstance(c, torch.Tensor):
                # cross-attn 模式下 c 是 [B, H*W, C], reshape 回 [B, C, H, W] 以便可视化
                cond_latent = c
                if cond_latent.dim() == 3:
                    hw = int(cond_latent.shape[1] ** 0.5)
                    cond_latent = cond_latent.transpose(1, 2).reshape(c.shape[0], -1, hw, hw)
                log[f"visualization/{self.cond_stage_key}/cond_latent"] = self._to_loggable_latent(cond_latent).detach().cpu()

        if sample:
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            x_samples = self.decode_first_stage(samples)
            log[f"visualization/{self.first_stage_key}/samples"] = x_samples.detach().cpu()

        return log

    @staticmethod
    def _to_loggable_latent(z):
        """
        把可能是 DiagonalGaussianDistribution / tuple / Tensor 的 latent
        统一成 (B, C, H, W) tensor,以便 SolarImageLogger 渲染。
        """
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.mode()
        elif isinstance(z, tuple):
            z = z[0]
        return z


if __name__ == "__main__":
    print("SolarLDM MRO:", [c.__name__ for c in SolarLDM.__mro__])
