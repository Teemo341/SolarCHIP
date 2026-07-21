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

from pytorch_lightning.utilities.rank_zero import rank_zero_only

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
    # scale_by_std 兼容: 覆写父类的 on_train_batch_start,
    # 因为 LatentDiffusion 的版本调用 super().get_input() 会走到
    # DDPM.get_input(batch, key), 它假设 batch[key] 是 (B, H, W, C) 的原始图像,
    # 而 SolarLDM 的 batch 是 {modal: tensor(B, C, H, W)} 的 dict。
    # ------------------------------------------------------------------
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert self.scale_factor == 1.0, (
                "Don't set both scale_factor and scale_by_std"
            )
            print("### USING STD-RESCALING ###")
            x = self._solar_get_raw(batch, self.first_stage_key).to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
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
        if self.model.conditioning_key == 'crossattn' and c.dim()==4:
            # [B, C, H, W] -> [B, H*W, C]
            c = c.flatten(2).transpose(1, 2)
        return c

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
