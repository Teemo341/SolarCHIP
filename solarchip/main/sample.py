"""
sample.py —— 对训练好的模型在测试集(validation split)上采样，并保存反归一化后的原始 .pt 张量。

设计要点（与 train.py 的采样流程对齐，但有 6 处刻意不同）:
1. 测试集不 shuffle；enhance_type / modal_list / load_imgs 等沿用训练日志里的参数，
   torch_augment_type 强制为 [1024, 0, 0]（不随机翻转/旋转）；
   time_interval / time_step 由命令行给出。
2. 扩散模型(solarldm / solarctrl)用完整 DDPM 马尔可夫链采样，而不是 DDIM。
3. 不经过 SolarImageLogger（它为了可视化会做数值 clip），
   这里用本文件内的 RawTensorLogger 直接 torch.save 原始张量。
4. 采样结果先反归一化再保存：dataset 的归一化是 signed-log1p → zscore，
   反归一化参数取自 data/dataset/SolarDataset.py 的 modal_status；
   compare/transfer 模型用自己的 target_mean/target_std（或 hmi_mean/hmi_std）
   与 metric_max_log_value 钳位（与其内部指标计算完全一致）。
5. 只保存生成结果，条件、重构、真实数据一律不保存。
6. 每张图单独保存一份 .pt（不再按 batch 合并成 BCHW 张量）：
   - 扩散模型: sample_<时间>.pt（ControlNet 另有 sample_cfg_<时间>.pt）
   - 对比模型: sample_<时间>.pt
   其中 <时间> 是条件模态数据的采集时间串，按 data/utils.py 的
   get_modal_dir 文件命名约定推导（如 20240109_000000 / 20240109_0000）。
7. 保存路径：logs/sample/pt/{目标模态}/{模型名字}/sample_<时间>.pt，
   模型名字从训练日志路径读取；pt 单独放一层，便于后续转成其它数据格式。
8. 虚拟 original 模型（-r original）：不采样，直接把 data/ 里保存好的真实数据
   pt 复制到 logs/sample/pt/{目标模态}/original/，只复制 time_interval / time_step
   筛选后的文件，文件保持原名（内容本身是物理量，与反归一化后的采样结果可比）。9. HMI 目标模态时，采样结果会乘上由真实 HMI 数据统计生成的日面 mask
   （solarchip/visualization/hmi_solar_mask.pt），确保只有太阳本体有数值、
   盘外没有模型产生的干扰项。仅 HMI 需要，其它模态盘外本身就有值。9. --visualization true 时采样后调用 solarchip/visualization/solarplot.py 出图，
   保存到 logs/sample/png/{目标模态}/{模型名字}/，命名与 pt 一致
   （sample_<时间>.png / sample_cfg_<时间>.png），只补充缺失的图；
   -r original 时同样支持，png 保存到 logs/sample/png/{目标模态}/original/，
   命名与复制的 pt 一致（原名换 .png 后缀）。

用法示例:
    # ControlNet: 0094 -> hmi
    python -m solarchip.main.sample \
        -r logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 \
        --time_interval 5000 6000 --time_step 1

    # 无条件 LDM: hmi
    python -m solarchip.main.sample \
        -r checkpoints/solarldm/sd_hmi_uncond \
        --time_interval 5000 6000 --time_step 1

    # 对比模型: Pix2PixCC
    python -m solarchip.main.sample \
        -r logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0094/2026-08-16T16-52-55 \
        --time_interval 5000 6000 --time_step 1

    # 指定 checkpoint 与少量 batch 做快速测试
    python -m solarchip.main.sample \
        -r logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 \
        --time_interval 5000 6000 --time_step 1 \
        --ckpt epoch=000198_val_loss_simple=0.0385.ckpt --max_batches 1

    # 虚拟 original 模型: 直接把真实数据 pt 复制到 logs/sample/pt/{目标模态}/original/
    # (加 --visualization true 可顺带为真实数据出图)
    python -m solarchip.main.sample -r original --target_modal hmi \
        --time_interval 5000 6000 --time_step 1
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# PyTorch >=2.6 默认 torch.load(weights_only=True)，旧 checkpoint 中的 numpy 类型
# 不在默认白名单会报 UnpicklingError。与 train.py 保持一致，一次性加入安全列表。
_safe_numpy = [np.core.multiarray.scalar, np.dtype, np.ndarray]
if hasattr(np, 'dtypes'):
    _safe_numpy.extend(
        getattr(np.dtypes, n) for n in dir(np.dtypes)
        if isinstance(getattr(np.dtypes, n), type) and issubclass(getattr(np.dtypes, n), np.dtype)
    )
torch.serialization.add_safe_globals(_safe_numpy)

from omegaconf import OmegaConf
try:
    from lightning.pytorch import seed_everything
except ImportError:
    from pytorch_lightning import seed_everything

# dataset 内部用 './data/idx_list/...' 相对路径，必须保证 CWD 是仓库根目录
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    print(f'[sample] 当前目录缺少 data/，切换到仓库根目录: {REPO_ROOT}')
    os.chdir(REPO_ROOT)

from solarchip.utils.util import get_obj_from_str
from solarchip.utils import musa_support  # noqa: F401  引入 MUSA 设备兼容补丁
from data.dataset.SolarDataset import modal_status
from data.utils import get_modal_dir, load_list, transfer_id_to_date  # data/ 里记录的数据 id -> 文件路径/采集时间转换

# 训练日志目录名形如 2026-08-07T17-58-09，模型名字是它的父目录名
TIMESTAMP_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$')

# 按模型族分发的采样方式
DIFFUSION_MODEL_NAMES = {'SolarControl', 'SolarLDM'}
COMPARE_MODEL_NAMES = {
    'DannehlPix2PixCC',   # compare.transfer.aia_to_hmi.dannehl_pix2pixcc
    'SayezI2IwFiLM',      # compare.transfer.aia_to_hmi.i2iwfilm
    'DashPix2PixHD',      # compare.transfer.hmi_to_aia.dash_pix2pixhd
    'GalvezSDOMLCNN',     # compare.transfer.hmi_to_aia.sdoml_cnn
}

# 虚拟模型：-r original 时不加载任何训练日志，直接把真实数据复制到保存目录
ORIGINAL_MODEL_NAME = 'original'


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        description='对训练好的模型在测试集上采样，保存反归一化后的原始 .pt 张量。')
    parser.add_argument(
        '-r', '--resume',
        type=str,
        required=True,
        help='训练日志目录，例如 logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09，'
             '须包含 configs/*-project.yaml 与 checkpoints/。',
    )
    parser.add_argument(
        '--time_interval',
        type=int,
        nargs=2,
        required=True,
        metavar=('START', 'END'),
        help='测试集时间区间 [start, end)，覆盖训练日志里的 validation 配置。',
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=1,
        help='测试集采样步长（存在性过滤 idx %% time_step == 0），默认 1。',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='覆盖训练日志里的 batch_size（默认用训练配置的值）。',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='覆盖训练日志里的 num_workers（默认用训练配置的值）。',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='要采样的 checkpoint：checkpoints/ 下的文件名或完整路径；'
             '默认优先 last.ckpt，其次最新的 epoch=*.ckpt。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，保证 DDPM 初始噪声可复现，默认 42。',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='采样设备：auto / cuda / cuda:0 / musa:0 / cpu 等，默认自动选择。',
    )
    parser.add_argument(
        '--save_root',
        type=str,
        default='logs/sample/pt',
        help='保存根目录，默认 logs/sample/pt；最终路径为 '
             'logs/sample/pt/{目标模态}/{模型名字}/。'
             'pt 单独放一层，便于后续转成其它数据格式。',
    )
    parser.add_argument(
        '--sample_subdir',
        type=str,
        default=None,
        help='在每个模型的保存目录下再加一层子目录（如 checkpoint 名），'
             '最终路径为 logs/sample/pt/{目标模态}/{模型名字}/{sample_subdir}/，'
             '用于按权重分别存放采样结果。',
    )
    parser.add_argument(
        '--target_modal',
        type=str,
        default=None,
        help='目标模态（hmi/0094/0131/0171/0193/0211/0304/0335/1600/1700/4500）；'
             '虚拟 original 模型必需，其它模型自动从训练配置读取。',
    )
    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=3.0,
        help='ControlNet classifier-free guidance 强度（hint 置零做无条件分支），'
             '默认 3.0，对应 samples_cfg3.0。',
    )
    parser.add_argument(
        '--no_cfg',
        action='store_true',
        help='跳过 ControlNet 的 CFG 采样，只保存 samples。',
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='只采样前 N 个 batch（快速测试用），默认全部。',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='关闭 DDPM 内部逐 batch 的进度条（外层 batch 进度条仍然保留）。',
    )
    parser.add_argument(
        '--visualization',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='采样后调用 solarplot.py 生成可视化图片，保存到 '
             'logs/sample/png/{目标模态}/{模型名字}/，命名与 pt 一致'
             '（sample_<时间>.png / sample_cfg_<时间>.png），只补充缺失的图。',
    )
    parser.add_argument(
        '--enhance',
        type=str,
        default='none',
        choices=['none', 'log1p'],
        help='可视化显示增强: none=原始物理量(默认，对应 #sym:enhance none)，'
             'log1p=signed-log1p 动态范围压缩。',
    )
    return parser


# ----------------------------------------------------------------------
# 日志 / checkpoint / 模型名字
# ----------------------------------------------------------------------
def load_project_config(logdir: str):
    """读取训练日志里保存的 project 配置（可能因 resume 而存在多份，取最新的）。"""
    cfg_files = sorted(glob.glob(os.path.join(logdir, 'configs', '*-project.yaml')))
    if not cfg_files:
        raise FileNotFoundError(
            f'在 {os.path.join(logdir, "configs")} 下找不到 *-project.yaml，'
            f'请确认 -r 指向的是训练日志目录（含 configs/ 与 checkpoints/）。')
    cfg_file = max(cfg_files, key=os.path.getmtime)
    print(f'[sample] 使用训练配置: {cfg_file}')
    return OmegaConf.load(cfg_file)


def resolve_checkpoint(logdir: str, ckpt_arg: str) -> str:
    if ckpt_arg:
        if os.path.isfile(ckpt_arg):
            return os.path.abspath(ckpt_arg)
        path = os.path.join(logdir, 'checkpoints', ckpt_arg)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'找不到 checkpoint: {path}')
        return path
    last = os.path.join(logdir, 'checkpoints', 'last.ckpt')
    if os.path.isfile(last):
        return last
    ckpts = sorted(glob.glob(os.path.join(logdir, 'checkpoints', '*.ckpt')))
    if not ckpts:
        raise FileNotFoundError(f'{os.path.join(logdir, "checkpoints")} 下没有 .ckpt 文件')
    return max(ckpts, key=os.path.getmtime)


def get_model_name(logdir: str) -> str:
    """从训练日志路径读取模型名字。

    logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 -> ctrl_best_0094-hmi
    checkpoints/solarldm/sd_hmi_uncond                        -> sd_hmi_uncond
    """
    logdir = logdir.rstrip('/')
    base = os.path.basename(logdir)
    if TIMESTAMP_RE.match(base):
        base = os.path.basename(os.path.dirname(logdir))
    return base


# ----------------------------------------------------------------------
# 设备 / 数据
# ----------------------------------------------------------------------
def select_device(arg: str) -> torch.device:
    if arg and arg.lower() != 'auto':
        return torch.device(arg)
    if hasattr(torch, 'musa') and torch.musa.is_available():
        return torch.device('musa:0')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def build_validation_loader(config, time_interval, time_step, batch_size, num_workers):
    """按训练日志的 validation 配置构造测试集，覆盖 time_interval/time_step，
    并强制 torch_augment_type=[1024, 0, 0]（不 shuffle、不随机增强）。

    返回 (loader, params, dataset)；dataset.exist_idx 保存了每个样本对应的
    日期 id，用于推导文件名里的采集时间。
    """
    if 'validation' not in config.data.params:
        raise ValueError('训练日志配置中没有 data.params.validation，无法构造测试集。')

    val_cfg = OmegaConf.to_container(config.data.params.validation, resolve=True)
    params = dict(val_cfg['params'])

    # 1. 测试集超参由命令行给出
    params['time_interval'] = [int(time_interval[0]), int(time_interval[1])]
    params['time_step'] = int(time_step)
    # 2. 采样阶段禁用随机增强，只做 resize 到 1024
    params['torch_augment_type'] = [1024, 0, 0]

    print('[sample] 测试集配置: '
          f"modal_list={params.get('modal_list')}, "
          f"enhance_type={params.get('enhance_type')}, "
          f"log1p_scale={params.get('log1p_scale', 1)}, "
          f'torch_augment_type={params["torch_augment_type"]}, '
          f"time_interval={params['time_interval']}, time_step={params['time_step']}")

    dataset = get_obj_from_str(val_cfg['target'])(**params)
    print(f'[sample] 测试集样本数: {len(dataset)}')

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,           # 需求 1: 测试集不 shuffle
                        num_workers=num_workers,
                        drop_last=False)
    return loader, params, dataset


# ----------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------
def instantiate_model(config):
    """从 project 配置实例化模型。

    SolarControl 训练配置里的 sd_backbone_ckpt 只在初始化时用于加载冻结的主 UNet；
    采样阶段直接从训练 checkpoint 恢复全部权重（含 control_model），
    因此这里移除该依赖，避免要求 backbone ckpt 必须存在。
    """
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    target = model_cfg['target']
    params = dict(model_cfg.get('params', {}))
    cls_name = target.rsplit('.', 1)[-1]

    if cls_name == 'SolarControl':
        if params.get('sd_backbone_ckpt') is not None:
            print('[sample] SolarControl: 采样阶段忽略 sd_backbone_ckpt，'
                  '训练 checkpoint 已包含主 UNet + ControlNet 全部权重。')
        params['sd_backbone_ckpt'] = None
        params['sd_locked'] = False

    return get_obj_from_str(target)(**params)


def load_weights(model, ckpt_path):
    """把 checkpoint 权重载入模型，返回 (global_step, epoch) 供日志输出。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    global_step = int(ckpt.get('global_step', 0) or 0)
    epoch = int(ckpt.get('epoch', 0) or 0)

    if hasattr(model, 'init_from_ckpt'):
        # DDPM 系（SolarLDM / SolarControl）与 solarchip_base 的标准加载入口
        model.init_from_ckpt(ckpt_path)
    else:
        # 对比模型是普通 LightningModule，直接 load_state_dict
        sd = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[sample] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')

    print(f'[sample] 已加载 checkpoint: {ckpt_path} '
          f'(global_step={global_step}, epoch={epoch})')
    return global_step, epoch


# ----------------------------------------------------------------------
# 反归一化：signed-log1p -> zscore 的逆变换
# ----------------------------------------------------------------------
def make_denormalizer(model, target_modal: str, log1p_scale: float):
    """构造反归一化函数。

    dataset 归一化: x -> zscore(log1p(|x|) * sign(x))，其中 log1p 前先乘 log1p_scale。
    逆变换:     z * std + mean -> / log1p_scale -> sign * expm1(|.|)

    - solarldm / solarctrl: 模型内部没有额外归一化，直接查 modal_status。
    - compare/transfer 模型: 用自己的 target_mean/target_std（i2iwfilm 用 hmi_mean/hmi_std）
      并用 metric_max_log_value 对 |log 值| 做钳位 —— 与其内部 _inverse_*_preprocess 一致。
    """
    if hasattr(model, 'target_mean') and hasattr(model, 'target_std'):
        mean, std = float(model.target_mean), float(model.target_std)
        print(f'[sample] 反归一化参数取自模型: mean={mean}, std={std} (模态 {target_modal})')
    elif hasattr(model, 'hmi_mean') and hasattr(model, 'hmi_std'):
        mean, std = float(model.hmi_mean), float(model.hmi_std)
        print(f'[sample] 反归一化参数取自模型: mean={mean}, std={std} (模态 {target_modal})')
    else:
        stats = modal_status[target_modal]
        mean, std = stats['mean'], stats['std']
        print(f'[sample] 反归一化参数取自 SolarDataset.modal_status: '
              f'{target_modal}: mean={mean}, std={std}')

    clamp = getattr(model, 'metric_max_log_value', None)
    if clamp is not None:
        print(f'[sample] 反归一化时对 |log 值| 钳位到 {clamp}（与对比模型指标计算一致）')

    def denormalize(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        signed_log = (x * std + mean) / log1p_scale
        magnitude = signed_log.abs()
        if clamp is not None:
            magnitude = magnitude.clamp(max=clamp)
        return signed_log.sign() * torch.expm1(magnitude)

    return denormalize


# ----------------------------------------------------------------------
# 样本时间戳：data/utils.py 记录了数据 id 与采集日期的转换
# ----------------------------------------------------------------------
def get_sample_time_string(modal: str, day_id: int) -> str:
    """返回该样本在 data/ 中的采集时间串，格式 YYYYMMDD_HHMM(SS)。

    直接复用 data/utils.py 的 get_modal_dir 命名约定（单一事实来源）:
        hmi     -> 20240109_000000
        1700    -> 20240109_0002
        其它aia -> 20240109_0000
    """
    _, pt_path = get_modal_dir(modal, int(day_id))
    fname = os.path.basename(pt_path)
    if modal == 'hmi':
        match = re.search(r'\.(\d{8}_\d{6})_TAI\.pt$', fname)
    else:
        match = re.search(r'(\d{8}_\d{4})_\d{4}\.pt$', fname)
    if match is None:
        raise ValueError(f'无法从 data/ 文件名解析采集时间: {fname}')
    return match.group(1)


# ----------------------------------------------------------------------
# 扩散模型采样（DDPM）
# ----------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample_cfg(model, cond, shape, unconditional_conditioning=None,
                    guidance_scale=1.0, verbose=True):
    """完整 DDPM 采样（逐时间步去噪），支持 classifier-free guidance。

    基础 DDPM 采样器（p_sample_loop）不实现 CFG —— CFG 只在 DDIM 路径里，
    因此这里按 DDIM 的公式手动实现 DDPM + CFG:
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    """
    b = shape[0]
    device = model.betas.device
    img = torch.randn(shape, device=device)
    iterator = tqdm(reversed(range(model.num_timesteps)),
                    desc='DDPM(CFG) sampling', total=model.num_timesteps,
                    disable=not verbose, leave=False)
    for i in iterator:
        t = torch.full((b,), i, device=device, dtype=torch.long)
        eps = model.apply_model(img, t, cond)
        if unconditional_conditioning is not None and guidance_scale != 1.0:
            eps_uncond = model.apply_model(img, t, unconditional_conditioning)
            eps = eps_uncond + guidance_scale * (eps - eps_uncond)

        x_recon = model.predict_start_from_noise(img, t=t, noise=eps)
        model_mean, _, model_log_variance = model.q_posterior(
            x_start=x_recon, x_t=img, t=t)
        noise = torch.randn_like(img)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(img.shape) - 1)))
        img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    return img


@torch.no_grad()
def sample_diffusion_batch(model, batch, denormalize, cfg_scale=None, verbose=True):
    """对一个 batch 做扩散采样，返回 {name: 反归一化后的 cpu 张量 (B, 1, H, W)}。

    SolarLDM（无条件）: 只产生 sample。
    SolarControl: 产生 sample（条件 DDPM）与 sample_cfg（CFG DDPM）。
    """
    z, cond = model.get_input(batch, k=model.first_stage_key)
    b = z.shape[0]
    shape = (b, model.channels, model.image_size, model.image_size)

    # ddim=False → 需求 2: 完整 DDPM 采样，而不是 DDIM
    with model.ema_scope('Sampling'):
        samples, _ = model.sample_log(cond=cond, batch_size=b,
                                      ddim=False, ddim_steps=None,
                                      verbose=verbose)
    x_samples = model.decode_first_stage(samples)
    results = {'sample': denormalize(x_samples.detach().cpu())}

    # ControlNet 的 classifier-free guidance：hint 置零作为无条件分支
    if (cfg_scale is not None and cfg_scale != 1.0
            and isinstance(cond, dict) and 'c_concat' in cond):
        hint = torch.cat(cond['c_concat'], dim=1)
        uncond = {'c_concat': [torch.zeros_like(hint)]}
        with model.ema_scope('Sampling CFG'):
            samples_cfg = ddpm_sample_cfg(
                model, cond=cond, shape=shape,
                unconditional_conditioning=uncond,
                guidance_scale=cfg_scale, verbose=verbose)
        x_cfg = model.decode_first_stage(samples_cfg)
        results['sample_cfg'] = denormalize(x_cfg.detach().cpu())

    return results


# ----------------------------------------------------------------------
# 对比模型采样（单次前向）
# ----------------------------------------------------------------------
@torch.no_grad()
def sample_compare_batch(model, batch, denormalize):
    """对比模型（Pix2PixCC / I2IwFiLM / Pix2PixHD / SDOML-CNN）单次前向采样。

    这些模型直接消费 dataset 归一化后的输入，输出也在归一化空间，
    反归一化后保存。只保存生成结果，不保存条件与真实数据（需求 5）。
    """
    source = batch[model.source_modal].to(model.device).float()
    generated = model(source)
    return {'sample': denormalize(generated.detach().cpu())}


# ----------------------------------------------------------------------
# 新的 logger：保存原始张量，不做任何可视化 clip
# ----------------------------------------------------------------------
class RawTensorLogger:
    """保存原始张量的 logger：每张图单独保存一份 .pt，不做任何可视化 clip。

    SolarImageLogger 会按模态把数值裁剪到可视化范围（get_cmap_and_limits），
    无法用于算指标；这里直接 torch.save 反归一化后的原始张量。
    一个 batch 的 (B, 1, H, W) 张量按第 0 维拆成 B 份逐图保存，
    每份去掉通道维，保存为 (H, W) 二维张量；
    命名为 {name}_{time}.pt，其中 time 是条件模态数据的采集时间串。
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.existing = 0
        os.makedirs(save_dir, exist_ok=True)

    def log_batch(self, tensors: dict, times):
        """tensors: {name: (B, 1, H, W) 张量}; times: 长度为 B 的采集时间串列表。"""
        if not isinstance(times, (list, tuple)):
            raise TypeError('times 必须是长度为 batch 大小的列表')
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            b = tensor.shape[0]
            if len(times) != b:
                raise ValueError(
                    f'{name}: batch 大小 {b} 与时间串数量 {len(times)} 不一致')
            for j in range(b):
                img = tensor[j]
                if img.dim() >= 3 and img.shape[0] == 1:
                    img = img[0]  # (1, H, W) -> (H, W)
                fname = f'{name}_{times[j]}.pt'
                path = os.path.join(self.save_dir, fname)
                if os.path.isfile(path):
                    self.existing += 1
                    continue
                torch.save(img, path)


# ----------------------------------------------------------------------
# HMI 日面 mask：生成模型（尤其 ControlNet）没有约束太阳外围应为 0，
# 会在盘外产生干扰项。这里用真实 HMI 数据统计生成的 mask 把盘外置零，
# 保证保存的 pt 与后续画图都只有太阳本体有数值。
# 仅 HMI 需要，其它模态盘外本身也有物理量。
# ----------------------------------------------------------------------
HMI_SOLAR_MASK_PATH = os.path.join(REPO_ROOT, 'solarchip', 'visualization',
                                   'hmi_solar_mask.pt')
_hmi_solar_mask = None


def get_hmi_solar_mask():
    """加载 HMI 日面 mask (H, W) 0/1 张量（惰性加载 + 缓存）。"""
    global _hmi_solar_mask
    if _hmi_solar_mask is None:
        if not os.path.isfile(HMI_SOLAR_MASK_PATH):
            raise FileNotFoundError(
                f'找不到 HMI 日面 mask: {HMI_SOLAR_MASK_PATH}\n'
                f'请先运行: python solarchip/visualization/generate_hmi_mask.py')
        _hmi_solar_mask = torch.load(HMI_SOLAR_MASK_PATH, weights_only=True)
        print(f'[sample] 已加载 HMI 日面 mask: {HMI_SOLAR_MASK_PATH}')
    return _hmi_solar_mask


def apply_hmi_solar_mask(tensors: dict):
    """对 HMI 结果张量 (B, 1, H, W) 逐图乘上日面 mask (H, W)，盘外置零。"""
    mask = get_hmi_solar_mask()
    return {k: v * mask for k, v in tensors.items() if v is not None}


# ----------------------------------------------------------------------
# 虚拟 original 模型：直接复制真实数据
# ----------------------------------------------------------------------
def run_original_copy(opt):
    """把 data/ 里保存好的真实数据 pt 直接复制到 logs/sample/pt/{目标模态}/original/。

    只复制 time_interval / time_step 筛选后的文件（筛选规则与
    SolarDataset.filter_exist_idx 一致），文件保持原名、内容不做任何处理——
    真实数据本身就是物理量，与反归一化后的采样结果直接可比。
    """
    modal = opt.target_modal
    if not modal:
        raise ValueError('虚拟 original 模型需要 --target_modal 指定目标模态。')
    if modal not in modal_status:
        raise ValueError(f'不支持的目标模态: {modal}，可选: {sorted(modal_status)}')

    exist_idx = load_list(f'./data/idx_list/{modal}_exist_idx.pkl')
    start, end = opt.time_interval
    day_ids = [i for i in range(start, min(end, len(exist_idx)))
               if bool(exist_idx[i]) and i % opt.time_step == 0]
    print(f'[sample] original 模型: 模态 {modal}, '
          f'time_interval=[{start}, {end}), time_step={opt.time_step}, '
          f'共 {len(day_ids)} 个时间点')

    save_dir = os.path.join(opt.save_root, modal, ORIGINAL_MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # --visualization 时给真实数据出图，保存到 logs/sample/png/{模态}/original/，
    # 命名与复制过去的 pt 一致（原名换 .png 后缀），只补充缺失的图
    if opt.visualization:
        # 按需导入，避免不需要可视化时加载 sunpy
        from solarchip.visualization.solarplot import solarplot, format_timestamp
        png_dir = get_png_dir(opt.save_root, modal, ORIGINAL_MODEL_NAME)
        os.makedirs(png_dir, exist_ok=True)
        vis_made = vis_skipped = 0

    copied, skipped = 0, 0
    copied_skip = 0
    enhance = None if opt.enhance == 'none' else opt.enhance
    for day_id in tqdm(day_ids, desc='Copying original data'):
        src = get_modal_dir(modal, day_id)[1]
        if not os.path.isfile(src):
            skipped += 1
            print(f'[sample] 缺少数据文件，跳过: {src}')
            continue
        dst = os.path.join(save_dir, os.path.basename(src))
        if os.path.isfile(dst):
            copied_skip += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

        if opt.visualization:
            png_path = os.path.join(
                png_dir, os.path.basename(src).replace('.pt', '.png'))
            if os.path.isfile(png_path):
                vis_skipped += 1
                continue
            data = torch.load(src, weights_only=True).numpy()
            dt = transfer_id_to_date(day_id)
            time_int = int(dt.strftime('%Y%m%d%H%M'))
            solarplot(data, modal, format_timestamp(time_int), png_path,
                      enhance=enhance)
            vis_made += 1

    if opt.visualization:
        print(f'[sample] original 可视化完成: 生成 {vis_made} 张, '
              f'已存在跳过 {vis_skipped} 张, 保存在 {png_dir}')

    print(f'[sample] original 完成: 复制 {copied} 个文件, '
          f'已存在跳过 {copied_skip} 个文件, 跳过 {skipped} 个缺失文件, '
          f'保存在 {save_dir}')


# ----------------------------------------------------------------------
# 可视化：调用 solarplot 出图（可选，只补充缺失的图）
# ----------------------------------------------------------------------
def get_png_dir(save_root, target_modal, model_name):
    """由 pt 保存根目录推出 png 目录：logs/sample/pt -> logs/sample/png。"""
    root = os.path.abspath(save_root).rstrip('/')
    if os.path.basename(root) == 'pt':
        root = os.path.dirname(root)
    return os.path.join(root, 'png', target_modal, model_name)


def visualize_tensors(results, times, day_ids, target_modal, png_dir):
    """把采样结果张量逐图保存成 png，命名与 pt 一致（sample_<时间>.png），
    只补充缺失的图。返回 (生成数, 跳过数)。"""
    # 按需导入，避免不需要可视化时加载 sunpy
    from solarchip.visualization.solarplot import solarplot, format_timestamp
    os.makedirs(png_dir, exist_ok=True)
    made = skipped = 0
    for name, tensor in results.items():
        if tensor is None:
            continue
        for j, t in enumerate(times):
            png_path = os.path.join(png_dir, f'{name}_{t}.png')
            if os.path.isfile(png_path):
                skipped += 1
                continue
            img = tensor[j]
            if img.dim() >= 3 and img.shape[0] == 1:
                img = img[0]
            dt = transfer_id_to_date(day_ids[j])
            time_int = int(dt.strftime('%Y%m%d%H%M'))
            solarplot(img.numpy(), target_modal, format_timestamp(time_int), png_path)
            made += 1
    return made, skipped


def resolve_modals_from_config(config):
    """仅从训练配置解析 (模型类别, 目标模态, 条件模态)，不实例化模型。

    已有采样需要跳过时使用，避免为纯出图加载模型权重。
    """
    cls_name = config.model.target.rsplit('.', 1)[-1]
    params = config.model.params
    if cls_name in DIFFUSION_MODEL_NAMES:
        target = params.first_stage_key
        time_modal = (params.cond_stage_key if cls_name == 'SolarControl'
                      else params.first_stage_key)
    elif cls_name in COMPARE_MODEL_NAMES:
        target = params.target_modal
        time_modal = params.source_modal
    else:
        raise NotImplementedError(
            f'暂不支持对 {cls_name} 采样。目前支持: '
            f'扩散模型 {sorted(DIFFUSION_MODEL_NAMES)}, '
            f'对比模型 {sorted(COMPARE_MODEL_NAMES)}。')
    return cls_name, target, time_modal


def visualize_existing_pt(all_ids, time_modal, target_modal, save_dir, png_dir,
                          result_names, enhance):
    """从已有 pt 直接出图（不重新采样），只补充缺失的 png。

    返回 (生成数, 已存在跳过数, 缺 pt 跳过数)。
    """
    # 按需导入，避免不需要可视化时加载 sunpy
    from solarchip.visualization.solarplot import solarplot, format_timestamp
    os.makedirs(png_dir, exist_ok=True)
    made = skipped = missing = 0
    for d in tqdm(all_ids, desc='Visualizing existing pt'):
        t = get_sample_time_string(time_modal, d)
        for name in result_names:
            png_path = os.path.join(png_dir, f'{name}_{t}.png')
            if os.path.isfile(png_path):
                skipped += 1
                continue
            pt_path = os.path.join(save_dir, f'{name}_{t}.pt')
            if not os.path.isfile(pt_path):
                missing += 1
                continue
            data = torch.load(pt_path, weights_only=True).numpy()
            dt = transfer_id_to_date(int(d))
            time_int = int(dt.strftime('%Y%m%d%H%M'))
            solarplot(data, target_modal, format_timestamp(time_int), png_path,
                      enhance=enhance)
            made += 1
    return made, skipped, missing


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    opt = get_parser().parse_args()

    # 虚拟 original 模型：不加载训练日志，直接复制真实数据后结束
    if os.path.basename(os.path.abspath(opt.resume).rstrip('/')).lower() == ORIGINAL_MODEL_NAME:
        run_original_copy(opt)
        return

    logdir = os.path.abspath(opt.resume)
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f'训练日志目录不存在: {logdir}')

    # 1. 训练日志配置 + 模型名字（已有采样时无需加载模型权重）
    config = load_project_config(logdir)
    model_name = get_model_name(logdir)
    cls_name, target_modal, time_modal = resolve_modals_from_config(config)
    print(f'[sample] 模型名字: {model_name}, 模型类别: {cls_name}, '
          f'目标模态: {target_modal}, 时间模态: {time_modal}')

    # 2. 测试集（不 shuffle，torch_augment_type=[1024,0,0]）
    data_params = config.data.params
    batch_size = (opt.batch_size if opt.batch_size is not None
                  else int(data_params.batch_size))
    num_workers = (opt.num_workers if opt.num_workers is not None
                   else int(data_params.get('num_workers', 0)))
    val_loader, val_params, val_dataset = build_validation_loader(
        config, opt.time_interval, opt.time_step, batch_size, num_workers)
    all_ids = [int(i) for i in val_dataset.exist_idx]
    print(f'[sample] 测试集共 {len(all_ids)} 个时间点')

    # 3. 保存目录与采样结果名
    save_dir = os.path.join(opt.save_root, target_modal, model_name)
    if opt.sample_subdir:
        save_dir = os.path.join(save_dir, opt.sample_subdir)
    os.makedirs(save_dir, exist_ok=True)
    png_dir = get_png_dir(opt.save_root, target_modal, model_name)
    logger = RawTensorLogger(save_dir)
    print(f'[sample] 保存目录: {save_dir}')

    cfg_scale = (opt.cfg_scale
                 if cls_name == 'SolarControl' and not opt.no_cfg else None)
    result_names = ['sample', 'sample_cfg'] if cfg_scale is not None else ['sample']

    # 4. 已有采样跳过：目标 pt 全部存在的时间点不重新采样（不覆盖已有文件）
    missing_ids = [
        d for d in all_ids
        if not all(os.path.isfile(os.path.join(
            save_dir, f'{n}_{get_sample_time_string(time_modal, d)}.pt'))
            for n in result_names)
    ]
    print(f'[sample] 已有采样结果: {len(all_ids) - len(missing_ids)}/{len(all_ids)} '
          f'个时间点, 需补采样: {len(missing_ids)} 个时间点')

    total_batches = 0
    if missing_ids:
        device = select_device(opt.device)
        ckpt_path = resolve_checkpoint(logdir, opt.ckpt)
        print(f'[sample] checkpoint: {ckpt_path}, device: {device}')
        seed_everything(opt.seed)

        # 模型
        model = instantiate_model(config)
        model.to(device)
        model.eval()
        log1p_scale = float(val_params.get('log1p_scale', 1.0))
        denormalize = make_denormalizer(model, target_modal, log1p_scale)
        load_weights(model, ckpt_path)

        # 只保留缺失的时间点，避免对已存在数据重复采样
        val_dataset.exist_idx = missing_ids
        sample_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers,
                                   drop_last=False)
        for batch_idx, batch in enumerate(tqdm(sample_loader, desc='Sampling batches')):
            if cls_name in DIFFUSION_MODEL_NAMES:
                results = sample_diffusion_batch(
                    model, batch, denormalize,
                    cfg_scale=cfg_scale, verbose=not opt.quiet)
            else:
                results = sample_compare_batch(model, batch, denormalize)

            # HMI 目标模态：乘上日面 mask，把盘外模型产生的干扰项置零
            if target_modal == 'hmi':
                results = apply_hmi_solar_mask(results)

            # 该 batch 内每张图的 day id -> 条件模态采集时间串
            first_tensor = next(iter(batch.values()))
            base = batch_idx * batch_size
            b = first_tensor.shape[0]
            day_ids = [int(val_dataset.exist_idx[base + j]) for j in range(b)]
            times = [get_sample_time_string(time_modal, d) for d in day_ids]

            logger.log_batch(results, times)

            total_batches += 1
            if opt.max_batches is not None and total_batches >= opt.max_batches:
                print(f'[sample] 达到 --max_batches={opt.max_batches}，提前结束。')
                break
        print(f'[sample] 补采样完成: {total_batches} 个 batch '
              f'(已存在跳过 {logger.existing} 个文件), 结果保存在 {save_dir}')
    else:
        print('[sample] 所有时间点采样结果已存在，跳过模型加载与采样。')

    # 5. 可视化：从已有 pt（含刚补采样的）直接出图，只补充缺失 png
    if opt.visualization:
        enhance = None if opt.enhance == 'none' else opt.enhance
        made, skipped, missing = visualize_existing_pt(
            all_ids, time_modal, target_modal, save_dir, png_dir,
            result_names, enhance)
        print(f'[sample] 可视化完成: 生成 {made} 张, 已存在跳过 {skipped} 张, '
              f'缺 pt 跳过 {missing} 张, 保存在 {png_dir}')
    else:
        print('[sample] --visualization 为 false，跳过可视化。')


if __name__ == '__main__':
    main()
