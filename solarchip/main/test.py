"""
test.py —— 根据采样结果与真实数据（original）计算指标，并可选生成可视化图片。

流程:
1. 输入参数与 sample.py 保持一致，额外新增 --metrics 与 --visualization。
2. 先检查 logs/sample/pt/{目标模态}/{模型名字}/ 下是否已有 time_interval /
   time_step 指定时间点的采样结果；缺失时自动以子进程调用 sample.py 补采样，
   采样参数与本脚本保持一致。
3. 计算指标：真实值取 logs/sample/pt/{目标模态}/original/ 里同一时间点的目标
   模态数据；若 original 目录缺少对应文件，同样自动调用
   sample.py -r original 补复制。例如 0094->hmi 的 controlnet，拿生成的 hmi
   对比 original 里对应时间的 hmi。结果保存为
   logs/sample/metrics/{目标模态}/{模型名字}/metrics.json；
   如果存在 CFG 采样（sample_cfg_*.pt），会同时计算 CFG 的指标并另存为
   metrics_cfg.json。
4. --visualization true 时委托 sample.py（--visualization）出图，保存到
   logs/sample/png/{目标模态}/{模型名字}/，命名与 sample.py 一致
   （sample_<时间>.png / sample_cfg_<时间>.png），只补充缺失的图。

用法示例:
    python -m solarchip.main.test \
        -r logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 \
        --time_interval 5000 6000 --time_step 1 \
        --metrics mse psnr ssim --visualization true

自定义指标: 在下方 METRIC_REGISTRY 里注册一个
    def 你的指标(pred: torch.Tensor, gt: torch.Tensor) -> float
函数即可，之后 --metrics 列表里写它的名字就能计算。
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.chdir(REPO_ROOT)

# 复用 sample.py 的路径/配置/时间戳工具，保证两个脚本对同一时间点的判断完全一致
from solarchip.main.sample import (
    DIFFUSION_MODEL_NAMES,
    COMPARE_MODEL_NAMES,
    ORIGINAL_MODEL_NAME,
    load_project_config,
    get_model_name,
    build_validation_loader,
    get_sample_time_string,
    get_png_dir,
)
from data.utils import get_modal_dir


# ----------------------------------------------------------------------
# 指标实现：pred / gt 都是 (H, W) 物理量张量，返回单个 float
# ----------------------------------------------------------------------
def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def _pair_data_range(gt: torch.Tensor) -> float:
    """PSNR/SSIM 的数据范围：取真实图的动态范围（max - min）。"""
    data_range = float(gt.max() - gt.min())
    if data_range <= 0:
        data_range = max(float(gt.abs().max()), 1e-6)
    return data_range


def metric_mse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """均方误差 MSE。"""
    return float(torch.mean((pred.float() - gt.float()) ** 2))


def metric_mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """平均绝对误差 MAE：比 MSE 更少被少量高强度区域主导。"""
    return float(torch.mean((pred.float() - gt.float()).abs()))


def metric_mape(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """平均绝对百分比误差 MAPE (%)。

    定义: mean(|pred - gt| / (|gt| + eps)) * 100, eps = 1.0（物理量单位，如高斯）。
    eps 用于防止 gt≈0 的噪声像素除零；|gt| >> eps 的像素即标准相对误差。
    对"少量高强度区域 + 大量近零噪声"的太阳图，MAPE 按像素自身量级归一，
    不偏向强场、也不会因预测全 0 而得到很小的值。
    """
    pred = pred.float()
    gt = gt.float()
    eps = 1.0
    return float(torch.mean((pred - gt).abs() / (gt.abs() + eps)) * 100.0)


def metric_nmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """归一化均方误差 NMSE：MSE / var(gt)，跨模态可比，对幅度不敏感。"""
    pred = pred.float()
    gt = gt.float()
    return float(torch.mean((pred - gt) ** 2) / (torch.var(gt) + 1e-8))


def metric_pearson(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """单张图的 Pearson 相关系数（尺度不变，衡量结构/形状）。"""
    pred = pred.float().flatten()
    gt = gt.float().flatten()
    pred = pred - pred.mean()
    gt = gt - gt.mean()
    denom = torch.sqrt((pred * pred).sum() * (gt * gt).sum())
    return float((pred * gt).sum() / denom.clamp_min(1e-12))


def metric_ccc(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """单张图的 Lin 一致性相关系数 CCC（同时惩罚形状偏差与幅度偏差）。"""
    pred = pred.float()
    gt = gt.float()
    pred_mean = pred.mean()
    gt_mean = gt.mean()
    cov = ((pred - pred_mean) * (gt - gt_mean)).mean()
    var_pred = ((pred - pred_mean) ** 2).mean()
    var_gt = ((gt - gt_mean) ** 2).mean()
    denom = var_pred + var_gt + (pred_mean - gt_mean) ** 2
    return float(2.0 * cov / (denom + 1e-8))


def metric_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """峰值信噪比 PSNR（dB），数据范围取真实图动态范围。"""
    mse = torch.mean((pred.float() - gt.float()) ** 2)
    if float(mse) == 0.0:
        return float('inf')
    data_range = _pair_data_range(gt)
    return float(10.0 * math.log10(data_range ** 2 / float(mse)))


def metric_ssim(pred: torch.Tensor, gt: torch.Tensor,
                window_size: int = 11, sigma: float = 1.5,
                k1: float = 0.01, k2: float = 0.03) -> float:
    """结构相似性 SSIM（11x11 高斯窗，K1=0.01, K2=0.03）。"""
    pred = pred.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    gt = gt.float().unsqueeze(0).unsqueeze(0)
    kernel = _gaussian_window(window_size, sigma).unsqueeze(0).unsqueeze(0)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad)
    mu2 = F.conv2d(gt, kernel, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * gt, kernel, padding=pad) - mu12

    data_range = _pair_data_range(gt)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
    return float(ssim_map.mean())


# 指标注册表：新增指标只需在这里注册一个 (pred, gt) -> float 的函数
METRIC_REGISTRY = {
    'mse': metric_mse,
    'mae': metric_mae,
    'mape': metric_mape,
    'nmse': metric_nmse,
    'psnr': metric_psnr,
    'ssim': metric_ssim,
    'pearson': metric_pearson,
    'ccc': metric_ccc,
}


# ----------------------------------------------------------------------
# 命令行参数：与 sample.py 保持一致，另加 --metrics / --visualization
# ----------------------------------------------------------------------
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
        description='根据采样结果与真实数据计算指标，可选生成可视化图片。')
    parser.add_argument(
        '-r', '--resume',
        type=str,
        required=True,
        help='训练日志目录，例如 logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09。',
    )
    parser.add_argument(
        '--time_interval',
        type=int,
        nargs=2,
        required=True,
        metavar=('START', 'END'),
        help='测试集时间区间 [start, end)。',
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=1,
        help='测试集采样步长，默认 1。',
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
        help='要采样的 checkpoint（转给 sample.py），默认 last.ckpt。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（转给 sample.py），默认 42。',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='采样设备（转给 sample.py）：auto / cuda / cuda:0 / musa:0 / cpu。',
    )
    parser.add_argument(
        '--save_root',
        type=str,
        default='logs/sample',
        help='保存根目录，默认 logs/sample；pt 在 {root}/pt、指标在 {root}/metrics、'
             '可视化在 {root}/png。',
    )
    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=3.0,
        help='ControlNet classifier-free guidance 强度（转给 sample.py），默认 3.0。',
    )
    parser.add_argument(
        '--no_cfg',
        action='store_true',
        help='跳过 ControlNet 的 CFG 采样（转给 sample.py），只处理 sample。',
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='只处理前 N 个 batch（快速测试用，同时转给 sample.py）。',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='关闭 DDPM 内部逐 batch 的进度条（转给 sample.py）。',
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['mse', 'psnr', 'ssim'],
        help='要计算的指标列表，例如 --metrics mse psnr ssim；'
             f'可用指标: {sorted(METRIC_REGISTRY)}。',
    )
    parser.add_argument(
        '--visualization',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='是否生成可视化图片（委托 sample.py --visualization），'
             '存到 {save_root}/png/...。',
    )
    return parser


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def resolve_model_info(config):
    """从训练配置读取模型类别 / 目标模态 / 时间模态（不实例化模型）。"""
    cls_name = config.model.target.rsplit('.', 1)[-1]
    if cls_name in DIFFUSION_MODEL_NAMES:
        target_modal = config.model.params.first_stage_key
        time_modal = (config.model.params.cond_stage_key
                      if cls_name == 'SolarControl'
                      else config.model.params.first_stage_key)
    elif cls_name in COMPARE_MODEL_NAMES:
        target_modal = config.model.params.target_modal
        time_modal = config.model.params.source_modal
    else:
        raise NotImplementedError(
            f'暂不支持 {cls_name}。目前支持: 扩散模型 {sorted(DIFFUSION_MODEL_NAMES)}, '
            f'对比模型 {sorted(COMPARE_MODEL_NAMES)}。')
    return cls_name, target_modal, time_modal


def expected_sample_files(pt_dir, day_ids, time_modal, has_cfg):
    """返回 (存在的文件列表, 缺失的文件列表)。"""
    expected = []
    for day_id in day_ids:
        t = get_sample_time_string(time_modal, day_id)
        names = ['sample', 'sample_cfg'] if has_cfg else ['sample']
        expected.extend(os.path.join(pt_dir, f'{name}_{t}.pt') for name in names)
    missing = [p for p in expected if not os.path.isfile(p)]
    return expected, missing


def build_sample_cmd(opt, has_cfg):
    """构造与 test.py 参数一致的 sample.py 命令行。"""
    cmd = [
        sys.executable, '-m', 'solarchip.main.sample',
        '-r', opt.resume,
        '--time_interval', str(opt.time_interval[0]), str(opt.time_interval[1]),
        '--time_step', str(opt.time_step),
        '--seed', str(opt.seed),
        '--device', str(opt.device),
        '--save_root', os.path.join(opt.save_root, 'pt'),
    ]
    if opt.batch_size is not None:
        cmd += ['--batch_size', str(opt.batch_size)]
    if opt.num_workers is not None:
        cmd += ['--num_workers', str(opt.num_workers)]
    if opt.ckpt is not None:
        cmd += ['--ckpt', opt.ckpt]
    if has_cfg:
        cmd += ['--cfg_scale', str(opt.cfg_scale)]
    if opt.no_cfg:
        cmd += ['--no_cfg']
    if opt.max_batches is not None:
        cmd += ['--max_batches', str(opt.max_batches)]
    if opt.quiet:
        cmd += ['--quiet']
    return cmd


def finite_or_none(value):
    """把 inf/nan 转成 None，保证 JSON 合法。"""
    value = float(value)
    return value if math.isfinite(value) else None


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    opt = get_parser().parse_args()

    logdir = os.path.abspath(opt.resume)
    if os.path.basename(logdir.rstrip('/')) == ORIGINAL_MODEL_NAME:
        raise ValueError('test.py 需要传入真实模型的训练日志路径；'
                         'original 是真实数据本身，不需要算指标。')
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f'训练日志目录不存在: {logdir}')

    # 1. 训练配置与路径
    config = load_project_config(logdir)
    model_name = get_model_name(logdir)
    cls_name, target_modal, time_modal = resolve_model_info(config)
    has_cfg = (cls_name == 'SolarControl') and not opt.no_cfg
    print(f'[test] 模型: {model_name} ({cls_name}), 目标模态: {target_modal}, '
          f'时间模态: {time_modal}, CFG: {has_cfg}')

    root = opt.save_root
    pt_dir = os.path.join(root, 'pt', target_modal, model_name)
    original_dir = os.path.join(root, 'pt', target_modal, ORIGINAL_MODEL_NAME)
    metrics_dir = os.path.join(root, 'metrics', target_modal, model_name)

    # 2. 与 sample.py 完全一致的 day id 列表
    data_params = config.data.params
    batch_size = (opt.batch_size if opt.batch_size is not None
                  else int(data_params.batch_size))
    num_workers = (opt.num_workers if opt.num_workers is not None
                   else int(data_params.get('num_workers', 0)))
    _, _, dataset = build_validation_loader(
        config, opt.time_interval, opt.time_step, batch_size, num_workers)
    day_ids = [int(i) for i in dataset.exist_idx]
    if opt.max_batches is not None:
        day_ids = day_ids[:opt.max_batches * batch_size]
    print(f'[test] 测试集共 {len(day_ids)} 个时间点')

    # 3. 检查采样结果，缺失时调用 sample.py 补采样
    _, missing = expected_sample_files(pt_dir, day_ids, time_modal, has_cfg)
    if missing:
        print(f'[test] 缺少 {len(missing)} 个采样文件，调用 sample.py 补采样 ...')
        cmd = build_sample_cmd(opt, has_cfg)
        if opt.visualization:
            cmd += ['--visualization']  # 补采样时顺带出图
        print('[test] 运行: ' + ' '.join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        _, missing = expected_sample_files(pt_dir, day_ids, time_modal, has_cfg)
        if missing:
            raise RuntimeError(f'补采样后仍缺少 {len(missing)} 个文件，'
                               f'例如: {missing[:3]}')
    else:
        print('[test] 采样结果完整，无需补采样。')

    # 3b. 检查 original 真实数据，缺失时调用 sample.py -r original 补复制
    expected_gt = [
        os.path.join(original_dir,
                     os.path.basename(get_modal_dir(target_modal, d)[1]))
        for d in day_ids
    ]
    missing_gt = [p for p in expected_gt if not os.path.isfile(p)]
    if missing_gt:
        print(f'[test] 缺少 {len(missing_gt)} 个真实数据文件，'
              f'调用 sample.py -r original 补复制 ...')
        cmd = [
            sys.executable, '-m', 'solarchip.main.sample',
            '-r', ORIGINAL_MODEL_NAME,
            '--target_modal', target_modal,
            '--time_interval', str(opt.time_interval[0]), str(opt.time_interval[1]),
            '--time_step', str(opt.time_step),
            '--save_root', os.path.join(opt.save_root, 'pt'),
        ]
        print('[test] 运行: ' + ' '.join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        missing_gt = [p for p in expected_gt if not os.path.isfile(p)]
        if missing_gt:
            print(f'[test] 警告: 补复制后仍缺少 {len(missing_gt)} 个真实数据文件，'
                  f'这些时间点将跳过（例如: {missing_gt[:3]}）。')
    else:
        print('[test] 真实数据完整，无需补复制。')

    # 4. 校验指标名并计算：有 CFG 采样时同时计算 sample 与 sample_cfg 两份指标
    for name in opt.metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f'未知指标 {name}，可用指标: {sorted(METRIC_REGISTRY)}')

    kinds = ['sample', 'sample_cfg'] if has_cfg else ['sample']
    os.makedirs(metrics_dir, exist_ok=True)
    for kind in kinds:
        per_sample = {name: {} for name in opt.metrics}
        skipped_gt = skipped_pred = 0
        for day_id in tqdm(day_ids, desc=f'Computing metrics ({kind})'):
            t = get_sample_time_string(time_modal, day_id)
            pred_path = os.path.join(pt_dir, f'{kind}_{t}.pt')
            gt_path = os.path.join(original_dir,
                                   os.path.basename(get_modal_dir(target_modal, day_id)[1]))
            if not os.path.isfile(pred_path):
                skipped_pred += 1
                print(f'[test] 缺少采样文件，跳过 {t}: {pred_path}')
                continue
            if not os.path.isfile(gt_path):
                skipped_gt += 1
                print(f'[test] 缺少真实数据，跳过 {t}: {gt_path}')
                continue
            pred = torch.load(pred_path, weights_only=True)
            gt = torch.load(gt_path, weights_only=True)
            for name in opt.metrics:
                per_sample[name][t] = METRIC_REGISTRY[name](pred, gt)

        summary = {}
        for name in opt.metrics:
            values = list(per_sample[name].values())
            summary[name] = {
                'mean': finite_or_none(float(np.mean(values))) if values else None,
                'std': finite_or_none(float(np.std(values))) if values else None,
                'n': len(values),
            }
            print(f'[test] {kind} {name}: mean={summary[name]["mean"]}, '
                  f'std={summary[name]["std"]}, n={summary[name]["n"]}')

        report = {
            'model_name': model_name,
            'target_modal': target_modal,
            'time_modal': time_modal,
            'sample_kind': kind,
            'time_interval': list(opt.time_interval),
            'time_step': opt.time_step,
            'n_samples': len(day_ids),
            'n_compared': len(per_sample[opt.metrics[0]]) if opt.metrics else 0,
            'n_skipped_missing_gt': skipped_gt,
            'n_skipped_missing_pred': skipped_pred,
            'metrics': summary,
            'per_sample': per_sample,
        }
        json_name = 'metrics.json' if kind == 'sample' else 'metrics_cfg.json'
        metrics_path = os.path.join(metrics_dir, json_name)
        with open(metrics_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'[test] 指标已保存: {metrics_path}')

    # 5. 可视化：委托 sample.py（--visualization），只补充缺失的图
    if opt.visualization:
        png_dir = get_png_dir(os.path.join(opt.save_root, 'pt'),
                              target_modal, model_name)
        names = ['sample', 'sample_cfg'] if has_cfg else ['sample']
        expected_pngs = [
            os.path.join(png_dir,
                         f'{name}_{get_sample_time_string(time_modal, d)}.png')
            for d in day_ids for name in names
        ]
        missing_pngs = [p for p in expected_pngs if not os.path.isfile(p)]
        if missing_pngs:
            print(f'[test] 缺少 {len(missing_pngs)} 张可视化图片，'
                  f'调用 sample.py --visualization 补图 ...')
            cmd = build_sample_cmd(opt, has_cfg) + ['--visualization']
            print('[test] 运行: ' + ' '.join(cmd))
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        else:
            print('[test] 可视化图片完整，无需补图。')
    else:
        print('[test] --visualization 为 false，跳过可视化。')


if __name__ == '__main__':
    main()
