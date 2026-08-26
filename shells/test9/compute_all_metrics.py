#!/usr/bin/env python3
"""
compute_all_metrics.py —— 对 logs/sample/pt/{目标模态}/{模型}/{权重} 下已有的采样结果，
计算 test.py（METRIC_REGISTRY）支持的全部指标，并把结果写入
logs/sample/metric/{目标模态}/{模型}/{权重}/metrics.json。

同时汇总 last vs best（best = epoch=*，即 val loss 最低的权重）的对比：
  - logs/sample/metric/summary_last_vs_best.json
  - logs/sample/metric/summary_last_vs_best.csv
  - logs/sample/metric/comparison_last_vs_best.md

用法:
    python shells/test9/compute_all_metrics.py
"""
import json
import math
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import torch

REPO = '/mnt/zj-data/data/ssy/SolarCHIP'
PT_ROOT = os.path.join(REPO, 'logs', 'sample', 'pt')
METRIC_ROOT = os.path.join(REPO, 'logs', 'sample', 'metric')

if REPO not in sys.path:
    sys.path.insert(0, REPO)
if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.chdir(REPO)

from solarchip.main.test import METRIC_REGISTRY, finite_or_none  # noqa: E402

ALL_METRICS = sorted(METRIC_REGISTRY)
# 方向: lower 越小越好, higher 越大越好
DIRECTIONS = {
    'mse': 'lower', 'mae': 'lower', 'mape': 'lower', 'nmse': 'lower',
    'psnr': 'higher', 'ssim': 'higher', 'pearson': 'higher', 'ccc': 'higher',
}
TIME_INTERVAL = [5000, 6000]
TIME_STEP = 10

PRED_RE = re.compile(r'^sample_(\d{8})_\d{4,6}\.pt$')
GT_RE = re.compile(r'(\d{8})_')


def discover_tasks():
    """找出所有 {目标模态}/{模型}/{权重} 采样目录。"""
    tasks = []
    for target in sorted(os.listdir(PT_ROOT)):
        tdir = os.path.join(PT_ROOT, target)
        if not os.path.isdir(tdir) or target == 'original':
            continue
        for model in sorted(os.listdir(tdir)):
            mdir = os.path.join(tdir, model)
            if not os.path.isdir(mdir):
                continue
            # 该模型的目标/时间模态: aia_hmi_xxx_M -> (source=M, target=hmi)
            #                              hmi_aia_xxx_M -> (source=hmi, target=M)
            m = re.search(r'_(\d{4})$', model)
            if model.startswith('aia_hmi_'):
                source, expect_target = m.group(1) if m else None, 'hmi'
            elif model.startswith('hmi_aia_'):
                source, expect_target = 'hmi', m.group(1) if m else None
            else:
                continue
            if source is None or expect_target != target:
                print(f'[skip] 模型目录命名与目标模态不一致: {target}/{model}')
                continue
            ckpts = []
            for d in sorted(os.listdir(mdir)):
                cdir = os.path.join(mdir, d)
                if not os.path.isdir(cdir):
                    continue
                if any(PRED_RE.match(f) for f in os.listdir(cdir)):
                    ckpts.append(d)
            if not ckpts:
                print(f'[skip] {target}/{model} 下没有带采样结果的权重目录')
                continue
            for ckpt in ckpts:
                tasks.append((target, model, source, ckpt))
    return tasks


def _init_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def compute_one(task):
    """计算单个 (模型, 权重) 的所有指标，返回报告 dict。"""
    target, model, time_modal, ckpt = task
    pred_dir = os.path.join(PT_ROOT, target, model, ckpt)
    orig_dir = os.path.join(PT_ROOT, target, 'original')

    # 真实数据: 日期串 -> 文件名 (aia: AIA20240108_0000_0094.pt / hmi: hmi.M_720s.20240108_*_TAI.pt)
    gt_by_date = {}
    for f in sorted(os.listdir(orig_dir)):
        mm = GT_RE.search(f)
        if mm:
            gt_by_date[mm.group(1)] = f

    per_sample = {name: {} for name in ALL_METRICS}
    skipped_gt = skipped_pred = 0
    file_list = sorted(f for f in os.listdir(pred_dir) if PRED_RE.match(f))
    for f in file_list:
        date = PRED_RE.match(f).group(1)
        if date not in gt_by_date:
            skipped_gt += 1
            continue
        pred_path = os.path.join(pred_dir, f)
        gt_path = os.path.join(orig_dir, gt_by_date[date])
        try:
            pred = torch.load(pred_path, weights_only=True)
            gt = torch.load(gt_path, weights_only=True)
        except Exception as e:  # noqa: BLE001
            skipped_pred += 1
            print(f'[warn] 加载失败 {pred_path}: {e}')
            continue
        if not hasattr(pred, 'shape') or not hasattr(gt, 'shape'):
            skipped_pred += 1
            continue
        for name in ALL_METRICS:
            per_sample[name][date] = METRIC_REGISTRY[name](pred, gt)

    summary = {}
    for name in ALL_METRICS:
        values = list(per_sample[name].values())
        summary[name] = {
            'mean': finite_or_none(float(np.mean(values))) if values else None,
            'std': finite_or_none(float(np.std(values))) if values else None,
            'n': len(values),
        }

    report = {
        'model_name': model,
        'target_modal': target,
        'time_modal': time_modal,
        'ckpt_name': ckpt,
        'sample_kind': 'sample',
        'time_interval': TIME_INTERVAL,
        'time_step': TIME_STEP,
        'n_samples': len(file_list),
        'n_compared': len(next(iter(per_sample.values()))) if per_sample else 0,
        'n_skipped_missing_gt': skipped_gt,
        'n_skipped_missing_pred': skipped_pred,
        'metrics': summary,
        'per_sample': per_sample,
    }
    return report


def is_better(a, b, metric):
    """a 是否比 b 好 (按指标方向)。返回 True / False (a<=b 平局时 False)。"""
    if a is None or b is None:
        return False
    if DIRECTIONS[metric] == 'lower':
        return a < b - 1e-12
    return a > b + 1e-12


def build_comparison(reports):
    """按模型汇总 last vs best，并生成汇总统计。"""
    by_model = {}
    for r in reports:
        key = (r['target_modal'], r['model_name'])
        by_model.setdefault(key, {})[r['ckpt_name']] = r

    rows = []
    for (target, model), by_ckpt in sorted(by_model.items()):
        last = next((r for ck, r in by_ckpt.items() if ck == 'last'), None)
        best = next((r for ck, r in by_ckpt.items() if ck != 'last'), None)
        if best is None and len(by_ckpt) == 1:
            best = next(iter(by_ckpt.values()))
        row = {'model_name': model, 'target_modal': target, 'ckpt_best': None}
        winners = {}
        for name in ALL_METRICS:
            lv = last['metrics'][name]['mean'] if last else None
            bv = best['metrics'][name]['mean'] if best else None
            row[f'last_{name}'] = lv
            row[f'best_{name}'] = bv
            if lv is None or bv is None:
                winners[name] = 'n/a'
            elif abs(lv - bv) <= 1e-12:
                winners[name] = 'tie'
            elif is_better(lv, bv, name):
                winners[name] = 'last'
            else:
                winners[name] = 'best'
            row[f'winner_{name}'] = winners[name]
        rows.append(row)

    # 总体统计
    per_metric = {}
    for name in ALL_METRICS:
        wins = {'last': 0, 'best': 0, 'tie': 0, 'n/a': 0}
        last_vals, best_vals = [], []
        for row in rows:
            w = row[f'winner_{name}']
            wins[w] += 1
            if row[f'last_{name}'] is not None and row[f'best_{name}'] is not None:
                last_vals.append(row[f'last_{name}'])
                best_vals.append(row[f'best_{name}'])
        per_metric[name] = {
            'direction': DIRECTIONS[name],
            'n_last_win': wins['last'],
            'n_best_win': wins['best'],
            'n_tie': wins['tie'],
            'n_na': wins['n/a'],
            'mean_last': float(np.mean(last_vals)) if last_vals else None,
            'mean_best': float(np.mean(best_vals)) if best_vals else None,
        }
    total_cells = len(rows) * len(ALL_METRICS)
    total_wins = {'last': 0, 'best': 0, 'tie': 0, 'n/a': 0}
    model_wins = {'last': 0, 'best': 0, 'tie': 0}
    for row in rows:
        c = {'last': 0, 'best': 0, 'tie': 0}
        for name in ALL_METRICS:
            w = row[f'winner_{name}']
            total_wins[w] = total_wins.get(w, 0) + 1
            if w in c:
                c[w] += 1
        if c['last'] > c['best']:
            model_wins['last'] += 1
        elif c['best'] > c['last']:
            model_wins['best'] += 1
        else:
            model_wins['tie'] += 1
    overall = {
        'total_cells': total_cells,
        'total_wins': total_wins,
        'model_level': model_wins,
        'n_models': len(rows),
    }
    return rows, per_metric, overall


def main():
    os.makedirs(METRIC_ROOT, exist_ok=True)
    tasks = discover_tasks()
    print(f'共 {len(tasks)} 个 (模型, 权重) 任务')
    if not tasks:
        return
    with Pool(processes=24, initializer=_init_worker) as pool:
        reports = pool.map(compute_one, tasks)

    # 写每个权重的指标 JSON
    for r in reports:
        out_dir = os.path.join(METRIC_ROOT, r['target_modal'], r['model_name'], r['ckpt_name'])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f'[save] {r["target_modal"]}/{r["model_name"]}/{r["ckpt_name"]} '
              f'n={r["n_compared"]} -> {path}')

    # last vs best 对比
    rows, per_metric, overall = build_comparison(reports)
    summary = {'per_metric': per_metric, 'overall': overall, 'per_model': rows}
    with open(os.path.join(METRIC_ROOT, 'summary_last_vs_best.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    import csv
    with open(os.path.join(METRIC_ROOT, 'summary_last_vs_best.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['model_name', 'target_modal']
        for name in ALL_METRICS:
            header += [f'last_{name}', f'best_{name}', f'winner_{name}']
        w.writerow(header)
        for row in rows:
            line = [row['model_name'], row['target_modal']]
            for name in ALL_METRICS:
                line += [row[f'last_{name}'], row[f'best_{name}'], row[f'winner_{name}']]
            w.writerow(line)

    # Markdown 可读版
    lines = ['# last vs best 对比 (all metrics)', '',
             f'- 指标: {", ".join(ALL_METRICS)}',
             f'- last = 最后一轮权重; best = epoch=* (val loss 最低) 权重',
             f'- lower 更好: {[m for m, d in DIRECTIONS.items() if d == "lower"]}',
             f'- higher 更好: {[m for m, d in DIRECTIONS.items() if d == "higher"]}', '']
    lines += ['## 总体', '']
    for name, s in per_metric.items():
        lines.append(f'- {name} ({s["direction"]}): last 赢 {s["n_last_win"]} 个模型, '
                     f'best 赢 {s["n_best_win"]} 个模型, 平局 {s["n_tie"]}; '
                     f'平均 last={s["mean_last"]:.4g} vs best={s["mean_best"]:.4g}')
    lines.append(f'\n- 全局: 总对比单元 {overall["total_cells"]} 个, '
                 f'last 赢 {overall["total_wins"]["last"]}, best 赢 {overall["total_wins"]["best"]}, '
                 f'平局 {overall["total_wins"]["tie"]}')
    lines.append(f'- 模型层面: {overall["n_models"]} 个模型中, '
                 f'last 多数指标更好: {overall["model_level"]["last"]}, '
                 f'best 多数指标更好: {overall["model_level"]["best"]}, '
                 f'平局: {overall["model_level"]["tie"]}')
    lines += ['', '## 每个配置', '', '| 目标模态 | 模型 | ' +
              ' | '.join(ALL_METRICS) + ' | 综合 |', '|:--|:--|' + '|:--:|' * len(ALL_METRICS) + '|:--:|']
    for row in rows:
        cells = []
        for name in ALL_METRICS:
            w = row[f'winner_{name}']
            cells.append({'last': 'last', 'best': 'best', 'tie': '=', 'n/a': '-'}[w])
        c = {'last': 0, 'best': 0, 'tie': 0}
        for name in ALL_METRICS:
            w = row[f'winner_{name}']
            if w in c:
                c[w] += 1
        verdict = 'last' if c['last'] > c['best'] else ('best' if c['best'] > c['last'] else '=')
        lines.append(f'| {row["target_modal"]} | {row["model_name"]} | ' +
                     ' | '.join(cells) + f' | {verdict} |')
    md = '\n'.join(lines) + '\n'
    with open(os.path.join(METRIC_ROOT, 'comparison_last_vs_best.md'), 'w') as f:
        f.write(md)
    print(f'[save] 汇总: {os.path.join(METRIC_ROOT, "summary_last_vs_best.json")}')
    print(md)


if __name__ == '__main__':
    main()
