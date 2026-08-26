#!/usr/bin/env python3
"""
compute_ctrl_metrics.py —— 对 logs/sample/pt/{目标模态}/ctrl_best_*/{权重} 下的
SolarControl 采样结果(正常采样 sample_*.pt + CFG 采样 sample_cfg_*.pt)计算
test.py(METRIC_REGISTRY)支持的全部指标。

输出:
  - logs/sample/metric/{目标模态}/{模型}/{权重}/metrics.json      (正常采样)
  - logs/sample/metric/{目标模态}/{模型}/{权重}/metrics_cfg.json  (CFG 采样)
  - logs/sample/metric/ctrl_ckpt_comparison.{json,csv,md}         (last/best1-3 + 正常vs CFG 汇总)

权重抽象: 3 个 epoch=*_val_loss_simple=* 按 val_loss_simple 升序映射为 best1/best2/best3
(loss 相同则按 epoch 升序); last 即最后一轮权重。

用法:
    python shells/test9/compute_ctrl_metrics.py
"""
import json
import math
import os
import re
import sys
import csv
from collections import Counter, defaultdict
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
DIRECTIONS = {
    'mse': 'lower', 'mae': 'lower', 'mape': 'lower', 'nmse': 'lower',
    'psnr': 'higher', 'ssim': 'higher', 'pearson': 'higher', 'ccc': 'higher',
}
TIME_INTERVAL = [5000, 6000]
TIME_STEP = 10

PRED_RE = re.compile(r'^(sample(?:_cfg)?)_(\d{8})_\d{4,6}\.pt$')
GT_RE = re.compile(r'(\d{8})_')
EPOCH_RE = re.compile(r'^epoch=(\d+)_val_loss_simple=([0-9.]+)$')


def discover_tasks():
    """返回 (target, model, time_modal, ckpt) 列表 —— 仅 ctrl_best_* 模型。"""
    tasks = []
    for modal in sorted(os.listdir(PT_ROOT)):
        mdir = os.path.join(PT_ROOT, modal)
        if not os.path.isdir(mdir):
            continue
        for model in sorted(os.listdir(mdir)):
            if not model.startswith('ctrl_best_'):
                continue
            d = os.path.join(mdir, model)
            if not os.path.isdir(d):
                continue
            m = re.match(r'^ctrl_best_(hmi-(\d{4})|(\d{4})-hmi)$', model)
            if m is None:
                continue
            if m.group(2):          # ctrl_best_hmi-0094 -> hmi -> 0094
                time_modal, target = 'hmi', m.group(2)
            else:                   # ctrl_best_0094-hmi -> 0094 -> hmi
                time_modal, target = m.group(3), 'hmi'
            if target != modal:
                continue
            for sub in sorted(os.listdir(d)):
                p = os.path.join(d, sub)
                if not os.path.isdir(p):
                    continue
                if any(PRED_RE.match(f) for f in os.listdir(p)):
                    tasks.append((target, model, time_modal, sub))
    return tasks


def _init_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def compute_one(task):
    """计算单个 (模型, 权重) 的正常/CFG 指标, 返回两个 report。"""
    target, model, time_modal, ckpt = task
    pred_dir = os.path.join(PT_ROOT, target, model, ckpt)
    orig_dir = os.path.join(PT_ROOT, target, 'original')

    gt_by_date = {}
    for f in sorted(os.listdir(orig_dir)):
        mm = GT_RE.search(f)
        if mm:
            gt_by_date[mm.group(1)] = f

    reports = {}
    for kind in ('sample', 'sample_cfg'):
        prefix = kind  # sample_* / sample_cfg_*
        files = sorted(
            f for f in os.listdir(pred_dir)
            if PRED_RE.match(f) and PRED_RE.match(f).group(1) == prefix)
        per_sample = {name: {} for name in ALL_METRICS}
        skipped_gt = skipped_pred = 0
        for f in files:
            date = PRED_RE.match(f).group(2)
            if date not in gt_by_date:
                skipped_gt += 1
                continue
            try:
                pred = torch.load(os.path.join(pred_dir, f), weights_only=True)
                gt = torch.load(os.path.join(orig_dir, gt_by_date[date]), weights_only=True)
            except Exception as e:  # noqa: BLE001
                skipped_pred += 1
                print(f'[warn] 加载失败 {pred_dir}/{f}: {e}')
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
        reports[kind] = {
            'model_name': model,
            'target_modal': target,
            'time_modal': time_modal,
            'ckpt_name': ckpt,
            'sample_kind': kind,
            'time_interval': TIME_INTERVAL,
            'time_step': TIME_STEP,
            'n_samples': len(files),
            'n_compared': len(next(iter(per_sample.values()))) if per_sample else 0,
            'n_skipped_missing_gt': skipped_gt,
            'n_skipped_missing_pred': skipped_pred,
            'metrics': summary,
            'per_sample': per_sample,
        }
    return reports


def ckpt_key(ckpt):
    """权重排序键: last 排最后; epoch 按 (val_loss_simple, epoch) 升序。"""
    if ckpt == 'last':
        return (1, 0.0, 0)
    m = EPOCH_RE.match(ckpt)
    if m:
        return (0, float(m.group(2)), int(m.group(1)))
    return (2, 0.0, 0)


def is_better(a, b, metric):
    if a is None or b is None:
        return False
    if DIRECTIONS[metric] == 'lower':
        return a < b - 1e-12
    return a > b + 1e-12


def load_reports():
    """从已保存的 metrics.json / metrics_cfg.json 重建 reports (--summary-only)。"""
    reports = {}
    for target in sorted(os.listdir(METRIC_ROOT)):
        tdir = os.path.join(METRIC_ROOT, target)
        if not os.path.isdir(tdir):
            continue
        for model in sorted(os.listdir(tdir)):
            if not model.startswith('ctrl_best_'):
                continue
            mdir = os.path.join(tdir, model)
            if not os.path.isdir(mdir):
                continue
            for ckpt in sorted(os.listdir(mdir)):
                cdir = os.path.join(mdir, ckpt)
                if not os.path.isdir(cdir):
                    continue
                for fname, kind in (('metrics.json', 'sample'),
                                    ('metrics_cfg.json', 'sample_cfg')):
                    p = os.path.join(cdir, fname)
                    if os.path.isfile(p):
                        with open(p) as f:
                            r = json.load(f)
                        reports[(r['target_modal'], r['model_name'],
                                 r['ckpt_name'], kind)] = r
    return reports


MET = ['mse', 'mae', 'mape', 'nmse', 'psnr', 'ssim', 'pearson', 'ccc']
WIN_CHAR = {'last': 'L', 'best1': '1', 'best2': '2', 'best3': '3',
            'epoch1': '1', 'epoch2': '2', 'epoch3': '3'}
CHAR_LABEL = {'L': 'last', '1': 'best1', '2': 'best2', '3': 'best3'}


def winner_metrics(values_by_lab):
    """values_by_lab: {label: report} 每指标取最优的权重, 返回 (序列字符串, Counter)。"""
    seq = []
    counts = Counter()
    for name in MET:
        valid = {lab: rep['metrics'][name]['mean'] for lab, rep in values_by_lab.items()
                 if rep['metrics'][name]['mean'] is not None}
        if not valid:
            seq.append('-')
            continue
        best_lab = (min(valid, key=lambda k: valid[k]) if DIRECTIONS[name] == 'lower'
                    else max(valid, key=lambda k: valid[k]))
        seq.append(WIN_CHAR.get(best_lab, '?'))
        counts[best_lab] += 1
    return ''.join(seq), counts


def main():
    summary_only = '--summary-only' in sys.argv
    if summary_only:
        reports = load_reports()
        print(f'[summary] 已加载 {len(reports)} 份指标文件')
    else:
        os.makedirs(METRIC_ROOT, exist_ok=True)
        tasks = discover_tasks()
        print(f'共 {len(tasks)} 个 (ctrl模型, 权重) 任务')
        if not tasks:
            return
        with Pool(processes=24, initializer=_init_worker) as pool:
            results = pool.map(compute_one, tasks)

        reports = {}
        for task, rec in zip(tasks, results):
            target, model, time_modal, ckpt = task
            for kind, r in rec.items():
                reports[(target, model, ckpt, kind)] = r
                out_dir = os.path.join(METRIC_ROOT, target, model, ckpt)
                os.makedirs(out_dir, exist_ok=True)
                fname = 'metrics.json' if kind == 'sample' else 'metrics_cfg.json'
                path = os.path.join(out_dir, fname)
                with open(path, 'w') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
                print(f'[save] {target}/{model}/{ckpt} [{kind}] n={r["n_compared"]} -> {path}')

    # ------- 汇总: 双对比维度 -------
    # 维度 A: best1/best2/best3 (val_loss_simple 升序) + last
    # 维度 B: epoch1/epoch2/epoch3 (epoch 升序) + last
    by_model = defaultdict(dict)
    for (target, model, ckpt, kind), r in reports.items():
        by_model[(target, model)][(ckpt, kind)] = r

    def best_order(ckpts):
        return sorted([c for c in ckpts if c != 'last'], key=ckpt_key)

    def epoch_order(ckpts):
        return sorted([c for c in ckpts if c != 'last'],
                      key=lambda c: int(EPOCH_RE.match(c).group(1)))

    label_maps = {}
    for (target, model), rec in by_model.items():
        ckpts = sorted({ck for ck, _ in rec.keys()}, key=ckpt_key)
        best_labs = {'last': 'last',
                     **{c: f'best{i+1}' for i, c in enumerate(best_order(ckpts))}}
        epoch_labs = {'last': 'last',
                      **{c: f'epoch{i+1}' for i, c in enumerate(epoch_order(ckpts))}}
        label_maps[(target, model)] = (best_labs, epoch_labs)

    rows = []
    for (target, model) in sorted(by_model.keys()):
        rec = by_model[(target, model)]
        best_labs, epoch_labs = label_maps[(target, model)]
        cks = sorted({c for c, _ in rec.keys()}, key=ckpt_key)
        row = {'target_modal': target, 'model_name': model}
        # 值列 (best 序, 同 epoch 序数值只是标签不同)
        for kind in ('sample', 'sample_cfg'):
            for ck in cks:
                lab = best_labs[ck]
                r = rec.get((ck, kind))
                for name in MET:
                    row[f'{kind}_{lab}_{name}'] = (r['metrics'][name]['mean']
                                                   if r else None)
        # 两个序的每指标 winner + 模型级 winner
        for kind in ('sample', 'sample_cfg'):
            bvals = {best_labs[c]: rec[(c, kind)] for c in cks if (c, kind) in rec}
            evals = {epoch_labs[c]: rec[(c, kind)] for c in cks if (c, kind) in rec}
            wseq, wc = winner_metrics(bvals)
            eseq, ec = winner_metrics(evals)
            row[f'{kind}_winner_metrics'] = wseq
            row[f'{kind}_winner'] = wc.most_common(1)[0][0] if wc else 'n/a'
            row[f'{kind}_winner_details'] = dict(wc)
            row[f'{kind}_epoch_winner_metrics'] = eseq
            row[f'{kind}_epoch_winner'] = ec.most_common(1)[0][0] if ec else 'n/a'
            row[f'{kind}_epoch_winner_details'] = dict(ec)
        # 正常 vs cfg: 每指标 winner (以 last 权重)
        cfg_win = {}
        cfg_counts = Counter()
        for name in MET:
            lv = row.get(f'sample_last_{name}')
            cv = row.get(f'sample_cfg_last_{name}')
            if lv is None or cv is None:
                cfg_win[name] = 'n/a'
                continue
            if abs(lv - cv) <= 1e-12:
                w = '='
            else:
                w = 'normal' if is_better(lv, cv, name) else 'cfg'
            cfg_win[name] = w
            if w in ('normal', 'cfg'):
                cfg_counts[w] += 1
        row['normal_vs_cfg_metrics'] = ''.join(
            {'normal': 'N', 'cfg': 'C', '=': '=', 'n/a': '-'}[cfg_win[m]] for m in MET)
        row['normal_vs_cfg'] = (cfg_counts.most_common(1)[0][0] if cfg_counts else 'n/a')
        row['normal_vs_cfg_details'] = dict(cfg_counts)
        rows.append(row)

    # 总体统计
    per_metric = {}
    for name in MET:
        wins = Counter()
        ewins = Counter()
        vals = defaultdict(list)
        for r in rows:
            for kind in ('sample', 'sample_cfg'):
                for lab in ('last', 'best1', 'best2', 'best3'):
                    v = r.get(f'{kind}_{lab}_{name}')
                    if v is not None:
                        vals[lab].append(v)
                w = r[f'{kind}_winner_metrics'][MET.index(name)]
                if w in ('L', '1', '2', '3'):
                    wins[CHAR_LABEL[w]] += 1
                ew = r[f'{kind}_epoch_winner_metrics'][MET.index(name)]
                if ew in ('L', '1', '2', '3'):
                    ewins[{'L': 'last', '1': 'epoch1', '2': 'epoch2', '3': 'epoch3'}[ew]] += 1
        per_metric[name] = {
            'direction': DIRECTIONS[name],
            'wins': dict(wins),
            'epoch_wins': dict(ewins),
            'mean_last': float(np.mean(vals['last'])) if vals['last'] else None,
            'mean_best1': float(np.mean(vals['best1'])) if vals['best1'] else None,
            'mean_best2': float(np.mean(vals['best2'])) if vals['best2'] else None,
            'mean_best3': float(np.mean(vals['best3'])) if vals['best3'] else None,
        }
    total = Counter()
    etotal = Counter()
    model_level = Counter()
    emodel_level = Counter()
    for r in rows:
        for kind in ('sample', 'sample_cfg'):
            for ch in r[f'{kind}_winner_metrics']:
                if ch in 'L123':
                    total[CHAR_LABEL[ch]] += 1
            for ch in r[f'{kind}_epoch_winner_metrics']:
                if ch in 'L123':
                    etotal[{'L': 'last', '1': 'epoch1', '2': 'epoch2', '3': 'epoch3'}[ch]] += 1
            model_level[r[f'{kind}_winner']] += 1
            emodel_level[r[f'{kind}_epoch_winner']] += 1
    # normal vs cfg 总体
    nvc = Counter()
    nvc_metric = {}
    for name in MET:
        c = Counter()
        for r in rows:
            w = r['normal_vs_cfg_metrics'][MET.index(name)]
            if w in ('N', 'C', '='):
                c[{'N': 'normal', 'C': 'cfg', '=': 'tie'}[w]] += 1
        nvc_metric[name] = dict(c)
        nvc.update(c)
    overall = {
        'n_models': len(rows),
        'n_cells': len(rows) * 2 * len(MET),
        'wins_by_ckpt': dict(total),
        'epoch_wins_by_ckpt': dict(etotal),
        'model_kind_wins': dict(model_level),
        'epoch_model_kind_wins': dict(emodel_level),
        'normal_vs_cfg': dict(nvc),
        'normal_vs_cfg_by_metric': nvc_metric,
        'label_maps': {f'{k[0]}/{k[1]}': {'best_order': v[0], 'epoch_order': v[1]}
                       for k, v in label_maps.items()},
    }

    with open(os.path.join(METRIC_ROOT, 'ctrl_ckpt_comparison.json'), 'w') as f:
        json.dump({'per_metric': per_metric, 'overall': overall, 'per_model': rows},
                  f, indent=2, ensure_ascii=False)

    with open(os.path.join(METRIC_ROOT, 'ctrl_ckpt_comparison.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['target_modal', 'model_name']
        for kind in ('sample', 'sample_cfg'):
            for lab in ('last', 'best1', 'best2', 'best3'):
                header += [f'{kind}_{lab}_{m}' for m in MET]
            header += [f'{kind}_winner', f'{kind}_epoch_winner']
        header += ['normal_vs_cfg']
        w.writerow(header)
        for r in rows:
            line = [r['target_modal'], r['model_name']]
            for kind in ('sample', 'sample_cfg'):
                for lab in ('last', 'best1', 'best2', 'best3'):
                    line += [r.get(f'{kind}_{lab}_{m}') for m in MET]
                line += [r[f'{kind}_winner'], r[f'{kind}_epoch_winner']]
            line += [r['normal_vs_cfg']]
            w.writerow(line)

    # markdown
    lines = ['# SolarControl: last vs best1/best2/best3 对比 (正常采样 + CFG 采样)', '',
             '- 指标: ' + ', '.join(MET),
             '- 维度A best1/best2/best3 = 3 个 epoch=*_val_loss_simple=* 权重, 按 val_loss_simple 升序',
             '- 维度B epoch1/epoch2/epoch3 = 同一批权重, 按 epoch 升序; last = 最后一轮权重',
             '- winner 序列: 8 个字符依次 = mse mae mape nmse psnr ssim pearson ccc (L=last, 1/2/3=对应序的1/2/3)',
             '- 正常采样 vs CFG 采样: 以 last 权重比较 (N=正常更好, C=CFG更好, ==平局)', '']
    lines += ['## 总体 (维度A: best1-3 / 维度B: epoch1-3)', '']
    lines.append(f'- 总对比单元 {overall["n_cells"]} 个 (20 模型 × 2 采样方式 × 8 指标)')
    lines.append('- best序: ' + ', '.join(f'{k} 赢 {v}' for k, v in sorted(total.items())))
    lines.append('- epoch序: ' + ', '.join(f'{k} 赢 {v}' for k, v in sorted(etotal.items())))
    lines.append('- 模型×采样方式 (best序): ' + ', '.join(f'{k} 赢 {v}' for k, v in sorted(model_level.items())))
    lines.append('- 模型×采样方式 (epoch序): ' + ', '.join(f'{k} 赢 {v}' for k, v in sorted(emodel_level.items())))
    lines.append('- 正常 vs CFG (以 last 权重): ' + ', '.join(f'{k} 赢 {v}' for k, v in sorted(nvc.items())) + f' (共 {sum(nvc.values())} 个对比单元)')
    lines += ['', '## 每指标 (best序/epoch序 赢的单元数)', '',
              '| 指标 | best序赢 | epoch序赢 | mean last | b1 | b2 | b3 |',
              '|:--|:--|:--|:--|:--|:--|:--|']
    for name in MET:
        s = per_metric[name]
        lines.append(f'| {name} | {s["wins"]} | {s["epoch_wins"]} | {s["mean_last"]:.4g} | '
                     f'{s["mean_best1"]:.4g} | {s["mean_best2"]:.4g} | {s["mean_best3"]:.4g} |')
    lines += ['', '## 每个配置', '',
              '| 目标 | 模型 | 正常winner(best序) | 正常最优 | CFGwinner(best序) | CFG最优 | 正常(vsCFG) | '
              '正常winner(epoch序) | 正常最优 | CFGwinner(epoch序) | CFG最优 |',
              '|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|']
    for r in rows:
        lines.append(f'| {r["target_modal"]} | {r["model_name"]} | {r["sample_winner_metrics"]} | {r["sample_winner"]} | '
                     f'{r["sample_cfg_winner_metrics"]} | {r["sample_cfg_winner"]} | {r["normal_vs_cfg_metrics"]} | '
                     f'{r["sample_epoch_winner_metrics"]} | {r["sample_epoch_winner"]} | '
                     f'{r["sample_cfg_epoch_winner_metrics"]} | {r["sample_cfg_epoch_winner"]} |')
    md = '\n'.join(lines) + '\n'
    with open(os.path.join(METRIC_ROOT, 'ctrl_ckpt_comparison.md'), 'w') as f:
        f.write(md)
    print(f'[save] 汇总: {os.path.join(METRIC_ROOT, "ctrl_ckpt_comparison.json")}')
    print(md)


if __name__ == '__main__':
    main()
