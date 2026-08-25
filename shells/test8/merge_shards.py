"""
把 shells/test8 的 8 个分片结果合并到原始 test.py 的保存地址。

- pt 采样: logs/sample/shard{i}/pt/** -> logs/sample/pt/** (按时间命名, 无冲突, 直接复制)
  (各分片内只保留 best 的采样; last 的采样已被脚本删除, 只留指标)
- 指标: 把同一模型在 8 个分片上的 per_sample 合并, 重新聚合 mean/std,
  写到 logs/sample/metrics/{目标模态}/{模型名}/metrics_last.json (last DDPM)
  metrics_cfg_last.json (last CFG) / metrics_best.json (best DDPM)
  metrics_cfg_best.json (best CFG); 另写 metrics.json / metrics_cfg.json 作为
  best 的默认命名, 与原 test.py 的读取地址保持一致。
"""
import glob
import json
import math
import os
import shutil

import numpy as np

REPO = '/mnt/zj-data/data/ssy/SolarCHIP'
os.chdir(REPO)

SHARD_ROOT = 'logs/sample/shard'
DST_ROOT = 'logs/sample'
TAGS = ['last', 'best']
KIND_NAMES = {
    'metrics_last.json': 'metrics_last.json',
    'metrics_best.json': 'metrics_best.json',
    'metrics_cfg_last.json': 'metrics_cfg_last.json',
    'metrics_cfg_best.json': 'metrics_cfg_best.json',
}


def finite_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def merge_metrics():
    # 收集: (target, model, fname) -> 合并后的 per_sample dict
    buckets = {}  # key -> {'per_sample': {...}, 'meta': {...}}
    for shard in range(8):
        metrics_root = f'{SHARD_ROOT}{shard}/metrics'
        for target_model in glob.glob(os.path.join(metrics_root, '*', '*')):
            if not os.path.isdir(target_model):
                continue
            target, model = os.path.relpath(target_model, metrics_root).split(os.sep)
            for fname in list(KIND_NAMES) + ['metrics.json', 'metrics_cfg.json']:
                path = os.path.join(target_model, fname)
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    data = json.load(f)
                per = data.get('per_sample', {})
                if not per:
                    continue
                key = (target, model, fname)
                if key not in buckets:
                    buckets[key] = {'per_sample': {}, 'meta': data}
                for mname, vals in per.items():
                    buckets[key]['per_sample'].setdefault(mname, {}).update(vals)

    written = 0
    report = {}
    for (target, model, fname), bucket in sorted(buckets.items()):
        meta = bucket['meta']
        merged_per = bucket['per_sample']
        metric_names = list(merged_per.keys())
        summary = {}
        for mname in metric_names:
            values = list(merged_per[mname].values())
            means = [finite_or_none(v) for v in values]
            finite_vals = [v for v in means if v is not None]
            summary[mname] = {
                'mean': (sum(finite_vals) / len(finite_vals)) if finite_vals else None,
                'std': float(np.std(finite_vals)) if len(finite_vals) > 1 else 0.0,
                'n': len(values),
            }
        merged = dict(meta)
        merged['time_interval'] = [5000, 6000]
        merged['time_step'] = 10
        merged['n_samples'] = len(next(iter(merged_per.values()))) if merged_per else 0
        merged['n_compared'] = merged['n_samples']
        merged['per_sample'] = merged_per
        merged['metrics'] = summary
        merged['merged_from_shards'] = True

        out_dir = os.path.join(DST_ROOT, 'metrics', target, model)
        os.makedirs(out_dir, exist_ok=True)
        out_name = fname
        with open(os.path.join(out_dir, out_name), 'w') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        written += 1
        report[(target, model)] = {
            fname: summary.get('mse', {}).get('mean'),
        }
    print(f'合并指标文件数: {written}')
    return report


def merge_pt():
    copied = 0
    for shard in range(8):
        pt_root = f'{SHARD_ROOT}{shard}/pt'
        if not os.path.isdir(pt_root):
            continue
        for src in glob.glob(os.path.join(pt_root, '**', '*'), recursive=True):
            if not os.path.isfile(src):
                continue
            rel = os.path.relpath(src, pt_root)
            dst = os.path.join(DST_ROOT, 'pt', rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f'合并 pt 文件数: {copied}')


if __name__ == '__main__':
    merge_pt()
    merge_metrics()
    print('完成')
