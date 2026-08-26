#!/usr/bin/env python3
"""
generate_vis_shards.py —— 生成 10 个按采样模态分片的可视化脚本 (shells/test9/vis_shard_<模态>.sh)。

- 每个分片: 一个 AIA 波长模态 (0094/0131/0171/0193/0211/0304/0335/1600/1700/4500),
  覆盖该模态相关的 6 个模型 (hmi->aia 3 个 + aia->hmi 3 个), 不切分 time_interval。
- 调用: sample.py --time_interval 5000 6000 --time_step 50 --visualization true
  --enhance none --save_root logs/sample/pt
- sample.py 已改为"目标 pt 存在则跳过采样", 因此已有采样直接从磁盘出图, 不动模型。
- 训练日志目录复用 shells/test8/shard_best_1.sh 里的 (模型, run) 映射。

用法:
    python shells/test9/generate_vis_shards.py
"""
import os
import re

REPO = '/mnt/zj-data/data/ssy/SolarCHIP'
SRC = os.path.join(REPO, 'shells', 'test8', 'shard_best_1.sh')
OUT_DIR = os.path.join(REPO, 'shells', 'test9')
os.makedirs(OUT_DIR, exist_ok=True)

BASE_MAP = {
    'checkpoints/compare_transfer': 'logs/compare_transfer',
    'checkpoints/solarctrl': 'logs/solarctrl',
}

LINE_RE = re.compile(r'^test_one\s+"([^"]+)"\s+"([^"]+)"\s+"([^"]+)"\s+"([^"]+)"')


def collect_models():
    """解析 shard_best_1.sh: (model, run, target, base) 列表。"""
    models = []
    with open(SRC) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                model, run, target, base = m.groups()
                if base not in BASE_MAP:
                    continue
                models.append({
                    'model': model, 'run': run, 'target': target,
                    'logdir': f'{BASE_MAP[base]}/{model}/{run}',
                })
    return models


MODELS = collect_models()
print(f'模型总数: {len(MODELS)}')

MODALS = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700', '4500']

for modal in MODALS:
    # hmi->aia: target == modal;  aia->hmi: 名字以 _<modal> 结尾 或 ctrl_best_<modal>-hmi
    part = [m for m in MODELS
            if m['target'] == modal
            or (m['target'] == 'hmi'
                and (m['model'].endswith(f'_{modal}')
                     or m['model'] == f'ctrl_best_{modal}-hmi'))]
    lines = ['#!/bin/bash',
             '# ============================================================',
             f'# SolarCHIP 已有 pt -> png 可视化分片 (模态 {modal})',
             '# 区间 [5000, 6000)  步长: 50  增强: none  只补缺失 png',
             '# 已有 pt 不重新采样 (sample.py 跳过存在文件)',
             f'# 输出: logs/sample/png/...',
             '# ============================================================',
             '', 'set -u', '',
             'vis_one() {', '    local logdir="$1"', '    echo "==== [viz] ${logdir} ===="',
             '    python -m solarchip.main.sample -r "$logdir" \\',
             '        --time_interval 5000 6000 --time_step 50 \\',
             '        --save_root logs/sample/pt --visualization true \\',
             '        --enhance none --device cpu --quiet', '}', '']
    for m in part:
        lines.append(f'# ---------- {m["model"]} ({m["target"]}) ----------')
        lines.append(f'vis_one "{m["logdir"]}"')
    lines += ['', f'echo "==== [viz] 模态 {modal} 全部完成 ===="']
    if not part:
        lines.append('echo "警告: 该模态没有模型"')
    path = os.path.join(OUT_DIR, f'vis_shard_{modal}.sh')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(path, 0o755)
    print(f'生成 {path} ({len(part)} 个模型)')

print('完成')
