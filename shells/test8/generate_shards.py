"""
生成大规模测试的 8 个分片脚本: shells/test8/shard_0.sh ~ shard_7.sh

- 每个分片: 一个 time_interval (把 [5000, 6000) 等分成 8 份), 一块显卡 (musa:i)
- 每个分片内: checkpoints/compare_transfer + checkpoints/solarctrl 全部模型依次测试
- 每个模型: last.ckpt 与 best checkpoint 各测一次 (best = val loss 最低 / 训练保存的 epoch=*.ckpt)
- 输出: logs/sample/shard{i}/pt|metrics/... (每个分片独立, 互不覆盖)
  metrics 目录内每个模型: metrics_last.json / metrics_cfg_last.json (last 的结果)
                           metrics_best.json / metrics_cfg_best.json (best 的结果)

用法:
    python generate_shards.py             # 完整脚本 shard_0..7.sh (last+best)
    python generate_shards.py --best-only # 只补跑 best: shard_best_0..7.sh
"""
import os
import glob
import re
import sys

REPO = '/mnt/zj-data/data/ssy/SolarCHIP'
OUT_DIR = os.path.join(REPO, 'shells', 'test8')
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DIRS = ['checkpoints/compare_transfer', 'checkpoints/solarctrl']


def target_modal(model: str) -> str:
    if model.startswith('ctrl_best_hmi-'):
        return model[len('ctrl_best_hmi-'):]
    m = re.search(r'_(\d{4})$', model)
    if m and model.startswith('hmi_aia_'):
        return m.group(1)
    return 'hmi'


def collect_models():
    models = []
    for bd in BASE_DIRS:
        for model in sorted(os.listdir(os.path.join(REPO, bd))):
            mdir = os.path.join(REPO, bd, model)
            if not os.path.isdir(mdir):
                continue
            run = sorted(os.listdir(mdir))[0]
            names = [os.path.basename(p) for p in glob.glob(
                os.path.join(mdir, run, 'checkpoints', '*.ckpt'))]
            last = [n for n in names if n.startswith('last')]
            best = None
            scored = []
            for n in names:
                m = re.search(r'val_[a-z_]+=([0-9.]+)\.ckpt', n)
                if m:
                    scored.append((float(m.group(1)), n))
            if scored:
                best = min(scored)[1]
            else:
                epoch_ck = [n for n in names if n.startswith('epoch=')]
                best = epoch_ck[0] if epoch_ck else None
            models.append({
                'model': model, 'run': run, 'last': last[0] if last else '',
                'best': best or '', 'target': target_modal(model),
                'base': bd,
            })
    return models


MODELS = collect_models()
print(f'模型总数: {len(MODELS)}')

BEST_ONLY = '--best-only' in sys.argv
FILE_PREFIX = 'shard_best' if BEST_ONLY else 'shard'

# [5000, 6000) 等分成 8 份
SHARD_COUNT = 8
TOTAL_START, TOTAL_END = 5000, 6000
chunk = (TOTAL_END - TOTAL_START) // SHARD_COUNT  # 125

for i in range(SHARD_COUNT):
    start = TOTAL_START + chunk * i
    end = TOTAL_START + chunk * (i + 1)
    save_root = f'logs/sample/shard{i}'
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# ============================================================')
    if BEST_ONLY:
        lines.append(f'# SolarCHIP 大规模测试 [best-only] 分片 {i}/8')
    else:
        lines.append(f'# SolarCHIP 大规模测试分片 {i}/8')
    lines.append(f'# 时间区间: [{start}, {end})  步长: 10  显卡: musa:{i}')
    if BEST_ONLY:
        lines.append('# 覆盖: checkpoints/compare_transfer + checkpoints/solarctrl '
                     '全部模型, 只补跑 best')
    else:
        lines.append('# 覆盖: checkpoints/compare_transfer + checkpoints/solarctrl '
                     '全部模型, 每个模型 last + best 各测一次')
    lines.append(f'# 输出: {save_root}/pt|metrics/... (独立分片目录, 互不覆盖)')
    if not BEST_ONLY:
        lines.append('# 指标文件: metrics_last.json / metrics_cfg_last.json (last), '
                     'metrics_best.json / metrics_cfg_best.json (best)')
    lines.append('# ============================================================')
    lines.append('')
    lines.append('set -u')
    lines.append('')
    lines.append(f'GPU={i}')
    lines.append(f'START={start}')
    lines.append(f'END={end}')
    lines.append(f'SAVE_ROOT={save_root}')
    lines.append('')
    lines.append('test_one() {')
    lines.append('    local model="$1" run="$2" target="$3" base="$4" ckpt="$5" tag="$6"')
    lines.append('    local logdir="${base}/${model}/${run}"')
    lines.append('    local mdir="${SAVE_ROOT}/metrics/${target}/${model}"')
    lines.append('    echo "==== [shard' + str(i) + '] ${model} / ${tag} ===="')
    lines.append('    if [ -n "$ckpt" ]; then')
    lines.append('        python -m solarchip.main.test -r "$logdir" --ckpt "$ckpt" \\')
    lines.append('            --time_interval $START $END --time_step 10 \\')
    lines.append('            --metrics mse psnr ssim --visualization false \\')
    lines.append('            --device musa:$GPU --quiet --save_root "$SAVE_ROOT"')
    lines.append('    else')
    lines.append('        python -m solarchip.main.test -r "$logdir" \\')
    lines.append('            --time_interval $START $END --time_step 10 \\')
    lines.append('            --metrics mse psnr ssim --visualization false \\')
    lines.append('            --device musa:$GPU --quiet --save_root "$SAVE_ROOT"')
    lines.append('    fi')
    lines.append('    mkdir -p "$mdir"')
    lines.append('    cp "$mdir/metrics.json" "$mdir/metrics_${tag}.json"')
    lines.append('    cp "$mdir/metrics_cfg.json" "$mdir/metrics_cfg_${tag}.json"')
    lines.append('}')
    lines.append('')
    for m in MODELS:
        lines.append(f'# ---------- {m["model"]} ({m["target"]}) ----------')
        if not BEST_ONLY:
            # last
            lines.append(f'test_one "{m["model"]}" "{m["run"]}" "{m["target"]}" '
                         f'"{m["base"]}" "" "last"')
        # 删除该模型采样, 确保 best 重新采样 (通配符必须不加引号)
        lines.append('rm -f "${SAVE_ROOT}/pt/%s/%s/"sample_[0-9]*.pt' %
                     (m['target'], m['model']))
        lines.append('rm -f "${SAVE_ROOT}/pt/%s/%s/"sample_cfg_[0-9]*.pt' %
                     (m['target'], m['model']))
        # best
        lines.append(f'test_one "{m["model"]}" "{m["run"]}" "{m["target"]}" '
                     f'"{m["base"]}" "{m["best"]}" "best"')
        lines.append('')
    lines.append('echo "==== [shard' + str(i) + '] 全部完成 ===="')
    path = os.path.join(OUT_DIR, f'{FILE_PREFIX}_{i}.sh')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(path, 0o755)
    print(f'生成 {path} ({len(lines)} 行)')

print('完成')
