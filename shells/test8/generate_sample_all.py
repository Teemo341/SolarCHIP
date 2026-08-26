"""
生成全量采样脚本: shells/test8/sample_ckpt_*.sh (compare_transfer) 或
shells/test8/sample_ckpt_ctrl_*.sh (solarctrl)

用法:
    python generate_sample_all.py                       # compare_transfer
    python generate_sample_all.py --family solarctrl    # solarctrl

- 8 个分片: 把 [5000, 6000) 等分成 8 份, 每份一块显卡 (musa:0..7)
- 每个分片内: checkpoints/{family} 全部模型 × 全部 checkpoint 依次采样
- 输出: logs/sample/pt/{目标模态}/{模型名}/{ckpt名(去掉.ckpt)}/sample_<时间>.pt
  (按时间命名, 8 个分片写不同文件, 并发安全, 跑完即"合并"完成)
"""
import os
import glob
import re
import sys

REPO = '/mnt/zj-data/data/ssy/SolarCHIP'
os.chdir(REPO)
OUT_DIR = os.path.join(REPO, 'shells', 'test8')
os.makedirs(OUT_DIR, exist_ok=True)

FAMILY = ('solarctrl' if '--family' in sys.argv and
          sys.argv[sys.argv.index('--family') + 1] == 'solarctrl'
          else 'compare_transfer')
BASE_DIR = f'checkpoints/{FAMILY}'
FILE_PREFIX = 'sample_ckpt_ctrl' if FAMILY == 'solarctrl' else 'sample_ckpt'


def target_modal(model: str) -> str:
    if model.startswith('ctrl_best_hmi-'):
        return model[len('ctrl_best_hmi-'):]
    m = re.search(r'_(\d{4})$', model)
    if m and model.startswith('hmi_aia_'):
        return m.group(1)
    return 'hmi'


def collect():
    models = []
    for model in sorted(os.listdir(BASE_DIR)):
        mdir = os.path.join(REPO, BASE_DIR, model)
        if not os.path.isdir(mdir):
            continue
        run = sorted(os.listdir(mdir))[0]
        ckpts = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(mdir, run, 'checkpoints', '*.ckpt')))
        models.append({
            'model': model, 'run': run, 'ckpts': ckpts,
            'target': target_modal(model),
        })
    return models


MODELS = collect()
print(f'模型数: {len(MODELS)}, checkpoint 总数: {sum(len(m["ckpts"]) for m in MODELS)}')
for m in MODELS:
    print(f'  {m["model"]} ({m["target"]}): {m["ckpts"]}')

SHARD_COUNT = 8
TOTAL_START, TOTAL_END = 5000, 6000
chunk = (TOTAL_END - TOTAL_START) // SHARD_COUNT

for i in range(SHARD_COUNT):
    start = TOTAL_START + chunk * i
    end = TOTAL_START + chunk * (i + 1)
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# 本机 lightning 为源码 editable 安装, 需手动加入 PYTHONPATH')
    lines.append('export PYTHONPATH=/mnt/zj-data/data/ssy/pytorch-lightning/src${PYTHONPATH:+:$PYTHONPATH}')
    lines.append('# ============================================================')
    lines.append(f'# {FAMILY} 全量采样分片 {i}/8')
    lines.append(f'# 时间区间: [{start}, {end})  步长: 10  显卡: musa:{i}')
    lines.append(f'# 覆盖: {BASE_DIR} 全部模型 × 全部 checkpoint')
    lines.append('# 输出: logs/sample/pt/{目标模态}/{模型名}/{ckpt名}/sample_<时间>.pt')
    lines.append('# ============================================================')
    lines.append('')
    lines.append('set -u')
    lines.append('')
    lines.append(f'GPU={i}')
    lines.append(f'START={start}')
    lines.append(f'END={end}')
    lines.append('')
    lines.append('sample_ckpt() {')
    lines.append('    local model="$1" run="$2" target="$3" ckpt="$4"')
    lines.append('    local ckpt_dir="${ckpt%.ckpt}"')
    lines.append('    echo "==== [shard' + str(i) + '] ${model} / ${ckpt} ===="')
    lines.append('    python -m solarchip.main.sample \\')
    lines.append(f'        -r "{BASE_DIR}/${{model}}/${{run}}" \\')
    lines.append('        --ckpt "$ckpt" \\')
    lines.append('        --time_interval $START $END --time_step 10 \\')
    lines.append('        --seed 42 --device musa:$GPU \\')
    lines.append('        --save_root logs/sample/pt \\')
    lines.append('        --sample_subdir "$ckpt_dir" --quiet')
    lines.append('}')
    lines.append('')
    for m in MODELS:
        lines.append(f'# ---------- {m["model"]} ({m["target"]}) ----------')
        for ckpt in m['ckpts']:
            lines.append(f'sample_ckpt "{m["model"]}" "{m["run"]}" '
                         f'"{m["target"]}" "{ckpt}"')
        lines.append('')
    lines.append('echo "==== [shard' + str(i) + '] 全部完成 ===="')
    path = os.path.join(OUT_DIR, f'{FILE_PREFIX}_{i}.sh')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(path, 0o755)
    print(f'生成 {path} ({len(lines)} 行)')
