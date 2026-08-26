#!/bin/bash
# 本机 lightning 为源码 editable 安装, 需手动加入 PYTHONPATH
export PYTHONPATH=/mnt/zj-data/data/ssy/pytorch-lightning/src${PYTHONPATH:+:$PYTHONPATH}
# ============================================================
# solarctrl 全量采样分片 1/8
# 时间区间: [5125, 5250)  步长: 10  显卡: musa:1
# 覆盖: checkpoints/solarctrl 全部模型 × 全部 checkpoint
# 输出: logs/sample/pt/{目标模态}/{模型名}/{ckpt名}/sample_<时间>.pt
# ============================================================

set -u

GPU=1
START=5125
END=5250

sample_ckpt() {
    local model="$1" run="$2" target="$3" ckpt="$4"
    local ckpt_dir="${ckpt%.ckpt}"
    echo "==== [shard1] ${model} / ${ckpt} ===="
    python -m solarchip.main.sample \
        -r "checkpoints/solarctrl/${model}/${run}" \
        --ckpt "$ckpt" \
        --time_interval $START $END --time_step 10 \
        --seed 42 --device musa:$GPU \
        --save_root logs/sample/pt \
        --sample_subdir "$ckpt_dir" --quiet
}

# ---------- ctrl_best_0094-hmi (hmi) ----------
sample_ckpt "ctrl_best_0094-hmi" "2026-08-07T17-58-09" "hmi" "epoch=000198_val_loss_simple=0.0385.ckpt"
sample_ckpt "ctrl_best_0094-hmi" "2026-08-07T17-58-09" "hmi" "epoch=000563_val_loss_simple=0.0355.ckpt"
sample_ckpt "ctrl_best_0094-hmi" "2026-08-07T17-58-09" "hmi" "epoch=000633_val_loss_simple=0.0377.ckpt"
sample_ckpt "ctrl_best_0094-hmi" "2026-08-07T17-58-09" "hmi" "last.ckpt"

# ---------- ctrl_best_0131-hmi (hmi) ----------
sample_ckpt "ctrl_best_0131-hmi" "2026-08-07T18-00-34" "hmi" "epoch=000198_val_loss_simple=0.0385.ckpt"
sample_ckpt "ctrl_best_0131-hmi" "2026-08-07T18-00-34" "hmi" "epoch=000563_val_loss_simple=0.0354.ckpt"
sample_ckpt "ctrl_best_0131-hmi" "2026-08-07T18-00-34" "hmi" "epoch=000633_val_loss_simple=0.0375.ckpt"
sample_ckpt "ctrl_best_0131-hmi" "2026-08-07T18-00-34" "hmi" "last.ckpt"

# ---------- ctrl_best_0171-hmi (hmi) ----------
sample_ckpt "ctrl_best_0171-hmi" "2026-08-07T18-03-05" "hmi" "epoch=000198_val_loss_simple=0.0385.ckpt"
sample_ckpt "ctrl_best_0171-hmi" "2026-08-07T18-03-05" "hmi" "epoch=000563_val_loss_simple=0.0354.ckpt"
sample_ckpt "ctrl_best_0171-hmi" "2026-08-07T18-03-05" "hmi" "epoch=000633_val_loss_simple=0.0376.ckpt"
sample_ckpt "ctrl_best_0171-hmi" "2026-08-07T18-03-05" "hmi" "last.ckpt"

# ---------- ctrl_best_0193-hmi (hmi) ----------
sample_ckpt "ctrl_best_0193-hmi" "2026-08-07T18-05-40" "hmi" "epoch=000198_val_loss_simple=0.0385.ckpt"
sample_ckpt "ctrl_best_0193-hmi" "2026-08-07T18-05-40" "hmi" "epoch=000563_val_loss_simple=0.0355.ckpt"
sample_ckpt "ctrl_best_0193-hmi" "2026-08-07T18-05-40" "hmi" "epoch=000633_val_loss_simple=0.0376.ckpt"
sample_ckpt "ctrl_best_0193-hmi" "2026-08-07T18-05-40" "hmi" "last.ckpt"

# ---------- ctrl_best_0211-hmi (hmi) ----------
sample_ckpt "ctrl_best_0211-hmi" "2026-08-07T18-08-12" "hmi" "epoch=000101_val_loss_simple=0.0392.ckpt"
sample_ckpt "ctrl_best_0211-hmi" "2026-08-07T18-08-12" "hmi" "epoch=000563_val_loss_simple=0.0362.ckpt"
sample_ckpt "ctrl_best_0211-hmi" "2026-08-07T18-08-12" "hmi" "epoch=000633_val_loss_simple=0.0365.ckpt"
sample_ckpt "ctrl_best_0211-hmi" "2026-08-07T18-08-12" "hmi" "last.ckpt"

# ---------- ctrl_best_0304-hmi (hmi) ----------
sample_ckpt "ctrl_best_0304-hmi" "2026-08-07T18-10-40" "hmi" "epoch=000198_val_loss_simple=0.0384.ckpt"
sample_ckpt "ctrl_best_0304-hmi" "2026-08-07T18-10-40" "hmi" "epoch=000563_val_loss_simple=0.0353.ckpt"
sample_ckpt "ctrl_best_0304-hmi" "2026-08-07T18-10-40" "hmi" "epoch=000633_val_loss_simple=0.0374.ckpt"
sample_ckpt "ctrl_best_0304-hmi" "2026-08-07T18-10-40" "hmi" "last.ckpt"

# ---------- ctrl_best_0335-hmi (hmi) ----------
sample_ckpt "ctrl_best_0335-hmi" "2026-08-07T18-13-12" "hmi" "epoch=000198_val_loss_simple=0.0385.ckpt"
sample_ckpt "ctrl_best_0335-hmi" "2026-08-07T18-13-12" "hmi" "epoch=000563_val_loss_simple=0.0355.ckpt"
sample_ckpt "ctrl_best_0335-hmi" "2026-08-07T18-13-12" "hmi" "epoch=000633_val_loss_simple=0.0376.ckpt"
sample_ckpt "ctrl_best_0335-hmi" "2026-08-07T18-13-12" "hmi" "last.ckpt"

# ---------- ctrl_best_1600-hmi (hmi) ----------
sample_ckpt "ctrl_best_1600-hmi" "2026-08-07T18-15-34" "hmi" "epoch=000554_val_loss_simple=0.0390.ckpt"
sample_ckpt "ctrl_best_1600-hmi" "2026-08-07T18-15-34" "hmi" "epoch=000563_val_loss_simple=0.0362.ckpt"
sample_ckpt "ctrl_best_1600-hmi" "2026-08-07T18-15-34" "hmi" "epoch=000633_val_loss_simple=0.0370.ckpt"
sample_ckpt "ctrl_best_1600-hmi" "2026-08-07T18-15-34" "hmi" "last.ckpt"

# ---------- ctrl_best_1700-hmi (hmi) ----------
sample_ckpt "ctrl_best_1700-hmi" "2026-08-08T10-00-51" "hmi" "epoch=000085_val_loss_simple=0.0387.ckpt"
sample_ckpt "ctrl_best_1700-hmi" "2026-08-08T10-00-51" "hmi" "epoch=000563_val_loss_simple=0.0354.ckpt"
sample_ckpt "ctrl_best_1700-hmi" "2026-08-08T10-00-51" "hmi" "epoch=000633_val_loss_simple=0.0376.ckpt"
sample_ckpt "ctrl_best_1700-hmi" "2026-08-08T10-00-51" "hmi" "last.ckpt"

# ---------- ctrl_best_4500-hmi (hmi) ----------
sample_ckpt "ctrl_best_4500-hmi" "2026-08-08T10-03-20" "hmi" "epoch=000085_val_loss_simple=0.0391.ckpt"
sample_ckpt "ctrl_best_4500-hmi" "2026-08-08T10-03-20" "hmi" "epoch=000563_val_loss_simple=0.0360.ckpt"
sample_ckpt "ctrl_best_4500-hmi" "2026-08-08T10-03-20" "hmi" "epoch=000633_val_loss_simple=0.0383.ckpt"
sample_ckpt "ctrl_best_4500-hmi" "2026-08-08T10-03-20" "hmi" "last.ckpt"

# ---------- ctrl_best_hmi-0094 (0094) ----------
sample_ckpt "ctrl_best_hmi-0094" "2026-08-10T11-00-23" "0094" "epoch=000563_val_loss_simple=0.0161.ckpt"
sample_ckpt "ctrl_best_hmi-0094" "2026-08-10T11-00-23" "0094" "epoch=000633_val_loss_simple=0.0172.ckpt"
sample_ckpt "ctrl_best_hmi-0094" "2026-08-10T11-00-23" "0094" "epoch=001488_val_loss_simple=0.0166.ckpt"
sample_ckpt "ctrl_best_hmi-0094" "2026-08-10T11-00-23" "0094" "last.ckpt"

# ---------- ctrl_best_hmi-0131 (0131) ----------
sample_ckpt "ctrl_best_hmi-0131" "2026-08-10T11-02-52" "0131" "epoch=000563_val_loss_simple=0.0145.ckpt"
sample_ckpt "ctrl_best_hmi-0131" "2026-08-10T11-02-52" "0131" "epoch=000633_val_loss_simple=0.0155.ckpt"
sample_ckpt "ctrl_best_hmi-0131" "2026-08-10T11-02-52" "0131" "epoch=001488_val_loss_simple=0.0153.ckpt"
sample_ckpt "ctrl_best_hmi-0131" "2026-08-10T11-02-52" "0131" "last.ckpt"

# ---------- ctrl_best_hmi-0171 (0171) ----------
sample_ckpt "ctrl_best_hmi-0171" "2026-08-10T11-05-21" "0171" "epoch=000563_val_loss_simple=0.0207.ckpt"
sample_ckpt "ctrl_best_hmi-0171" "2026-08-10T11-05-21" "0171" "epoch=000633_val_loss_simple=0.0218.ckpt"
sample_ckpt "ctrl_best_hmi-0171" "2026-08-10T11-05-21" "0171" "epoch=000664_val_loss_simple=0.0219.ckpt"
sample_ckpt "ctrl_best_hmi-0171" "2026-08-10T11-05-21" "0171" "last.ckpt"

# ---------- ctrl_best_hmi-0193 (0193) ----------
sample_ckpt "ctrl_best_hmi-0193" "2026-08-10T11-07-53" "0193" "epoch=000535_val_loss_simple=0.0254.ckpt"
sample_ckpt "ctrl_best_hmi-0193" "2026-08-10T11-07-53" "0193" "epoch=000563_val_loss_simple=0.0242.ckpt"
sample_ckpt "ctrl_best_hmi-0193" "2026-08-10T11-07-53" "0193" "epoch=000633_val_loss_simple=0.0251.ckpt"
sample_ckpt "ctrl_best_hmi-0193" "2026-08-10T11-07-53" "0193" "last.ckpt"

# ---------- ctrl_best_hmi-0211 (0211) ----------
sample_ckpt "ctrl_best_hmi-0211" "2026-08-10T11-10-22" "0211" "epoch=000563_val_loss_simple=0.0110.ckpt"
sample_ckpt "ctrl_best_hmi-0211" "2026-08-10T11-10-22" "0211" "epoch=000633_val_loss_simple=0.0113.ckpt"
sample_ckpt "ctrl_best_hmi-0211" "2026-08-10T11-10-22" "0211" "epoch=000664_val_loss_simple=0.0119.ckpt"
sample_ckpt "ctrl_best_hmi-0211" "2026-08-10T11-10-22" "0211" "last.ckpt"

# ---------- ctrl_best_hmi-0304 (0304) ----------
sample_ckpt "ctrl_best_hmi-0304" "2026-08-10T11-12-54" "0304" "epoch=000563_val_loss_simple=0.0179.ckpt"
sample_ckpt "ctrl_best_hmi-0304" "2026-08-10T11-12-54" "0304" "epoch=000633_val_loss_simple=0.0193.ckpt"
sample_ckpt "ctrl_best_hmi-0304" "2026-08-10T11-12-54" "0304" "epoch=001488_val_loss_simple=0.0189.ckpt"
sample_ckpt "ctrl_best_hmi-0304" "2026-08-10T11-12-54" "0304" "last.ckpt"

# ---------- ctrl_best_hmi-0335 (0335) ----------
sample_ckpt "ctrl_best_hmi-0335" "2026-08-10T11-15-22" "0335" "epoch=000563_val_loss_simple=0.0165.ckpt"
sample_ckpt "ctrl_best_hmi-0335" "2026-08-10T11-15-22" "0335" "epoch=000633_val_loss_simple=0.0174.ckpt"
sample_ckpt "ctrl_best_hmi-0335" "2026-08-10T11-15-22" "0335" "epoch=000664_val_loss_simple=0.0179.ckpt"
sample_ckpt "ctrl_best_hmi-0335" "2026-08-10T11-15-22" "0335" "last.ckpt"

# ---------- ctrl_best_hmi-1600 (1600) ----------
sample_ckpt "ctrl_best_hmi-1600" "2026-08-10T11-17-53" "1600" "epoch=000010_val_loss_simple=0.0026.ckpt"
sample_ckpt "ctrl_best_hmi-1600" "2026-08-10T11-17-53" "1600" "epoch=000085_val_loss_simple=0.0026.ckpt"
sample_ckpt "ctrl_best_hmi-1600" "2026-08-10T11-17-53" "1600" "epoch=000090_val_loss_simple=0.0025.ckpt"
sample_ckpt "ctrl_best_hmi-1600" "2026-08-10T11-17-53" "1600" "last.ckpt"

# ---------- ctrl_best_hmi-1700 (1700) ----------
sample_ckpt "ctrl_best_hmi-1700" "2026-08-10T10-50-19" "1700" "epoch=000633_val_loss_simple=0.0040.ckpt"
sample_ckpt "ctrl_best_hmi-1700" "2026-08-10T10-50-19" "1700" "epoch=001392_val_loss_simple=0.0038.ckpt"
sample_ckpt "ctrl_best_hmi-1700" "2026-08-10T10-50-19" "1700" "epoch=001488_val_loss_simple=0.0036.ckpt"
sample_ckpt "ctrl_best_hmi-1700" "2026-08-10T10-50-19" "1700" "last.ckpt"

# ---------- ctrl_best_hmi-4500 (4500) ----------
sample_ckpt "ctrl_best_hmi-4500" "2026-08-10T10-52-48" "4500" "epoch=000019_val_loss_simple=0.0107.ckpt"
sample_ckpt "ctrl_best_hmi-4500" "2026-08-10T10-52-48" "4500" "epoch=000101_val_loss_simple=0.0080.ckpt"
sample_ckpt "ctrl_best_hmi-4500" "2026-08-10T10-52-48" "4500" "epoch=000563_val_loss_simple=0.0089.ckpt"
sample_ckpt "ctrl_best_hmi-4500" "2026-08-10T10-52-48" "4500" "last.ckpt"

echo "==== [shard1] 全部完成 ===="
