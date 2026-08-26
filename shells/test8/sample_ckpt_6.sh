#!/bin/bash
# ============================================================
# compare_transfer 全量采样分片 6/8
# 时间区间: [5750, 5875)  步长: 10  显卡: musa:6
# 覆盖: checkpoints/compare_transfer 全部模型 × 全部 checkpoint
# 输出: logs/sample/pt/{目标模态}/{模型名}/{ckpt名}/sample_<时间>.pt
# ============================================================

set -u

GPU=6
START=5750
END=5875

sample_ckpt() {
    local model="$1" run="$2" target="$3" ckpt="$4"
    local ckpt_dir="${ckpt%.ckpt}"
    echo "==== [shard6] ${model} / ${ckpt} ===="
    python -m solarchip.main.sample \
        -r "checkpoints/compare_transfer/${model}/${run}" \
        --ckpt "$ckpt" \
        --time_interval $START $END --time_step 10 \
        --seed 42 --device musa:$GPU \
        --save_root logs/sample/pt \
        --sample_subdir "$ckpt_dir" --quiet
}

# ---------- aia_hmi_dannehl_pix2pixcc_0094 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0094" "2026-08-16T16-52-55" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0094" "2026-08-16T16-52-55" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0131 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0131" "2026-08-16T16-55-23" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0131" "2026-08-16T16-55-23" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0171 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0171" "2026-08-16T16-58-02" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0171" "2026-08-16T16-58-02" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0193 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0193" "2026-08-16T17-00-23" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0193" "2026-08-16T17-00-23" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0211 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0211" "2026-08-16T17-02-53" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0211" "2026-08-16T17-02-53" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0304 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0304" "2026-08-16T17-05-23" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0304" "2026-08-16T17-05-23" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_0335 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0335" "2026-08-16T17-07-58" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_0335" "2026-08-16T17-07-58" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_1600 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_1600" "2026-08-16T17-10-23" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_1600" "2026-08-16T17-10-23" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_1700 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_1700" "2026-08-16T17-16-08" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_1700" "2026-08-16T17-16-08" "hmi" "last.ckpt"

# ---------- aia_hmi_dannehl_pix2pixcc_4500 (hmi) ----------
sample_ckpt "aia_hmi_dannehl_pix2pixcc_4500" "2026-08-16T17-18-38" "hmi" "epoch=000000.ckpt"
sample_ckpt "aia_hmi_dannehl_pix2pixcc_4500" "2026-08-16T17-18-38" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0094 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0094" "2026-08-17T14-13-11" "hmi" "epoch=000028.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0094" "2026-08-17T14-13-11" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0131 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0131" "2026-08-17T14-15-43" "hmi" "epoch=000031.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0131" "2026-08-17T14-15-43" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0171 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0171" "2026-08-17T14-18-10" "hmi" "epoch=000060.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0171" "2026-08-17T14-18-10" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0193 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0193" "2026-08-17T14-20-52" "hmi" "epoch=000062.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0193" "2026-08-17T14-20-52" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0211 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0211" "2026-08-17T14-23-11" "hmi" "epoch=000044.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0211" "2026-08-17T14-23-11" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0304 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0304" "2026-08-17T14-25-41" "hmi" "epoch=000046.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0304" "2026-08-17T14-25-41" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_0335 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_0335" "2026-08-17T14-28-12" "hmi" "epoch=000026.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_0335" "2026-08-17T14-28-12" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_1600 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_1600" "2026-08-17T14-30-41" "hmi" "epoch=000135.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_1600" "2026-08-17T14-30-41" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_1700 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_1700" "2026-08-17T14-13-02" "hmi" "epoch=000126.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_1700" "2026-08-17T14-13-02" "hmi" "last.ckpt"

# ---------- aia_hmi_i2iwfilm_4500 (hmi) ----------
sample_ckpt "aia_hmi_i2iwfilm_4500" "2026-08-17T14-15-32" "hmi" "epoch=000009.ckpt"
sample_ckpt "aia_hmi_i2iwfilm_4500" "2026-08-17T14-15-32" "hmi" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0094 (0094) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0094" "2026-08-17T18-28-38" "0094" "epoch=000071.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0094" "2026-08-17T18-28-38" "0094" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0131 (0131) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0131" "2026-08-17T18-31-08" "0131" "epoch=000028.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0131" "2026-08-17T18-31-08" "0131" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0171 (0171) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0171" "2026-08-17T18-33-34" "0171" "epoch=000083.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0171" "2026-08-17T18-33-34" "0171" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0193 (0193) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0193" "2026-08-17T18-36-05" "0193" "epoch=000061.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0193" "2026-08-17T18-36-05" "0193" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0211 (0211) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0211" "2026-08-17T18-38-37" "0211" "epoch=000064.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0211" "2026-08-17T18-38-37" "0211" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0304 (0304) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0304" "2026-08-17T18-41-08" "0304" "epoch=000022.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0304" "2026-08-17T18-41-08" "0304" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_0335 (0335) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_0335" "2026-08-17T18-43-35" "0335" "epoch=000031.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_0335" "2026-08-17T18-43-35" "0335" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_1600 (1600) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_1600" "2026-08-17T18-46-05" "1600" "epoch=000078.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_1600" "2026-08-17T18-46-05" "1600" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_1700 (1700) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_1700" "2026-08-17T19-17-17" "1700" "epoch=000084.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_1700" "2026-08-17T19-17-17" "1700" "last.ckpt"

# ---------- hmi_aia_dash_pix2pixhd_4500 (4500) ----------
sample_ckpt "hmi_aia_dash_pix2pixhd_4500" "2026-08-17T19-19-45" "4500" "epoch=000064.ckpt"
sample_ckpt "hmi_aia_dash_pix2pixhd_4500" "2026-08-17T19-19-45" "4500" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0094 (0094) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0094" "2026-08-17T18-04-42" "0094" "epoch=000003.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0094" "2026-08-17T18-04-42" "0094" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0131 (0131) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0131" "2026-08-17T18-07-05" "0131" "epoch=000078.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0131" "2026-08-17T18-07-05" "0131" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0171 (0171) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0171" "2026-08-17T18-09-41" "0171" "epoch=000026.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0171" "2026-08-17T18-09-41" "0171" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0193 (0193) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0193" "2026-08-17T18-12-09" "0193" "epoch=000026.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0193" "2026-08-17T18-12-09" "0193" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0211 (0211) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0211" "2026-08-17T18-14-36" "0211" "epoch=000026.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0211" "2026-08-17T18-14-36" "0211" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0304 (0304) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0304" "2026-08-17T18-17-06" "0304" "epoch=000012.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0304" "2026-08-17T18-17-06" "0304" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_0335 (0335) ----------
sample_ckpt "hmi_aia_sdoml_cnn_0335" "2026-08-17T18-19-36" "0335" "epoch=000127.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_0335" "2026-08-17T18-19-36" "0335" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_1600 (1600) ----------
sample_ckpt "hmi_aia_sdoml_cnn_1600" "2026-08-17T18-22-06" "1600" "epoch=000010.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_1600" "2026-08-17T18-22-06" "1600" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_1700 (1700) ----------
sample_ckpt "hmi_aia_sdoml_cnn_1700" "2026-08-17T18-04-38" "1700" "epoch=000026.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_1700" "2026-08-17T18-04-38" "1700" "last.ckpt"

# ---------- hmi_aia_sdoml_cnn_4500 (4500) ----------
sample_ckpt "hmi_aia_sdoml_cnn_4500" "2026-08-17T18-07-09" "4500" "epoch=000072.ckpt"
sample_ckpt "hmi_aia_sdoml_cnn_4500" "2026-08-17T18-07-09" "4500" "last.ckpt"

echo "==== [shard6] 全部完成 ===="
