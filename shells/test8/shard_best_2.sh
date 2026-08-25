#!/bin/bash
# ============================================================
# SolarCHIP 大规模测试 [best-only] 分片 2/8
# 时间区间: [5250, 5375)  步长: 10  显卡: musa:2
# 覆盖: checkpoints/compare_transfer + checkpoints/solarctrl 全部模型, 只补跑 best
# 输出: logs/sample/shard2/pt|metrics/... (独立分片目录, 互不覆盖)
# ============================================================

set -u

GPU=2
START=5250
END=5375
SAVE_ROOT=logs/sample/shard2

test_one() {
    local model="$1" run="$2" target="$3" base="$4" ckpt="$5" tag="$6"
    local logdir="${base}/${model}/${run}"
    local mdir="${SAVE_ROOT}/metrics/${target}/${model}"
    echo "==== [shard2] ${model} / ${tag} ===="
    if [ -n "$ckpt" ]; then
        python -m solarchip.main.test -r "$logdir" --ckpt "$ckpt" \
            --time_interval $START $END --time_step 10 \
            --metrics mse psnr ssim --visualization false \
            --device musa:$GPU --quiet --save_root "$SAVE_ROOT"
    else
        python -m solarchip.main.test -r "$logdir" \
            --time_interval $START $END --time_step 10 \
            --metrics mse psnr ssim --visualization false \
            --device musa:$GPU --quiet --save_root "$SAVE_ROOT"
    fi
    mkdir -p "$mdir"
    cp "$mdir/metrics.json" "$mdir/metrics_${tag}.json"
    cp "$mdir/metrics_cfg.json" "$mdir/metrics_cfg_${tag}.json"
}

# ---------- aia_hmi_dannehl_pix2pixcc_0094 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0094/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0094/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0094" "2026-08-16T16-52-55" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0131 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0131/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0131/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0131" "2026-08-16T16-55-23" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0171 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0171/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0171/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0171" "2026-08-16T16-58-02" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0193 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0193/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0193/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0193" "2026-08-16T17-00-23" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0211 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0211/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0211/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0211" "2026-08-16T17-02-53" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0304 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0304/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0304/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0304" "2026-08-16T17-05-23" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_0335 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0335/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_0335/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_0335" "2026-08-16T17-07-58" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_1600 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_1600/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_1600/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_1600" "2026-08-16T17-10-23" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_1700 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_1700/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_1700/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_1700" "2026-08-16T17-16-08" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_dannehl_pix2pixcc_4500 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_4500/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_dannehl_pix2pixcc_4500/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_dannehl_pix2pixcc_4500" "2026-08-16T17-18-38" "hmi" "checkpoints/compare_transfer" "epoch=000000.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0094 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0094/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0094/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0094" "2026-08-17T14-13-11" "hmi" "checkpoints/compare_transfer" "epoch=000028.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0131 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0131/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0131/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0131" "2026-08-17T14-15-43" "hmi" "checkpoints/compare_transfer" "epoch=000031.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0171 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0171/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0171/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0171" "2026-08-17T14-18-10" "hmi" "checkpoints/compare_transfer" "epoch=000060.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0193 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0193/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0193/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0193" "2026-08-17T14-20-52" "hmi" "checkpoints/compare_transfer" "epoch=000062.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0211 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0211/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0211/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0211" "2026-08-17T14-23-11" "hmi" "checkpoints/compare_transfer" "epoch=000044.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0304 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0304/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0304/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0304" "2026-08-17T14-25-41" "hmi" "checkpoints/compare_transfer" "epoch=000046.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_0335 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0335/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_0335/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_0335" "2026-08-17T14-28-12" "hmi" "checkpoints/compare_transfer" "epoch=000026.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_1600 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_1600/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_1600/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_1600" "2026-08-17T14-30-41" "hmi" "checkpoints/compare_transfer" "epoch=000135.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_1700 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_1700/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_1700/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_1700" "2026-08-17T14-13-02" "hmi" "checkpoints/compare_transfer" "epoch=000126.ckpt" "best"

# ---------- aia_hmi_i2iwfilm_4500 (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_4500/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/aia_hmi_i2iwfilm_4500/"sample_cfg_[0-9]*.pt
test_one "aia_hmi_i2iwfilm_4500" "2026-08-17T14-15-32" "hmi" "checkpoints/compare_transfer" "epoch=000009.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0094 (0094) ----------
rm -f "${SAVE_ROOT}/pt/0094/hmi_aia_dash_pix2pixhd_0094/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0094/hmi_aia_dash_pix2pixhd_0094/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0094" "2026-08-17T18-28-38" "0094" "checkpoints/compare_transfer" "epoch=000071.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0131 (0131) ----------
rm -f "${SAVE_ROOT}/pt/0131/hmi_aia_dash_pix2pixhd_0131/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0131/hmi_aia_dash_pix2pixhd_0131/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0131" "2026-08-17T18-31-08" "0131" "checkpoints/compare_transfer" "epoch=000028.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0171 (0171) ----------
rm -f "${SAVE_ROOT}/pt/0171/hmi_aia_dash_pix2pixhd_0171/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0171/hmi_aia_dash_pix2pixhd_0171/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0171" "2026-08-17T18-33-34" "0171" "checkpoints/compare_transfer" "epoch=000083.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0193 (0193) ----------
rm -f "${SAVE_ROOT}/pt/0193/hmi_aia_dash_pix2pixhd_0193/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0193/hmi_aia_dash_pix2pixhd_0193/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0193" "2026-08-17T18-36-05" "0193" "checkpoints/compare_transfer" "epoch=000061.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0211 (0211) ----------
rm -f "${SAVE_ROOT}/pt/0211/hmi_aia_dash_pix2pixhd_0211/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0211/hmi_aia_dash_pix2pixhd_0211/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0211" "2026-08-17T18-38-37" "0211" "checkpoints/compare_transfer" "epoch=000064.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0304 (0304) ----------
rm -f "${SAVE_ROOT}/pt/0304/hmi_aia_dash_pix2pixhd_0304/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0304/hmi_aia_dash_pix2pixhd_0304/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0304" "2026-08-17T18-41-08" "0304" "checkpoints/compare_transfer" "epoch=000022.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_0335 (0335) ----------
rm -f "${SAVE_ROOT}/pt/0335/hmi_aia_dash_pix2pixhd_0335/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0335/hmi_aia_dash_pix2pixhd_0335/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_0335" "2026-08-17T18-43-35" "0335" "checkpoints/compare_transfer" "epoch=000031.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_1600 (1600) ----------
rm -f "${SAVE_ROOT}/pt/1600/hmi_aia_dash_pix2pixhd_1600/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1600/hmi_aia_dash_pix2pixhd_1600/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_1600" "2026-08-17T18-46-05" "1600" "checkpoints/compare_transfer" "epoch=000078.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_1700 (1700) ----------
rm -f "${SAVE_ROOT}/pt/1700/hmi_aia_dash_pix2pixhd_1700/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1700/hmi_aia_dash_pix2pixhd_1700/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_1700" "2026-08-17T19-17-17" "1700" "checkpoints/compare_transfer" "epoch=000084.ckpt" "best"

# ---------- hmi_aia_dash_pix2pixhd_4500 (4500) ----------
rm -f "${SAVE_ROOT}/pt/4500/hmi_aia_dash_pix2pixhd_4500/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/4500/hmi_aia_dash_pix2pixhd_4500/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_dash_pix2pixhd_4500" "2026-08-17T19-19-45" "4500" "checkpoints/compare_transfer" "epoch=000064.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0094 (0094) ----------
rm -f "${SAVE_ROOT}/pt/0094/hmi_aia_sdoml_cnn_0094/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0094/hmi_aia_sdoml_cnn_0094/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0094" "2026-08-17T18-04-42" "0094" "checkpoints/compare_transfer" "epoch=000003.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0131 (0131) ----------
rm -f "${SAVE_ROOT}/pt/0131/hmi_aia_sdoml_cnn_0131/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0131/hmi_aia_sdoml_cnn_0131/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0131" "2026-08-17T18-07-05" "0131" "checkpoints/compare_transfer" "epoch=000078.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0171 (0171) ----------
rm -f "${SAVE_ROOT}/pt/0171/hmi_aia_sdoml_cnn_0171/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0171/hmi_aia_sdoml_cnn_0171/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0171" "2026-08-17T18-09-41" "0171" "checkpoints/compare_transfer" "epoch=000026.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0193 (0193) ----------
rm -f "${SAVE_ROOT}/pt/0193/hmi_aia_sdoml_cnn_0193/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0193/hmi_aia_sdoml_cnn_0193/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0193" "2026-08-17T18-12-09" "0193" "checkpoints/compare_transfer" "epoch=000026.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0211 (0211) ----------
rm -f "${SAVE_ROOT}/pt/0211/hmi_aia_sdoml_cnn_0211/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0211/hmi_aia_sdoml_cnn_0211/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0211" "2026-08-17T18-14-36" "0211" "checkpoints/compare_transfer" "epoch=000026.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0304 (0304) ----------
rm -f "${SAVE_ROOT}/pt/0304/hmi_aia_sdoml_cnn_0304/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0304/hmi_aia_sdoml_cnn_0304/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0304" "2026-08-17T18-17-06" "0304" "checkpoints/compare_transfer" "epoch=000012.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_0335 (0335) ----------
rm -f "${SAVE_ROOT}/pt/0335/hmi_aia_sdoml_cnn_0335/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0335/hmi_aia_sdoml_cnn_0335/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_0335" "2026-08-17T18-19-36" "0335" "checkpoints/compare_transfer" "epoch=000127.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_1600 (1600) ----------
rm -f "${SAVE_ROOT}/pt/1600/hmi_aia_sdoml_cnn_1600/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1600/hmi_aia_sdoml_cnn_1600/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_1600" "2026-08-17T18-22-06" "1600" "checkpoints/compare_transfer" "epoch=000010.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_1700 (1700) ----------
rm -f "${SAVE_ROOT}/pt/1700/hmi_aia_sdoml_cnn_1700/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1700/hmi_aia_sdoml_cnn_1700/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_1700" "2026-08-17T18-04-38" "1700" "checkpoints/compare_transfer" "epoch=000026.ckpt" "best"

# ---------- hmi_aia_sdoml_cnn_4500 (4500) ----------
rm -f "${SAVE_ROOT}/pt/4500/hmi_aia_sdoml_cnn_4500/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/4500/hmi_aia_sdoml_cnn_4500/"sample_cfg_[0-9]*.pt
test_one "hmi_aia_sdoml_cnn_4500" "2026-08-17T18-07-09" "4500" "checkpoints/compare_transfer" "epoch=000072.ckpt" "best"

# ---------- ctrl_best_0094-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0094-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0094-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0094-hmi" "2026-08-07T17-58-09" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0355.ckpt" "best"

# ---------- ctrl_best_0131-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0131-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0131-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0131-hmi" "2026-08-07T18-00-34" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0354.ckpt" "best"

# ---------- ctrl_best_0171-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0171-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0171-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0171-hmi" "2026-08-07T18-03-05" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0354.ckpt" "best"

# ---------- ctrl_best_0193-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0193-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0193-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0193-hmi" "2026-08-07T18-05-40" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0355.ckpt" "best"

# ---------- ctrl_best_0211-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0211-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0211-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0211-hmi" "2026-08-07T18-08-12" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0362.ckpt" "best"

# ---------- ctrl_best_0304-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0304-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0304-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0304-hmi" "2026-08-07T18-10-40" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0353.ckpt" "best"

# ---------- ctrl_best_0335-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0335-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_0335-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_0335-hmi" "2026-08-07T18-13-12" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0355.ckpt" "best"

# ---------- ctrl_best_1600-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_1600-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_1600-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_1600-hmi" "2026-08-07T18-15-34" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0362.ckpt" "best"

# ---------- ctrl_best_1700-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_1700-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_1700-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_1700-hmi" "2026-08-08T10-00-51" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0354.ckpt" "best"

# ---------- ctrl_best_4500-hmi (hmi) ----------
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_4500-hmi/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/hmi/ctrl_best_4500-hmi/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_4500-hmi" "2026-08-08T10-03-20" "hmi" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0360.ckpt" "best"

# ---------- ctrl_best_hmi-0094 (0094) ----------
rm -f "${SAVE_ROOT}/pt/0094/ctrl_best_hmi-0094/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0094/ctrl_best_hmi-0094/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0094" "2026-08-10T11-00-23" "0094" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0161.ckpt" "best"

# ---------- ctrl_best_hmi-0131 (0131) ----------
rm -f "${SAVE_ROOT}/pt/0131/ctrl_best_hmi-0131/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0131/ctrl_best_hmi-0131/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0131" "2026-08-10T11-02-52" "0131" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0145.ckpt" "best"

# ---------- ctrl_best_hmi-0171 (0171) ----------
rm -f "${SAVE_ROOT}/pt/0171/ctrl_best_hmi-0171/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0171/ctrl_best_hmi-0171/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0171" "2026-08-10T11-05-21" "0171" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0207.ckpt" "best"

# ---------- ctrl_best_hmi-0193 (0193) ----------
rm -f "${SAVE_ROOT}/pt/0193/ctrl_best_hmi-0193/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0193/ctrl_best_hmi-0193/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0193" "2026-08-10T11-07-53" "0193" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0242.ckpt" "best"

# ---------- ctrl_best_hmi-0211 (0211) ----------
rm -f "${SAVE_ROOT}/pt/0211/ctrl_best_hmi-0211/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0211/ctrl_best_hmi-0211/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0211" "2026-08-10T11-10-22" "0211" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0110.ckpt" "best"

# ---------- ctrl_best_hmi-0304 (0304) ----------
rm -f "${SAVE_ROOT}/pt/0304/ctrl_best_hmi-0304/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0304/ctrl_best_hmi-0304/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0304" "2026-08-10T11-12-54" "0304" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0179.ckpt" "best"

# ---------- ctrl_best_hmi-0335 (0335) ----------
rm -f "${SAVE_ROOT}/pt/0335/ctrl_best_hmi-0335/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/0335/ctrl_best_hmi-0335/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-0335" "2026-08-10T11-15-22" "0335" "checkpoints/solarctrl" "epoch=000563_val_loss_simple=0.0165.ckpt" "best"

# ---------- ctrl_best_hmi-1600 (1600) ----------
rm -f "${SAVE_ROOT}/pt/1600/ctrl_best_hmi-1600/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1600/ctrl_best_hmi-1600/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-1600" "2026-08-10T11-17-53" "1600" "checkpoints/solarctrl" "epoch=000090_val_loss_simple=0.0025.ckpt" "best"

# ---------- ctrl_best_hmi-1700 (1700) ----------
rm -f "${SAVE_ROOT}/pt/1700/ctrl_best_hmi-1700/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/1700/ctrl_best_hmi-1700/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-1700" "2026-08-10T10-50-19" "1700" "checkpoints/solarctrl" "epoch=001488_val_loss_simple=0.0036.ckpt" "best"

# ---------- ctrl_best_hmi-4500 (4500) ----------
rm -f "${SAVE_ROOT}/pt/4500/ctrl_best_hmi-4500/"sample_[0-9]*.pt
rm -f "${SAVE_ROOT}/pt/4500/ctrl_best_hmi-4500/"sample_cfg_[0-9]*.pt
test_one "ctrl_best_hmi-4500" "2026-08-10T10-52-48" "4500" "checkpoints/solarctrl" "epoch=000101_val_loss_simple=0.0080.ckpt" "best"

echo "==== [shard2] 全部完成 ===="
