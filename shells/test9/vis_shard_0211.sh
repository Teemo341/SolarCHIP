#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 0211)
# 区间 [5000, 6000)  步长: 50  增强: none  只补缺失 png
# 已有 pt 不重新采样 (sample.py 跳过存在文件)
# 输出: logs/sample/png/...
# ============================================================

set -u

vis_one() {
    local logdir="$1"
    echo "==== [viz] ${logdir} ===="
    python -m solarchip.main.sample -r "$logdir" \
        --time_interval 5000 6000 --time_step 50 \
        --save_root logs/sample/pt --visualization true \
        --enhance none --device cpu --quiet
}

# ---------- aia_hmi_dannehl_pix2pixcc_0211 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0211/2026-08-16T17-02-53"
# ---------- aia_hmi_i2iwfilm_0211 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_0211/2026-08-17T14-23-11"
# ---------- hmi_aia_dash_pix2pixhd_0211 (0211) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_0211/2026-08-17T18-38-37"
# ---------- hmi_aia_sdoml_cnn_0211 (0211) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_0211/2026-08-17T18-14-36"
# ---------- ctrl_best_0211-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_0211-hmi/2026-08-07T18-08-12"
# ---------- ctrl_best_hmi-0211 (0211) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-0211/2026-08-10T11-10-22"

echo "==== [viz] 模态 0211 全部完成 ===="
