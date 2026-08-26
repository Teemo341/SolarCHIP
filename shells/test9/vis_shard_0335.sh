#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 0335)
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

# ---------- aia_hmi_dannehl_pix2pixcc_0335 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0335/2026-08-16T17-07-58"
# ---------- aia_hmi_i2iwfilm_0335 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_0335/2026-08-17T14-28-12"
# ---------- hmi_aia_dash_pix2pixhd_0335 (0335) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_0335/2026-08-17T18-43-35"
# ---------- hmi_aia_sdoml_cnn_0335 (0335) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_0335/2026-08-17T18-19-36"
# ---------- ctrl_best_0335-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_0335-hmi/2026-08-07T18-13-12"
# ---------- ctrl_best_hmi-0335 (0335) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-0335/2026-08-10T11-15-22"

echo "==== [viz] 模态 0335 全部完成 ===="
