#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 0193)
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

# ---------- aia_hmi_dannehl_pix2pixcc_0193 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0193/2026-08-16T17-00-23"
# ---------- aia_hmi_i2iwfilm_0193 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_0193/2026-08-17T14-20-52"
# ---------- hmi_aia_dash_pix2pixhd_0193 (0193) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_0193/2026-08-17T18-36-05"
# ---------- hmi_aia_sdoml_cnn_0193 (0193) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_0193/2026-08-17T18-12-09"
# ---------- ctrl_best_0193-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_0193-hmi/2026-08-07T18-05-40"
# ---------- ctrl_best_hmi-0193 (0193) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-0193/2026-08-10T11-07-53"

echo "==== [viz] 模态 0193 全部完成 ===="
