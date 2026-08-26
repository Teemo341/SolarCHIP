#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 4500)
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

# ---------- aia_hmi_dannehl_pix2pixcc_4500 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_4500/2026-08-16T17-18-38"
# ---------- aia_hmi_i2iwfilm_4500 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_4500/2026-08-17T14-15-32"
# ---------- hmi_aia_dash_pix2pixhd_4500 (4500) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_4500/2026-08-17T19-19-45"
# ---------- hmi_aia_sdoml_cnn_4500 (4500) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_4500/2026-08-17T18-07-09"
# ---------- ctrl_best_4500-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_4500-hmi/2026-08-08T10-03-20"
# ---------- ctrl_best_hmi-4500 (4500) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-4500/2026-08-10T10-52-48"

echo "==== [viz] 模态 4500 全部完成 ===="
