#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 0094)
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

# ---------- aia_hmi_dannehl_pix2pixcc_0094 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0094/2026-08-16T16-52-55"
# ---------- aia_hmi_i2iwfilm_0094 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_0094/2026-08-17T14-13-11"
# ---------- hmi_aia_dash_pix2pixhd_0094 (0094) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_0094/2026-08-17T18-28-38"
# ---------- hmi_aia_sdoml_cnn_0094 (0094) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_0094/2026-08-17T18-04-42"
# ---------- ctrl_best_0094-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09"
# ---------- ctrl_best_hmi-0094 (0094) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-0094/2026-08-10T11-00-23"

echo "==== [viz] 模态 0094 全部完成 ===="
