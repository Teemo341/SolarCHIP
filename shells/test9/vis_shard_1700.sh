#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 1700)
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

# ---------- aia_hmi_dannehl_pix2pixcc_1700 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_1700/2026-08-16T17-16-08"
# ---------- aia_hmi_i2iwfilm_1700 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_1700/2026-08-17T14-13-02"
# ---------- hmi_aia_dash_pix2pixhd_1700 (1700) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_1700/2026-08-17T19-17-17"
# ---------- hmi_aia_sdoml_cnn_1700 (1700) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_1700/2026-08-17T18-04-38"
# ---------- ctrl_best_1700-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_1700-hmi/2026-08-08T10-00-51"
# ---------- ctrl_best_hmi-1700 (1700) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-1700/2026-08-10T10-50-19"

echo "==== [viz] 模态 1700 全部完成 ===="
