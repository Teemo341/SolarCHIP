#!/bin/bash
# ============================================================
# SolarCHIP 已有 pt -> png 可视化分片 (模态 0131)
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

# ---------- aia_hmi_dannehl_pix2pixcc_0131 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0131/2026-08-16T16-55-23"
# ---------- aia_hmi_i2iwfilm_0131 (hmi) ----------
vis_one "logs/compare_transfer/aia_hmi_i2iwfilm_0131/2026-08-17T14-15-43"
# ---------- hmi_aia_dash_pix2pixhd_0131 (0131) ----------
vis_one "logs/compare_transfer/hmi_aia_dash_pix2pixhd_0131/2026-08-17T18-31-08"
# ---------- hmi_aia_sdoml_cnn_0131 (0131) ----------
vis_one "logs/compare_transfer/hmi_aia_sdoml_cnn_0131/2026-08-17T18-07-05"
# ---------- ctrl_best_0131-hmi (hmi) ----------
vis_one "logs/solarctrl/ctrl_best_0131-hmi/2026-08-07T18-00-34"
# ---------- ctrl_best_hmi-0131 (0131) ----------
vis_one "logs/solarctrl/ctrl_best_hmi-0131/2026-08-10T11-02-52"

echo "==== [viz] 模态 0131 全部完成 ===="
