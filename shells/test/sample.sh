python -m solarchip.main.sample \
        -r logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 \
        --time_interval 5000 6000 --time_step 100 \
        --visualization true

python -m solarchip.main.sample \
        -r logs/compare_transfer/aia_hmi_dannehl_pix2pixcc_0094/2026-08-16T16-52-55 \
        --time_interval 5000 6000 --time_step 100 \
        --visualization true

python -m solarchip.main.sample \
        -r logs/compare_transfer/aia_hmi_i2iwfilm_0094/2026-08-17T14-13-11 \
        --time_interval 5000 6000 --time_step 100 \
        --visualization true

python -m solarchip.main.sample \
        -r original --target_modal hmi \
        --time_interval 5000 6000 --time_step 100 \
        --visualization true