python -m solarchip.main.test \
    -r logs/solarctrl/ctrl_best_0094-hmi/2026-08-07T17-58-09 \
    --time_interval 5000 6000 --time_step 1 \
    --metrics mse psnr ssim --visualization true