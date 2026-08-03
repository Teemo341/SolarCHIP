SECONDS_UNTIL_8AM=$(( $(date -d "tomorrow 08:00" +%s) - $(date +%s) ))
echo "Will stop after ${SECONDS_UNTIL_8AM}s"

timeout --signal=SIGINT ${SECONDS_UNTIL_8AM} python -m solarchip.main.train \
  --base ./configs/solarchip/CNN_AE_base_zscore.yaml \
  --resume logs/solarchip/CNN_AE_base_zscore_2026-07-07T15-43-29