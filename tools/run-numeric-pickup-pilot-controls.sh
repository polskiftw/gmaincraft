#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
for control in uniform_random greedy sampled;do
  case "$control" in
    uniform_random) tag=numitem_val_random1104a;;
    greedy) tag=numitem_val_greedy1104a;;
    sampled) tag=numitem_val_sample1104a;;
  esac
  echo "EVAL_START $control $tag"
  if [[ -e "logs/numeric_pickup_$tag.jsonl" ]];then
    echo "REFUSING existing eval $tag" >&2;exit 1
  fi
  set +e
  GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh \
    .venv/bin/python -u experiments/train_numeric_item_pickup.py \
    --run-tag "$tag" --mode eval --model-tag numitem_pilot1104a \
    --episodes 30 --max-actions 24 --case-seed 663311 \
    --policy-seed 884422 --device cpu --control "$control" \
    --report-every 10 --visible \
    > "logs/numeric_pickup_$tag.console" 2>&1
  outer=$?
  set -e
  .venv/bin/python - "$tag" "$control" "$outer" <<'PY'
import json,sys
tag,control,outer=sys.argv[1:4]
rows=[json.loads(s) for s in open('logs/numeric_pickup_'+tag+'.jsonl') if s.strip()]
ep=[x for x in rows if x.get('stage')=='episode_committed']
summary=rows[-1]
if summary.get('stage')!='completed' or len(ep)!=30 or summary['optimizer_updates']!=0:
    raise RuntimeError('Frozen real-game item pickup control did not complete')
print('FROZEN VERIFIED',control,'outer_code',outer,
      'successes',summary['successes'],'episodes',summary['episodes'],
      'items',summary['inventory_items'],flush=True)
PY
  echo "EVAL_FINISHED $control"
done
echo PICKUP_CONTROL_SUITE_COMPLETED
