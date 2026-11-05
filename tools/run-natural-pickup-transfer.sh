#!/usr/bin/env bash
# Exact matched real vanilla noise worlds: original frozen mining Q +
# optional player-visible, UNLEARNED switch to a frozen numeric pickup motor.
# No staged items, no evaluator-owned progression handoff and NO Q training.
set -euo pipefail
cd "$(dirname "$0")/.."
usage(){
  echo 'Usage: tools/run-natural-pickup-transfer.sh start PREFIX EVAL_INDEX WORLDS SIX_TICK_UNITS ACTION_RNG [Q_MODEL] [PICKUP_MODEL] [v1|budget_v2|explore_v1|learned_v1] [eval|data_train] [OPTION_MODEL] [AUTO_TRAIN_OPTION_TAG]' >&2
  echo '       tools/run-natural-pickup-transfer.sh status PREFIX' >&2
  echo 'This is independent frozen skill transfer evaluation, NOT new training.' >&2
  exit 2
}
[[ $# -ge 2 ]] || usage
mode="$1";prefix="$2";shift 2
[[ "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,11}$ ]] || usage
shadow="${prefix}_s"
active="${prefix}_p"
log="logs/natural_transfer_${prefix}_launcher.console"
report="logs/natural_transfer_${prefix}.summary.json"
case "$mode" in
 status)
   [[ $# == 0 ]] || usage
   suite_lock="checkpoints/natural_transfer_${prefix}.lock"
   if [[ -f "$suite_lock" ]];then
     exec 8<"$suite_lock"
     if flock -n 8;then
       echo "NATURAL SUITE LOCK FREE: $prefix is not currently running"
       flock -u 8
     else
       echo "NATURAL SUITE LOCK HELD: $prefix has an active sequential worker"
     fi
   fi
   for tag in "$shadow" "$active";do
     .venv/bin/python - "$tag" <<'PY'
import json,sys
from pathlib import Path
p=Path(f'logs/fresh_eval_{sys.argv[1]}.jsonl')
if not p.is_file():
 print('NATURAL STATUS',sys.argv[1], 'not launched');sys.exit()
rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
eps=[r for r in rows if r.get('stage')=='episode_committed']
print('NATURAL STATUS',sys.argv[1],'committed_worlds',len(eps),
      'wood_inventory_gains',sum(e['counts'].get('wood_inventory_gain',0) for e in eps),
      'pickup_motors',sum(e.get('pickup_transfer_audit',{}).get('pickup_policy_actions',0) for e in eps),
      'last_stage',rows[-1]['stage'])
PY
   done
   if [[ -f "$report" ]];then echo "NATURAL SUITE FINISHED: $report";fi
   [[ ! -f "$log" ]] || tail -3 "$log" | cut -c1-280
   exit 0
   ;;
 start)
   [[ $# -ge 4 && $# -le 10 ]] || usage
   [[ ! -e "$log" && ! -e "$report" && ! -e "logs/fresh_eval_${shadow}.jsonl" && ! -e "logs/fresh_eval_${active}.jsonl" ]] || {
      echo 'Existing natural suite data; never overwrite experiments.' >&2;exit 1;
   }
   # The user may ask for changes while a long study runs. Bash reads its
   # script on demand, so editing the original in place can corrupt the
   # executing shell (previous suite completed but failed after both arms).
   # Freeze EXACT verified launcher source bytes under this run tag.
   snapshot="logs/natural_transfer_${prefix}_launcher_snapshot.sh"
   if [[ -e "$snapshot" ]];then echo "Launcher snapshot exists: $snapshot" >&2;exit 1;fi
   cp -- "$0" "$snapshot"
   chmod 500 "$snapshot"
   nohup setsid bash "$snapshot" execute "$prefix" "$@" > "$log" 2>&1 < /dev/null &
   echo "NATURAL SUITE LAUNCHED prefix=$prefix pid=$!"
   echo "STATUS: tools/run-natural-pickup-transfer.sh status $prefix"
   echo "WATCH: tail -f $log"
   exit 0
   ;;
 execute)
   [[ $# -ge 4 && $# -le 10 ]] || usage
   ;;
 *)usage;;
esac
index="$1";count="$2";units="$3";rng="$4"
q_model="${5:-qv2memf1104}"
pickup_model="${6:-hardrep1105b}"
variant="${7:-v1}"
data_mode="${8:-eval}"
option_model="${9:-}"
auto_option_tag="${10:-}"
if [[ -n "$auto_option_tag" ]];then
  [[ "$data_mode" == data_train && "$auto_option_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] || usage
  [[ ! -e "checkpoints/natural_option_${auto_option_tag}.pt" &&
     ! -e "logs/natural_option_${auto_option_tag}.train.json" ]] || {
      echo 'Refusing to overwrite previous trained natural option selector' >&2;exit 1;
  }
fi
[[ "$variant" == v1 || "$variant" == budget_v2 || "$variant" == explore_v1 || "$variant" == learned_v1 ]] || usage
[[ "$data_mode" == eval || "$data_mode" == data_train ]] || usage
if [[ "$variant" == budget_v2 ]];then
  shadow_selector='shadow_budget_v2'
  active_selector='visible_memory_budget_v2'
elif [[ "$variant" == explore_v1 ]];then
  [[ "$data_mode" == data_train ]] || usage
  shadow_selector='shadow_option_explore_v1'
  active_selector='option_explore_v1'
elif [[ "$variant" == learned_v1 ]];then
  [[ "$data_mode" == eval && "$option_model" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] || usage
  shadow_selector='shadow_learned_option_v1'
  active_selector='learned_option_v1'
else
  shadow_selector='shadow'
  active_selector='visible_memory_v1'
fi
# Source name is consistent between arms, independent of live-world tags.
cache_tag="${prefix}_paired"
[[ "$index" =~ ^[0-9]{1,12}$ && "$count" =~ ^[1-9][0-9]*$ &&
   "$units" =~ ^[1-9][0-9]*$ && "$rng" =~ ^[1-9][0-9]*$ &&
   "$q_model" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ &&
   "$pickup_model" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] || usage
(( count >= 2 && count <= 40 && units >= 120 && units <= 24000 && index + count < 1000000000000 )) || usage
mkdir -p logs checkpoints
# One suite at a time. Individual fresh-eval locks still protect each tag.
exec 9>"checkpoints/natural_transfer_${prefix}.lock"
flock -n 9 || { echo "Natural pair suite $prefix already running" >&2;exit 1; }
[[ ! -e "logs/fresh_eval_${shadow}.jsonl" && ! -e "logs/fresh_eval_${active}.jsonl" ]] || {
  echo 'Partial prior suite must be inspected; refusing automatic restart.' >&2;exit 1;
}
for selector in "$shadow_selector" "$active_selector";do
  if [[ "$selector" == "$shadow_selector" ]];then tag="$shadow";else tag="$active";fi
  console="logs/natural_transfer_${prefix}_${selector}.console"
  echo "NATURAL SUITE START tag=$tag selector=$selector count=$count index=$index"
  args=(--mode eval --control q_sampled --run-tag "$tag" --q-model "$q_model"
        --q-goal effect:visible_wood_crack_progress_v2
        --pickup-model "$pickup_model" --pickup-selector "$selector"
        --paired-world-cache "$cache_tag"
        --decisions "$((count*units))" --episode-decisions "$units"
        --eval-seed-start "$index" --seed-root 917211 --policy-seed "$rng"
        --reward-profile wood_temporal_v9 --memory-only-no-rgb
        --retain-worlds 1 --device cpu --visible --report-every "$units"
        --record-option-experience --option-data-namespace "${data_mode#data_}")
  if [[ "$variant" == learned_v1 ]];then
    args+=(--natural-option-model "$option_model")
  fi
  set +e
  .venv/bin/python -u tools/stream-numeric-pickup.py "$console" -- \
    tools/run-mcio-viewer.sh .venv/bin/python -u experiments/fresh_seed_treadmill.py "${args[@]}"
  code=$?
  set -e
  .venv/bin/python - "$tag" "$count" "$units" "$index" "$q_model" "$pickup_model" "$selector" "$code" "$data_mode" <<'PY'
import json,sys
from pathlib import Path
tag,count,units,index,model,pickup,selector,outer,data_mode=sys.argv[1:]
count=int(count);units=int(units);index=int(index);outer=int(outer)
path=Path('logs/fresh_eval_'+tag+'.jsonl')
rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
eps=[x for x in rows if x.get('stage')=='episode_committed']
config=rows[0];last=rows[-1];meta=config['metadata']
if (outer not in (0,139) or last.get('stage')!='completed' or
    last.get('evaluated_episodes')!=count or last.get('decisions')!=count*units or
    last.get('new_optimizer_updates_in_eval')!=0 or
    config.get('q_model')!=model or config.get('q_goal')!='effect:visible_wood_crack_progress_v2' or
    meta.get('pickup_model')!=pickup or meta.get('pickup_selector')!=selector or
    meta.get('gameplay_commands')!=0 or len(eps)!=count or
    meta.get('record_real_option_experience') is not True or
    meta.get('option_data_namespace')!=('train' if data_mode=='data_train' else 'eval') or
    last.get('seed_namespace')!=('train' if data_mode=='data_train' else 'eval') or
    any('first_player_pose' not in e or e.get('updates')!=0 or e.get('episode')!=index+i or e['decisions']!=units
        for i,e in enumerate(eps))):
    sys.exit('FAILED NATURAL EVAL: incomplete or mismatched frozen/real-game evidence for '+tag)
print('NATURAL RUN VERIFIED',tag,'mode',selector,'worlds',count,
      'wood_inventory',sum(e['counts'].get('wood_inventory_gain',0) for e in eps),
      'pickup_actions',sum(e['pickup_transfer_audit'].get('pickup_policy_actions',0) for e in eps),
      'outer',outer,flush=True)
PY
done
if [[ "$data_mode" == data_train ]];then
  # Authentic off-policy data remains usable if two independent rollout
  states differ; the report MUST label them as a diagnostic, never as
  causal heldout uplift. Offline loader separately verifies every REAL row.
  .venv/bin/python tools/analyze-natural-pickup-transfer.py \
    "$shadow" "$active" --diagnose-mismatched-spawns --output "$report"
else
  .venv/bin/python tools/analyze-natural-pickup-transfer.py \
    "$shadow" "$active" --output "$report"
fi
echo "NATURAL TRANSFER SUITE COMPLETED $report"
if [[ -n "$auto_option_tag" ]];then
  echo "NATURAL OPTION TRAINING FROM COMMITTED REAL TRAIN-ONLY MINE/PICKUP OUTCOMES: $auto_option_tag"
  .venv/bin/python -u experiments/train_natural_option_offline.py \
    --tag "$auto_option_tag" --source "$shadow" --source "$active" \
    --steps 320 --batch-size 64
fi
