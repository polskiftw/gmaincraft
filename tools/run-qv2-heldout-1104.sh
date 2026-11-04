#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
# All tests are frozen and sequential; one Minecraft process at a time.
# Names and seed stream never overlap TRAIN indices (the eval namespace is disjoint).
# Honor an existing journal by REFUSING to start, never overwrite.
models=(qv2longf1104 qv2triplef1104 qv2longf1104 qv2triplef1104)
tags=(qv2long_eval10a qv2base_eval10a qv2long_eval10b qv2base_eval10b)
rngs=(813777 813777 1639377 1639377)
for i in "${!models[@]}"; do
  echo "RUN_START ${tags[i]} model ${models[i]} RNG ${rngs[i]}"
  if [[ -e "logs/fresh_eval_${tags[i]}.jsonl" ]];then
     echo "FAILED existing journal ${tags[i]}";exit 1
  fi
  tools/run-replay-q-evaluation.sh "${models[i]}" "${tags[i]}" effect:visible_wood_crack_progress_v2 3600 600 10 "${rngs[i]}" q_sampled
  echo "RUN_FINISHED ${tags[i]}"
done
echo "HELDOUT_SUITE_COMPLETED"
