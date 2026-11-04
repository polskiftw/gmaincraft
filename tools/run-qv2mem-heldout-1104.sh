#!/usr/bin/env bash
set -euo pipefail
cd /workspace
models=(qv2memf1104 qv2longf1104 qv2memf1104 qv2longf1104)
tags=(memf_h16a1104 longf_h16a1104 memf_h16b1104 longf_h16b1104)
rngs=(813777 813777 1639377 1639377)
for i in "${!models[@]}"; do
  echo "RUN_START ${tags[i]} model ${models[i]} action_rng ${rngs[i]}"
  if [[ -e "logs/fresh_eval_${tags[i]}.jsonl" ]];then echo "REFUSING existing journal ${tags[i]}" >&2; exit 1; fi
  GMAINCRAFT_MEMORY_ONLY_NO_RGB=1 tools/run-replay-q-evaluation.sh "${models[i]}" "${tags[i]}" effect:visible_wood_crack_progress_v2 4800 600 16 "${rngs[i]}" q_sampled
  echo "RUN_FINISHED ${tags[i]}"
done
echo HELDOUT_SUITE_COMPLETED
