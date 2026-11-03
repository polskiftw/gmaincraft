# START HERE — next ChatGPT/Gentoo Shell session

**Priority order is intentional.** Updated November 2, 2026. This document
is the continuation contract for the next conversation. **The user wants
memory-derived structured game state**, not a permanently screenshot-only
agent; read [OBSERVATION_POLICY.md](OBSERVATION_POLICY.md). The memory-state
work follows **after** the performance regression investigation.

## Current continuation point — V4 policy, measured render-distance lever

**This paragraph supersedes the historical FIRST/SECOND instructions below.**
The user approved changing any disposable game settings within `/workspace`,
and requested that performance be watched continuously. The V4
`player_knowledge_v1`+RGB neural policy now ACTUALLY trains and runs as frozen
Minecraft gameplay with the same opt-in bounded memory observation:
`src/gmaincraft/memory_factorized.py`, `experiments/joint_log_memory.py`,
`experiments/memory_frozen_eval.py`, and `natural_world_benchmark.py --modes
memory_factorized`. Ten real CUDA episodes and 12 frozen cases passed
pipeline/sequence tests, but learned **0 wood inventory pickups**. See
[`MEMORY_POLICY_V4.md`](MEMORY_POLICY_V4.md) for exact checkpoints, seeds,
commands and limitations. Do not reuse run tags/checkpoints or claim success.

**Very actionable performance measurement:** on real, genuinely natural
seed246813579 (600 idle Minecraft ticks, same initial pose, no neural net),
original options took **4.535 s** in MCio game I/O; VSync-off/FPS260 took
**4.265 s**; adding renderDistance8 took **3.387 s**, 6 took **2.590 s**,
4 took **1.695 s**. The first 100 game ticks after launch are slower than
later ones. This strongly narrows the current 4s regression to render/chunk
load/viewport costs, not model inference. **It does not establish a unique
internal Java cause for every historical slowdown.** `instance_options.py`
backs up and restores only the disposable MCio options; no global changes.
See [`MINECRAFT_RENDER_PERF.md`](MINECRAFT_RENDER_PERF.md). V4 now defaults
to legal 4-chunk rendering to prioritize learner throughput, but this also
reduces distant visual perception and MUST match paired scored comparisons.

**Critical operational instruction:** use separate Gamescope sessions for
V4 training and frozen evaluation. An otherwise valid 10-episode CUDA run
`memfast_nov03c` wrote a valid checkpoint, then Gamescope aborted during the
SECOND Minecraft launch (`vulkan_remake_swapchain()` assertion) and left a
Python child; SIGTERM to **that owned child only** restored the temporary
options backup. Running `memory_frozen_eval.py` separately passed 12 cases;
wrapper exit 139 AFTER child success remains expected. Avoid repeating
failed tags. Check local ignored `logs/` and checkpoint before cleanup.

**NEXT:** implement and TEST read-only MCio/Fabric **targeted-visible block
information** and other player-knowable status unavailable in current
ObservationPacket; start with actual vanilla hit raycast as an unambiguous
baseline, then empirically calibrate FOV/range/occlusion/transparency/
projected size for a bounded visible object list before any general model
consumes it. MCio Java source at tag 1.6.1+1.21.3 has been inspected in
ignored `mcio-data/mcio-upstream-inspect`. Keep vanilla inputs/scored actions,
no X-ray/hidden seed oracle, parity between train/eval. Compare numeric+
RGB, numeric ablation, and legacy RGB on identical held-out near/broad and
new natural-world seeds; continue from-scratch full first-log acquisition
ONLY when the observable world/object pipeline is real. Run pure tests,
short live game probes, update docs/TODO/results, commit and push Git. Do not
blindly scale the currently zero-success PPO.

## November 2 update: current runtime recovered, cause not yet proven

Fresh independent Minecraft launches using `experiments/mcio_throughput_probe.py`
now show **0.62–0.72 s/300 synchronized vanilla game ticks after the first
300-tick repeat**, with real oak-log inventory, no missing frames and no
PyTorch. First repeats take ~1.17–1.24 s. 3 launches tested original
options and temporarily uncapped graphics; the uncapped option provided no
consistent material improvement, and options were restored. A fourth launch
with the new opt-in `player_knowledge_v1` memory state passed live Minecraft
checks and took 0.72 s after warm-up. See
[`MCIO_THROUGHPUT_2026_11_02.md`](MCIO_THROUGHPUT_2026_11_02.md) for exact
values, limitations and reproduction. **This is repeatable current speed,
NOT a proven explanation of the older ~4-second runs.** The MCio pacing
TODO remains open for reproducibility under the older workload, though
recovering the historical throughput range permits short follow-on sensor
research without committing to prolonged training.

**Current next implementation:** `src/gmaincraft/player_knowledge.py` is a
versioned opt-in, immutable adapter for player numeric state/inventory/known
stats; pure tests and the first live packet/inventory smoke pass. Existing
legacy RGB-only agents are untouched. The new structured input must next be
integrated **identically** into train and held-out inference, with explicit
schema versioning, and visible block/entity eligibility research is still
pending. Do not claim a learned general log-finding agent yet.

## HISTORICAL plan: earlier MCio regression diagnosis (now characterized)

**Do this before further reinforcement-learning architecture work, extended GPU training, or crafting/speedrun subtasks.** The present bottleneck is a ~5x-6x regression in real Minecraft game-step throughput, making research expensive. It is **not yet explained or fixed**.

Measured, not inferred:

| Measured workload | Earlier | Current |
|---|---:|---:|
| ~300 simulated game ticks in an ordinary joint-policy episode | ~0.6-0.8 seconds | ~3.9-4.0 seconds |
| `profile01`, 300 ticks, CUDA PPO, 320x180 RGB | — | MCio send/receive 3.876 s; CUDA inference 0.09 s; RGB processing 0.015 s |
| `no_torch_speed01`, 300 ticks, normal controls without Torch | — | 5.008 s |
| `smallframeperf01`, same control at 160x96 | — | 4.892 s |

Historical and current runs are **not necessarily a strictly matched A/B comparison** (different run times, host conditions, process startup/world state). Establish a reproducible matched baseline before claiming a root cause or speedup. See `docs/VISUAL_SELF_SUPERVISION.md` section **5** and local ignored `logs/` for original measurement evidence.

### Immediate operating procedure for the assistant

1. Confirm the user enabled **Gentoo Shell** for this conversation. Work only in its isolated `/workspace` filesystem. Run `pwd`, `git status --short --branch`, `git log -1 --oneline`, and optionally `git pull --ff-only` if clean and remotes are accessible. Git works **directly in the bridge** (including push); the GitHub connector is unnecessary for normal repo tasks. If the bridge is not connected, explain it plainly instead of pretending to access the computer.
2. Read `AGENTS.md`, **`docs/OBSERVATION_POLICY.md`**, `TODO.md`, `docs/HANDOFF.md`, **this file**, `docs/VISUAL_SELF_SUPERVISION.md`, `docs/COURSE_CORRECTION.md`, and `docs/FACTOR_POLICY_REVIEW.md`. Respect the newer instructions even if older headings below describe historical priorities.
3. Verify the supported graphics compute stack with `.venv/bin/python tools/gpu-preflight.py`. NVIDIA hardware OpenGL and CUDA were already working on an RTX 4070 SUPER; **don't reinstall Gentoo drivers or widen the Bubblewrap sandbox** based on an untested hypothesis.
4. **First new technical work:** isolate where per-game-tick time is spent, using small fresh-tagged no-neural Minecraft tests. The existing `MCioAgentBridge.apply` loops one send/receive per tick. Compare time in `send_action`, `recv_observation`, server synchronization, rendering, Python frame decode, and between-run startup/warmup as instrumentation permits. Sample the same world/viewport/action sequence across conditions. Review `tools/run-mcio-viewer.sh`, `src/gmaincraft/agent_interface.py`, `experiments/joint_log_factorized.py`, `experiments/factorized_controls.py` and MCio runtime logs. Do **not** assume Minecraft is CPU-rendering: NVIDIA GLX hardware rendering has already been verified.
5. Reproduce the slowdown **without PyTorch**. Confirm effect of launching through the nested Gamescope viewer vs a safe comparable configuration, and isolate Minecraft-versus-client/server pacing. Do not confuse a 90-second outer shell timeout or Gamescope's occasional post-child **139** with an unsuccessful Python experiment; inspect the child logs and durable outputs. Cleanup **only child processes owned by the experiment**.
6. Implement the smallest justified repo-scoped performance fix and remeasure **equal game ticks, same rendering settings, same seed/arena/action sequence**. Preserve correctness: truly synchronized vanilla input, RGB frames, player camera, real inventory events and no action/observation cheating. Update benchmarks/docs with the actual before/after comparison. If not resolved, clearly document the narrowed bottleneck and the next falsifiable experiment; **do not mark this TODO completed**.
7. Run the pure tests (`.venv/bin/python -m unittest discover -s tests -q`) and an appropriate short real-Minecraft smoke. Commit and `git push origin main` through the bridge, updating this file, `TODO.md`, `AGENTS.md`, `docs/HANDOFF.md` and experiment results. Use unique run tags and leave ignored worlds/datasets/checkpoints intact.

**Performance exit criterion:** reproducibly demonstrate a material throughput improvement on matched real-game measurements (and preserve correctness), ideally approaching the previous ~300 ticks / 0.6-0.8 s range. That historical number is a goal, **not a guaranteed entitlement**. If it is not technically attainable, provide an evidence-based explanation and explicitly ask the user whether to proceed with slower training; do not quietly switch priorities after a single inconclusive run.

## HISTORICAL SECOND: course correction principles still apply

After the throughput issue has been diagnosed and fixed or clearly quantified with the user deciding to proceed:

- Preserve the goal: a **from-scratch learning agent on unseen random
  Minecraft Java survival seeds**, using direct **player-knowable game-state
  memory reads in BOTH training and gameplay**. Numeric position, facing,
  health, inventory and stat counters are intended policy inputs; screen
  RGB/audio can supplement them, not replace them. See
  [OBSERVATION_POLICY.md](OBSERVATION_POLICY.md). The existing RGB-only
  `AgentView` is a **legacy prototype**, not permanent project policy.
- Engine-derived block/entity IDs and positions may be fed directly when
  **visible or legitimately remembered**. Research/calibrate eligibility
  thresholds (FOV, range, projected size, occlusion, lighting, raycast,
  interaction/reach) after the throughput issue; do not invent fixed numbers
  or expose a complete hidden map. Normal Minecraft actions and survival
  rules still apply.
- In training and scored play, the policy sees the SAME versioned bounded
  observation fields. Controller-only hidden state may separately calculate
  scalar rewards, reset worlds or score outcomes, but not become an oracle
  destination/secret world-state feature. No human demonstrations, externally
  pretrained models, or scripted speedrun standing in for a learned policy.
- The **first functional milestone** is actual unrestricted finding, mining **and collecting** a log, judged by normal Minecraft inventory. Do not rush to separate crafting/Nether/dragon subsystems while unrestricted first-log collection still fails.
- The current factorized policy has separate **6-tick V2** and **12-tick V3** schemas; real vanilla-mine event stats are evaluator-only, and `CumulativeMinedTracker` persists across resets. **Never** use contaminated `v3event01` as a valid checkpoint or evidence: cumulative stat reset falsely awarded multiple blocks. `v3eventfixed01` is the clean follow-up (10 training episodes; 3 truly mined; 0 inventory pickups in training). In frozen restricted stationary audit, old/new mined 4/4 with 0/4 vs 2/4 inventory pickups; both failed 0/4 fully unrestricted. New survival seed 31415927: 0/240-tick wood for random and V3.
- `docs/VISUAL_SELF_SUPERVISION.md`: own random-action screenshots + CUDA inverse dynamics learned to predict camera and attack, but not movement reliably when simultaneously turning. Camera-locked movement recovered ~38.5% vs a ~14.7% baseline. Causal GRU and frame-delta encodings exist as **offline diagnostics**, not working online memory. Test action disentanglement/history and sample-efficient learning against **same held-out real Minecraft cases**, not merely offline accuracy. Self-collected datasets, local `.pt` weights and world files are **Git-ignored**; do not regenerate/overwrite without need.
- Negative controls and proper validity are mandatory: initial inventory clear, paired scenarios/fixed spawn, held-out different scenario/world seeds, frozen inference, real inventory, checkpoint hashes, no manual command interventions during scored play. The simple normal-input control scored 8/12 versus V2 learned 0/12 on matched nearby log scenes, and the privileged geometry oracle 11/12 is **not** a valid pixel-only policy comparison.
- Optimize Minecraft **game time to win** eventually; faster computation is a tool, not the Any% objective.

## Fresh chat starter (copy-paste into a new conversation)

> Connect to Gentoo Shell `/workspace` and continue `polskiftw/gmaincraft`.
> Read `AGENTS.md`, `TODO.md`, `docs/NEXT_SESSION.md`,
> `docs/OBSERVATION_POLICY.md`, `docs/MEMORY_POLICY_V4.md`,
> `docs/MINECRAFT_RENDER_PERF.md`, and `docs/HANDOFF.md`. V4 memory+RGB
> from-scratch training/frozen natural-world evaluation now works but all
> completed V4 learning tests have 0 log pickups; do not blindly retrain.
> Natural-world MCio throughput improves ~2.68x with the temporary 4-chunk
> display profile; keep watching game I/O vs warmup. Preserve existing local
> checkpoints/worlds and restore any pending options backup. Next implement
> bounded, read-only **player-visible targeted-block**/player-status memory
> instrumentation within MCio/Fabric and experimentally derive a reliable
> visible/actionable world-object eligibility filter without hidden map
> cheating. Run small actual Minecraft tests and controls, update docs,
> commit/push source changes via Git in the isolated shell. Never use an
> already-used run tag or mix training and frozen evaluation in the same
> Gamescope session if possible.

**Bridge availability is per conversation:** a new chat does not automatically inherit tool authorization or the old chat's detailed working context. The repository records the project decisions; give the new chat the starter text and enable Gentoo Shell when needed.
