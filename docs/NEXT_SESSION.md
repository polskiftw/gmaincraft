# START HERE — next ChatGPT/Gentoo Shell session

## START HERE: natural survival training is NOW the main project

READ [`NATURAL_SURVIVAL_ONLINE_TRAINING.md`](NATURAL_SURVIVAL_ONLINE_TRAINING.md), `OBSERVATION_POLICY.md`, `AGENTS.md`, `TODO.md` first. The user explicitly said scripted/pre-aimed fixtures have diminishing returns and wants the AI ready for sustained autonomous **normal survival training**. That capability is now implemented and exercised end-to-end in live Minecraft. The trainable `natural_memory_online_v1_first_hit_objects_ppo6` neural policy starts random, chooses vanilla move/look/attack together using own memory and a 24-record bounded first-hit camera surface sampling, with no privileged world map, cheats, preaimed blocks, fixed routes or scripted phase handoff. A direct source-world seed generator and actual start/resume/clean-stop foreground shell helper exist; no daemon installed.

`tools/prepare-natural-training-worlds.py --tag X --count 8 --seed 917211` builds an independent cohort of vanilla noise survival sources. `tools/run-long-natural.sh start mylong X 100000` launches an unattended foreground trainer; from another terminal `touch logs/natural_online_mylong.stop` gracefully saves; `tools/run-long-natural.sh resume mylong X 150000` adds to the SAME model and evolving world saves. Three locally ignored checkpoint datasets were made: `natpilot_1103a` verified actual CUDA RNG restore/resume; `natreal_1103a` explore reward 480 decisions,12 updates,114 explored cells,5727 first-hit surfaces,40 blocks mined,wood0; `natwood_1103a` wood-sighting reward 720 decisions,12 updates,136 cells,7499 surfaces,366 new visible wood surfaces,21 blocks mined,**no wood**. The exact v2 `natwood` checkpoint is ready locally. Two new fully noise-generated Minecraft source seeds `freshworld1103` are also verified. Frozen evaluation on 2 unseen seeds: trained v2 **greedy** 2 cells,0 mined; v2 **sampled** 24 cells,5 mined; **uniform random** 28 cells,15 mined; **wood inventory 0 for all**. The architecture is a real learning platform but **NOT an effective wood finder yet**. Don't claim a natural or Any% win.

**Immediate next effort:** natural training sample efficiency/degenerate greedy actions and meaningful actual wood acquisition, comparing genuine held-out survival seeds to random. Follow user goal rather than another month of staged tree tests. Improve hard-crash model/world alignment; clean stop already works. Add calibrated in-view item-entity memory and new versioned motor controls only when they help genuine normal-world training. All worlds and checkpoint weights under ignored `mcio-data/`, `checkpoints/` and `logs/`, not Git. Existing old arena and mod backups preserved. Real 2-3 ms/tick on warmed natural NVIDIA scenes, original headless Gamescope teardown exit139 persists after successful child; launcher verifies journal before declaring success. Read dedicated doc for all caveats, output tags, commands and eval numbers.

## Prior: memory-visible dropped items and autonomous phase choice

Read [`COMPOSED_MINE_PICKUP_AUDIT.md`](COMPOSED_MINE_PICKUP_AUDIT.md),
[`MINING_FEEDBACK_AND_ATTACK_LEARNING.md`](MINING_FEEDBACK_AND_ATTACK_LEARNING.md)
and [`OBSERVATION_POLICY.md`](OBSERVATION_POLICY.md), then AGENTS/TODO.
The trained crack-memory attack motor and an older scratch-trained RGB
pickup controller now PHYSICALLY obtained the same mined oak log in real
Minecraft pre-aimed staged scenes. On 12 unseen held-out case jitter seed
784232: learned+pickup p02 11/12 **actual inventory wood**, forward9/12,
random7/12, no-attack0 and no-movement0. Independent case seed784233:
learned+p02 9/12, learned+p01 7/12, simple forward9-10/12; always-hold
attack performs as well as learned attack. Current observation is
**not** fully memory-based because pickup still uses historical RGB, and
controller PRIVILEGED PREP plus SCRIPTED MINED-EVENT PHASE HANDOFF persists.
Do not describe the result as natural first-log finding or an Any% win.

On-demand `MiningFeedbackRequest` integrated, current JAR SHA256
`28fd3209bd15c9fecd1c2cdb3828b8dca3c17ff0840ea97e57be1f6da0cc05e6`.
Old/new versioned models preserved ignored. 50 fresh crack readings in
300 ticks and 0 on all other packets, headless RGB continuous; same-JAR
wall time on-demand was NOT faster than continuous in the one-run pair.
Additional 13/13 real combined surface+mining request and 4/4 legacy
V5 crosshair-only tests passed under the latest patched mod. The exact
Minecraft packet inputs are opt-in and the original V5 model unchanged.
New independent from-scratch on-demand motor14/28 training, 12/12 frozen,
with rare out-of-range crack stage causing a FAILED separate early training
attempt (needs replication/instrumentation, not guesswork).

NEXT substantial work: inspect Minecraft client item entity API, implement
bounded **player-perceivable dropped item** observations with occlusion/
render-distance/FOV tests, and train one new-from-scratch shared-schema
policy to choose mining versus collecting without the evaluator handoff.
Continue real held-out inventory controls; no hidden world item lists,
no pretrained model, no admin gameplay, preserve all previous data.

## Earlier: mined progress from Minecraft memory and trained attack motor

See [`MINING_FEEDBACK_AND_ATTACK_LEARNING.md`](MINING_FEEDBACK_AND_ATTACK_LEARNING.md)
for complete steps, source hashes and real-game results. A separate
opt-in Java packet now reports **own visible crack stage**, attack active,
and vanilla targeting without hidden world data. Nine lab checks passed;
holding attack moves stage1..9 to true mined+1; release resets stage.
From random weights a narrow CUDA learned attack motor trained 28 genuine
six-tick Minecraft rollouts. Frozen held-out pre-aimed oak cases12/12
mined vs untrained0/12 and always-attack12/12. The learned motor is real,
but simple always-hold is equally good and **0 logs entered inventory**.
Future agent MUST solve independent discovery, approach/aim and actual
pickup. Next compare composed learned attack and existing learned pickup
against explicit controls on REAL mined items; then derive player-visible
item entities from game memory and integrate into joint policy. Keep
all prior worlds/weights, no controller admin commands during gameplay.

## Earlier handoff: robust visible-block calibration and three natural seeds

Read `VISIBLE_SURFACE_CALIBRATION_2026_11_03.md` and
`SURFACE_REQUEST_V1.md`. The first visibility experiment without a
server-settlement delay was INVALID; the later 23-scene paired rendered-RGB
comparison and three natural Minecraft noise seed controls are the
verified evidence. 13x9 default vs 25x17 research fine: coarse 13/14
strong image controls detected, fine 14/14, while fine STILL misses
some small visible blocks and can cost more on dense terrain. Exact
camera pitch +/-90 ray basis bug fixed, tested in live Minecraft.
Current independent tests: **92 pass**. Every new test/temporary world
was confined to the disposable isolated DemoInstance, preserved previous
JARs/checkpoints, no changed user world. The next substantial work is
FOV/projection/lighting/transparency/entity validation on natural scenes;
then bounded versioned policy features with identical train/frozen inputs
and renewed learned mining/pickup research. Do not blindly scale PPO.

## Earlier NEXT: calibrate view-eligible objects; on-demand solved

See `SURFACE_REQUEST_V1.md`. A fresh grouped camera-ray object sample is
now requestable on the last normal MCio action tick and absent otherwise.
Eight real bridge controls, 87 pure tests passed. Fixed-300-tick warm
measurements: no surface 0.7302s, 50 on-demand samples 0.7140s, prior
300 continuous samples 0.8898s, each in separate non-neural launches.
This is NOT a trained policy or validated human-visual object list.
Next derive true perceptual eligibility in natural worlds across dynamic
FOV, small distant blocks, transparency, lighting and entities; then design
the same bounded train/frozen neural input and improve mining/pickup.

## Earlier NEXT step: calibrate and optimize grouped surface sensor

The latest real Minecraft memory change is summarized in
[`SURFACE_SENSOR_V1.md`](SURFACE_SENSOR_V1.md). Two opt-in read-only camera
research streams (raw 13x9 first-hit rays, grouped block positions) passed
six controlled live comparisons and all 86 pure tests; prior V5 and
pristine JARs are backed up. **Not yet policy inputs.** Measured same-JAR
no-surface warm 300 ticks 0.6746s vs grouped 0.8898s on one fixed idle
scene, so optimize sampling frequency with a correctly synchronized
per-decision request before any extended RL. Then calibrate true visual
FOV/distance, fine objects, occlusion/transparency, light, entities and
non-fixture survival scenes. Maintain first-hit no-X-ray boundary and
versioned train/frozen parity. Do not claim learned log acquisition or
feed experimental surface data into existing V5 checkpoints.

**Priority order is intentional.** Updated November 2, 2026. This document
is the continuation contract for the next conversation. **The user wants
memory-derived structured game state**, not a permanently screenshot-only
agent; read [OBSERVATION_POLICY.md](OBSERVATION_POLICY.md). The memory-state
work follows **after** the performance regression investigation.

## Historical continuation point — V5 real crosshair status and independent performance causes

**This is the active priority for the next chat; older V4 and pre-V4 FIRST
sections below are historical.** The user wants a from-scratch Minecraft
Java Any% agent on unseen survival seeds, with direct **skilled-player-
knowable game-engine/memory observations** in BOTH training and gameplay.
We have now tested the new opt-in `gmaincraft_status_v1` Fabric MCio patch
for hunger/saturation/armor/air, dimension/slot and ONLY the vanilla
crosshair's real targeted block ID/pos/face/distance. Track source changes
as `patches/mcio-1.6.1-player-status.patch`, rebuilt from pinned upstream
by `tools/build-mcio-player-status.sh`; the original MCio game JAR is backed
up under ignored `mcio-data/mod_backups/` and can be restored. **Read
[`MCIO_PLAYER_SENSOR.md`](MCIO_PLAYER_SENSOR.md) BEFORE editing sensors.**

Four independent scene controls (oak, mouse turned away, restored aim,
stone occlusion) passed, plus glass and distance calibration: at measured
block-face 1.5/2.5/3.5 blocks the vanilla crosshair hit oak, at4.5–7.5 it
missed in this controlled scene. **Crosshair actionability != broader visual
perceptibility**. There is NO loaded-chunk world oracle or generalized
calibrated visible list. The larger FOV/occlusion/light/projected-size test
still needs implementation, with held-out mixed scenes and false-X-ray
negative controls. Keep a versioned bounded, train/eval-parity schema.

**V5 `factorized_v5_status_v1_player_v1_rgb2_6tick` is already integrated**
using 40 numerics, real full block-ID byte tokens, optional RGB84×2 and
994,686 random-initialized actor-critic weights. `joint_log_memory.py
--with-status --skip-eval` trained 4 actual CUDA Minecraft episodes /one
PPO update and saved a valid 12MB local checkpoint; **0/4 real pickups**.
`memory_frozen_eval.py --with-status` ran 8 actual independent frozen
cases, **0/8**. `natural_world_benchmark.py --modes status_factorized`
compared frozen V5 against random with identical spawn in seed31415927;
**0/180-tick inventory logs for each**. Later V5 frozen policy telemetry
showed oak under the crosshair on all30 decisions in stationary lab yet
attack_ticks=0; a near case saw bedrock29/30 and oak1/30. **Core next
learning problem is action persistence/camera strategy, not OCR for already
available own game state**. Do NOT blindly scale PPO from zero-success
smokes. Old V4 frozen checkpoint was rerun successfully with mod patched
but extension OFF; preserve all previous weights/worlds.

**Performance watchdog:** the independent natural-world 4-chunk display
profile has measured 2.68x benefit (see `MINECRAFT_RENDER_PERF.md`), AND
there is a **separate fluctuating 60-tick/s SDL Gamescope cap** (old original
MCio jar ~5s/300 ticks even with virtual Gamescope refresh300). Setting
`GMAINCRAFT_GAMESCOPE_BACKEND=headless` on `tools/run-mcio-viewer.sh`
restores ~0.77s/300 warm, with NVIDIA still rendering real game RGB offscreen.
Opt-in status ON vs OFF in patched mod measured ~0.723 vs0.714s/300
warm; no material regression seen in this small pair. If user wants to
WATCH the game window, omit backend variable for visible SDL but expect
conditional cap; never widen host sandbox. Log every experiment's tick I/O,
render profile and cold-start. `GMAINCRAFT_GAMESCOPE_REFRESH_HZ=300` is a
research control and did NOT fix SDL cap. Launcher wrapper exit139 usually
occurs AFTER Python child success; check durable checkpoint/JSONL, and keep
training and frozen eval in separate Gamescope sessions.

**Next substantial work:** investigate/calibrate player-*visible* blocks and
entities beyond the one crosshair hit, and test sample-efficient neural
memory/action-duration/persistence methods on paired held-out Minecraft
cases. Real unrestricted first-log mining+pickup remains unsolved; long-
horizon wood/crafting/Nether/End/Any% must wait. Always run pure tests, fresh-
tagged real-game short pilots and held-out controls, update `TODO.md` and
ledger, commit/push Git directly in `/workspace`. Do not ask the user to
recap or overwrite ignored saved worlds/weights.

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

> Continue `polskiftw/gmaincraft` in my connected Gentoo Shell `/workspace`.
> Read `AGENTS.md`, `TODO.md`, `docs/NEXT_SESSION.md`,
> `docs/OBSERVATION_POLICY.md`, `docs/MCIO_PLAYER_SENSOR.md`,
> `docs/MEMORY_POLICY_V4.md`, and `docs/HANDOFF.md`. The new, reversible
> opt-in MCio/Fabric patch delivers hunger/player status and exact VANILLA
> crosshair-targeted block ID/pos/face/distance. V5 consumes it with the
> same bounded schema in actual CUDA training and normal frozen Minecraft,
> but early V5 tests (4 training,8 frozen and paired natural survival)
> got ZERO real log pickups. Don't blindly scale PPO. Work NEXT on
> empirical *visible* block/entity eligibility beyond the one raycast hit,
> player-knowable block-breaking progress, and better action history/credit
> for continuous real vanilla mining+pickup. Keep player X-ray/hidden seed
> prohibited, real inventory scoreboard, held-out matched seeds.
> Performance: use `GMAINCRAFT_GAMESCOPE_BACKEND=headless` for accelerated
> NVIDIA+RGB MCio; visible SDL sometimes caps at60 ticks/s despite 300Hz
> virtual refresh. The reversible 4-chunk game profile also improves real
> natural-world FPS. Watch actual I/O and clean game launch/exit separately.
> Preserve every ignored world/checkpoint/model, use fresh run tags,
> update AGENTS/TODO/experiment ledger/docs, run real tests and commit/push
> through direct Git in the isolated shell. No confirmation needed.

**Bridge availability is per conversation:** a new chat does not automatically inherit tool authorization or the old chat's detailed working context. The repository records the project decisions; give the new chat the starter text and enable Gentoo Shell when needed.
