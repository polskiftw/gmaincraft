# Handoff for a brand-new ChatGPT conversation

## CURRENT: completed V4 long run, matched holdout and sandbox GPU blocker

Read [`docs/V4_24000_TRAINING_AND_HELDOUT_AUDIT.md`](V4_24000_TRAINING_AND_HELDOUT_AUDIT.md) first. The user finished `persistpilot1103` to **24,000 game-time units**, 40 genuine natural seeds and 138 updates: **3 real natural wood inventory pickups** during training (episodes 0,34,37), 419 ordinary blocks mined; no proven wood generalization. New same-game-time frozen eval on heldout indices4,5: **grown V4 74 blocks/0 wood/0 deaths, original 600-unit V4 16 blocks/0 wood/3 deaths, uniform random 40 blocks/1 wood/2 deaths**. Policy is concentrating on 72-tick holds but not consistently attacking visible wood. Sandbox CUDA cannot currently initialize (`/dev/nvidia-uvm` missing after 6.18.60 host reboot); optional `GMAINCRAFT_EVAL_DEVICE=cpu` completed all three frozen evaluations correctly. Original checkpoints retained, no new training started. NEXT: tackle player-visible wood targeting/persistent no-look mining in a fresh versioned run; never silently change V4 checkpoint metadata. See new audit for raw journal names.


## Latest milestone: autonomous natural-world wood acquisition with learned action duration

Read [`docs/NATURAL_WOOD_FIRST_LOG_PERSISTENCE_V4.md`](docs/NATURAL_WOOD_FIRST_LOG_PERSISTENCE_V4.md)
**first**. V3 `woodv3_01` finished24,000 real vanilla game-time decisions,
338 ordinary blocks mined, 2,895 new wood surface sightings, **0 wood
mined or inventoried**. A new V4 **four-head** from-scratch memory policy
lets the actor choose ordinary move/look/attack **and 6/24/72 tick hold**;
it receives NO scripted target/mining phase and the same bounded train/frozen
player-visible memory. World budgets still measure six-tick GAME-TIME units;
long actions use real 6-tick reward sampling and timed PPO discounting.
In its very first real 600-unit/3,600-tick pilot `persistpilot1103`,
the actor autonomously mined wood at unit355 and acquired the mined item
in actual inventory at unit361; one real natural seed, 4 CUDA updates.
Frozen at these weights on FOUR distinct unseen matched seeds: **1/4 real
wood inventory** vs 0/4 uniform-random duration control. V3 frozen on
the first two seeds at equal game-time budget: 0/2. Sample is too small
to prove reliable learned skill, and random broke more nonwood blocks.
V4 start + real resume, separate held-out indices2/3, old V3 frozen
checkpoint, exact schema protection, pure **134 tests** all passed.
Preserved original successful 600-unit model hash in ignored
`checkpoints/fresh_stream_holdv4_pilot600.pt`.

**NEXT user training:** `tools/run-fresh-treadmill.sh resume persistpilot1103
24000 600 wood_persistence_v4` (600 horizon MUST remain; 24,000 is a
TOTAL target). The original model continues, frozen 600-unit snapshot
is intact. Compare full V4 learning curve against the historical snapshot
and same-seed uniform-random controls on several more unseen seeds. Do not
claim Any% or consistent wood pickup; actual dropped-item entity sensor
and crafting remain unsolved. Only use `/workspace`, preserve ignored
checkpoint/world data, update docs/tests and commit/push validated changes.


## November 3 update: speedrun01 completed, mining-action collapse measured

Read [`docs/SPEEDRUN01_POLICY_COLLAPSE_2026_11_03.md`](docs/SPEEDRUN01_POLICY_COLLAPSE_2026_11_03.md)
for the committed 24,000-decision/10-seed result, the matched frozen
policy-vs-random Minecraft baseline and the newly versioned `wood_progress_v3`
training profile. `speedrun01` is preserved as its own original v2 run: 106
ordinary blocks mined, 759 visible wood surfaces, **0 wood mined/picked**.
Frozen policy (two heldout Minecraft seeds, 480 decisions) chose jump-forward
452 times and attack only 12, mined 0 blocks; matched random mined 2.
Added non-policy per-head/crosshair/mining action telemetry, a conservative
one-time player-visible wood crack-stage reward, and lower learning rate plus
higher entropy regularization ONLY for new v3 runs. An independent 96-decision
actual vanilla Minecraft v3 smoke saved a valid CUDA checkpoint. **129 pure
tests pass.** Next user run is a NEW from-scratch experiment:
`tools/run-fresh-treadmill.sh start woodv3_01 24000 2400 wood_progress_v3`.
Do NOT silently resume `speedrun01` with a changed objective; use original
v2 settings to resume v2, or begin the separately named v3 comparison.
After v3 training, compare on matched unseen seeds and inspect logged
attacks-on-wood, sustained camera-still mining, real mined wood and inventory.
No evidence of a general self-directed first log yet.


New optional scoped NVIDIA mitigation: fresh treadmill now launches
Minecraft JVM with `__GL_THREADED_OPTIMIZATIONS=0` by default, reversible
via `--nvidia-gl-threading default` for direct Python evaluation. Verified
12-decision real Java Minecraft smoke `nvidia_gl_safe_1103` completed
without a crash; not proof of fixing the user's prior NVIDIA SIGFPE.

## LATEST handoff: crashed speedrun01, repaired watchdog, checkpoint safe

Read [`NVIDIA_STARTUP_CRASH_RECOVERY.md`](NVIDIA_STARTUP_CRASH_RECOVERY.md)
first. User's first real 24k run finished exactly12k decisions/5 seeds
and125 updates, then NVIDIA SIGFPE in native GL driver on sixth seed.
Python MCio sync recv ignored timeout, hanging while holding flock; new
sandbox could not see original host process. That flock has subsequently
released; no lock files were deleted, last checkpoint unchanged. Added
real 50s-startup/30s-observation watchdog, bounded retries, original
checkpoint/RNG restore, owned aborted seed quarantine, exact-settings
recovery, early flock preflight. New actual full Minecraft smoke
`watchdog_smoke_1103` recovered from early rc=0 JVM exit and finished12
real decisions+one CUDA update. This does NOT prove driver crash fixed.
User can resume `tools/run-fresh-treadmill.sh resume speedrun01 24000`;
no system services or auto-long training launched. 127 tests pass.

## LATEST: automatic fresh survival worlds (not eight-world repetition)

**FIRST READ [`FRESH_SEED_TREADMILL.md`](FRESH_SEED_TREADMILL.md).** The
project now has a tested FOREGROUND endless fresh-seed training treadmill:
`tools/run-fresh-treadmill.sh start TAG forever`, `status TAG`, `stop TAG`,
`resume TAG forever`. One scratch CUDA actor learns on never-repeated
signed-long seeds (full-width invertible keyed mixing, 2^63 train and 2^63
disjoint heldout eval seeds per root). Newly generated vanilla survival
worlds are **not just copies with an altered NBT seed**; Minecraft's real
server generates all terrain. The runner automatically removes old
completed owned worlds after checkpoint, keeping last2 and quarantining
aborted ones. All player-owned memory reward/obs remain identical per
train/frozen seed, and per-world novelty resets as expected. No model
weights or world seed are supplied to policy except its own observations.

Fresh program `experiments/fresh_seed_treadmill.py` and source/world
helpers have pure tests and REAL game smoke tests: 4 old pilot worlds
and 3 new seed-format worlds, in-game clean stop, model+optimizer resume
on next seed, deliberate interrupted world quarantine/rollback, real
held-out frozen test versus a uniform-random policy. Old matched-seed
control with default `spawnRadius10` had different starting coordinates,
so it is INVALID as a strict matched-start test. Current generator sets
vanilla spawn radius0 without changing noise terrain. Actual v2 fixed
spawn heldout same seed and start `(48.5,64.0,32.5)`: frozen1 ordinary
block mined, random0, both0 wood (12 steps, too few for performance claim).
**118 pure tests passed**. Current Minecraft mod JAR and historical
checkpoints untouched. Runtime proof: fresh worldgen6.6-7.4 seconds plus
client load9.4-14.3 seconds per seed; no safe in-process world reset in
MCio's current global sync design. 2,400 steps/seed default to amortize.
Next improve actual natural wood acquisition and heldout long-run
learning curves, develop item-visible memory, separately explore hot
world reload, and DON'T return to repeated preaimed tree fixtures.

## LATEST verified: ready-to-run natural survival training

Start with [`NATURAL_SURVIVAL_ONLINE_TRAINING.md`](NATURAL_SURVIVAL_ONLINE_TRAINING.md), not preaimed experiment notebooks. User now prioritizes **leaving a from-scratch agent training in real survival worlds** instead of more scripted controlled scenes. The new **memory-only 50,913-param** PyTorch actor controls normal movement, camera and attack simultaneously at six-game-tick cadence, learns from own-player state, crosshair/visible crack stage and 24 bounded first-hit camera block surfaces with registry-ID byte embeddings; no whole-world X-ray, admin commands, fixed tree, directed phase handoff or pretrained model. Each natural-world save evolves uninterrupted across training segments. Independent source seed cohort generator, verified live train/resume/clean-stop helper, periodic after-world-save checkpoint of network+optimizer+RNG+per-world exploration/event memory, append-only journal, frozen evaluation and random/untrained controls all exist. No daemon was started. User commands and details in new doc.

Actual real Minecraft tests: `natpilot_1103a` started20 and resumed to40 successfully after CUDA RNG restore fix; `natreal_1103a` 480 decisions/12 updates,114 new map cells,5727 new first-hit surfaces,40 ordinary blocks, **0 logs**; `natwood_1103a` from scratch to720 decisions/12 updates across 3 natural seeds,136 cells,7499 surfaces,366 genuine previously unseen wood surfaces,21 nonwood mined, **0 logs**. Fresh source cohort `freshworld1103` contains 2 new real noise seeds; all local copies and ignored checkpoints preserved. Held-out v2 frozen on unseen seeds: greedy2 cells/0 mined, sampled24 cells/5 mined, uniform random28 cells/15 mined, all 0 wood. The random baseline currently outperforms learned exploratory and mining behavior; do NOT overclaim. Minecraft sometimes gives `isDestroying=true` crack=-1, now explicitly typed as *unknown progress*; avoid fabricated stages. More testing and natural-world RL improvement, hard-crash alignment and player-visible entity sensor are next. Old staged composed skill 11/12 is historical and not general natural capability.

## Prior: frozen learned mining AND pickup of real physical drops

**START with [`COMPOSED_MINE_PICKUP_AUDIT.md`](COMPOSED_MINE_PICKUP_AUDIT.md).**
Two independently trained, frozen Minecraft policies have for the first
time combined to break actual pre-aimed oak via opt-in *own* crack progress
and then obtain an actual mined drop in normal survival inventory, with
NO spawned items or admin commands during play. 12-case held-out seed
784232 results: learned attack+frozen pickup p02 **11/12 inventory**,
always-attack+pickup11/12, forward9/12, random7/12, no-move0,
no-attack0. Replicate on seed784233: learned+p02 9/12, learned+p01 7/12,
forward9-10/12. **NOT a complete trained general agent**: evaluator
places/aims the trunk BEFORE each trial and SWITCHES TO PICKUP AFTER a
real mined counter advances. No learned tree finding, aiming, phase
selection, unknown world navigation or Any%. **Stat deltas corrected**:
MCio sends cumulative counts, not per-episode, so persistent per-world
trackers are essential; earlier pilot event DELTAS were invalid.

The final mod also passed **13/13** simultaneous fresh surface+mining
last-action-request checks and **4/4** older V5 crosshair controls with
new research sensors OFF. Raw reports `surface_mining_joint_nov03d` and
`mining_req_compat_nov03d` under ignored `logs/`.
New opt-in `MiningFeedbackRequest` on-demand CBOR, one fresh own crack
observation at each six-tick policy decision, no stale skipped packets.
It reduced records 300->50/300 ticks but same-build warm timing 0.7527s
status-only, 0.7915s continuous, 0.8055s on-demand: NOT a wall-time
performance win yet. Old frozen attack policy preserved 12/12; independently
trained on-demand model14/28 training, 12/12 frozen. An earlier on-demand
training attempt aborted on an out-of-range crack stage; unresolved rare
edge case, preserve failed data and investigate before changing type
bounds. **100+ pure tests** must pass after next changes. All old Git-
ignored worlds/weights and pristine+prior Fabric JAR backups preserved.
NEXT: add a *bounded on-screen visible dropped-item memory sensor*,
measure occlusion/FOV and light; then train one versioned from-scratch
policy that chooses attack/approach/collect without a scripted transition.

## Earlier: learned vanilla mining motor, no pickup yet

Read [`MINING_FEEDBACK_AND_ATTACK_LEARNING.md`](MINING_FEEDBACK_AND_ATTACK_LEARNING.md)
FIRST. The pinned, reversible Fabric patch now exposes read-only THIS
player's `gmaincraft_mining_feedback_v1` break animation stage, opt-in
via `GMAINCRAFT_MINING_FEEDBACK_V1=1`; no hidden-world positions. Real lab
9/9 confirmed stage1..9, reset and actual mined stat+1. A separate
**from-scratch CUDA attack motor** over six own-memory features, choosing
ordinary `LEFT_BUTTON` hold/release every six game ticks, was trained on
28 real Minecraft cases, with 3/14 first-half vs 10/14 second-half true
break events. Frozen 12-case evaluation: learned12, untrained0, random0,
no-attack0, always-hold12. **This only matches the simple always-hold
baseline and is pre-aimed. 0 learned inventory pickups; no natural Any%.**
Checkpoint `checkpoints/mcio_attack_motor_motorlearn_nov03a.pt` is LOCAL
Git-ignored, SHA256 in the detailed document. Prior V5 models, previous
mod JAR versions and all disposable worlds kept. Main next work: compose
this learned attack with the existing scratch-trained RGB pickup for a
real mined-drop **inventory** audit, then transition pickup to a calibrated
memory-derived visible item sensor. Longer-term build versioned V6 joint
world navigation/aim/action/pickup with same train/frozen observation schema.

## Earlier handoff: 23-scene RGB/memory calibration plus natural tests

Read **`VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`** FIRST. The previous
on-demand sensor is still opt-in and read-only, never injected into V5.
An early apparent no-detection result was INVALID because actual Minecraft
server block updates lag MCio frames: the new lab waits for server
updates and confirms unchanged-before/after scene stability. In 23 paired
real rendered-image tests, 13x9 reported 16 targets and 25x17 reported
18, including recovering a 182-pixel off-center block. Glass/leaves still
produce false negatives (not no-X-ray leaks). Three real noise world
source seeds, paired normal RGB/idle actions, no frame gaps and variable
warm throughput are recorded. Exact vertical camera pitch +/-90 now works:
12/12 live controls; Java source compiled, 92 pure tests passed.
Current installed isolated mod SHA256 is documented in the study. Preserve
previous backed-up mod versions, all Git-ignored worlds, logs, datasets and
checkpoints. NEXT: derive **actual** perceived-block/entity eligibility,
calibrate FOV/transparency/light/distance in natural moving scenes, then
schema-tagged V6 same-memory train/frozen and real sustained mining+pickup.
Default 13x9, 25x17 only explicit research option. No general first log.

## Earlier: real tested on-demand surface observations

See `SURFACE_REQUEST_V1.md`. Grouped first-hit samples can now be requested
only on the last tick of a normal 6-tick action, rather than every tick.
Eight real Minecraft bridge checks and 87 Python tests passed. Warm fixed
300-tick no-neural timing: 0.714s with 50 requests, 0.7302s disabled on
same JAR, 0.8898s prior always-on. No training/policy integration yet.
Next: rendered-visibility calibration, natural-world throughput and
sustained learned mining/pickup. Preserve all ignored data and old JARs.

## Earlier handoff: bounded off-crosshair surface-object experiments

**Start at [`SURFACE_SENSOR_V1.md`](SURFACE_SENSOR_V1.md).** The isolated
Fabric patch now optionally sends first-hit 13x9 camera rays and grouped
block-position objects (IDs/positions/sample rays/faces/distance/light).
It detects off-center oak and stops at stone/glass, preserving no-X-ray
behavior in six live tests; group records exactly match all original rays.
These are **research-only, not part of V5 or a trained V6**, and lack proper
transparency, daylight/fog, dynamic FOV, projected-size and entity
calibration. A genuinely visible oak behind glass is conservatively missed.
Matched 300 game tick idle warm results: no-surface current JAR 0.6746s,
raw-ray v0 previous JAR 0.9497s, grouped current JAR 0.8898s. Current
full-time sensor is too expensive; **next implement research into on-demand
per-policy-decision rays, tune real rendered eligibility and add temporal
mining feedback before building V6**. Never bulk world-scan; preserve all
checkpoints/worlds. Exactly 86 pure tests passed. New patch JAR SHA256
`8d1152104e6646c59f963456b2d3c243b6c87b0e4b6d2ca9773b8fbb88a34fb3`.
Versions of the pristine, earlier status and first raw-grid mod were saved
in ignored `mcio-data/mod_backups/`. Real logs under ignored `logs/`; viewer
139 after child SAVED remains an unrelated teardown issue.

**Last verified: 2026-11-02.** Canonical code: [polskiftw/gmaincraft](https://github.com/polskiftw/gmaincraft), branch `main`. **Historical early milestone (superseded by the sections below):** learn movement W/A/S/D from real RGB features (2 independently trained models; fresh fixture audit p01 **30/32**, p02 **24/32**, always-W 21/32); then **combine frozen RGB aim/ATTACK and pickup policies** on *real mined logs* (two independent camera-angle sets, trained p02 **51/56**, hardcoded W **49/56**, untrained movement 21/56). Earlier numerical-navigation and RGB log-aiming milestones remain preserved; `docs/EXPERIMENTS.md` records all results. The initial uncleaned aim 12/14 and aborted first drop-distance audits are **invalid evidence**. No general world-seed Minecraft speedrunning. Do not ask for old test logs.

## Historical handoff: patched MCio real block status, V5 parity, headless speed

Read `MCIO_PLAYER_SENSOR.md` FIRST for source patch/build/install/revert,
exact tests, performance A/B and V5 train/eval commands. Exact upstream MCio
1.6.1+1.21.3 Fabric checkout is under ignored `mcio-data/mcio-upstream-inspect`;
tracked patch `patches/mcio-1.6.1-player-status.patch` adds a READ-ONLY
opt-in `PlayerStatusOption`. `tools/build-mcio-player-status.sh --install`
backed up original JAR under ignored `mcio-data/mod_backups/` and installed
new patched jar **only** in disposable `DemoInstance`; `--restore` reverts.
Patched jar SHA256 `fff86beea0da5c66dc1e78ee8ae450260dace9d5f112120f9c6088388e77ffbb`,
original `21368effcb3b579954deef345c4a8d965ebb667d82bab25723626da406224ab3`.
Extra fields emitted ONLY with Minecraft env `GMAINCRAFT_STATUS_V1=1`.
Register Python CBOR type BEFORE receiving packet. The actual vanilla
`Minecraft.client.hitResult` supplies targeted block ID/pos/face/distance,
not any hidden/chunk list. `PlayerStatusV1` adds hunger/armor/saturation/air,
hotbar slot and dimension. Reversible installation, no host settings changed.

Real-scene sensor controls (two independent smoke worlds) oak/turn away/
restore/stone occlude **4/4 each**. Additional reach-calibration test:
front faces 1.5/2.5/3.5 blocks all target oak; 4.5/5.5/6.5/7.5 all miss.
Glass intercepts vanilla crosshair ray even though transparent. These are
**not calibrated visual FOV/size/range thresholds for a visible-object list**.

Actual V5 schema `factorized_v5_status_v1_player_v1_rgb2_6tick`: 40
numeric own+crosshair fields, full128 UTF-8 byte IDs with trainable char
CNN, 2×84×84 RGB, 994,686 random initialized model weights. Same
`MCioAgentBridge.multimodal_status_view()` during real CUDA training and
normal frozen scoring; controller-only geometry stays scalar reward, never
policy input. `statuspilot_1103a`: 4 train 180-tick episodes,1 CUDA update,
**0/4** logs, checkpoint SHA256
`77b24ce058fda616eb143432a9ae1c52782e8bd7dfdda0ce0afc353dba59d190`.
`statusfrozen_1103a`: 8 scored held-out cases **0/8** pickups.
`statuscounts_1103b`: 4 further held-out cases (0/4), stationary agent saw
actual oak ID on all30 decisions but **attacked0 ticks**. Aim oak3/30,
near oak1/30 + bedrock29/30, broad miss30/30. Natural seed31415927:
random 0 logs/180 ticks and frozen V5 0/180 with identical spawn
`(-15.5,76.0,16.5)`; normal game actions, no admin edits during play.
Original V4 checkpoint retested with new mod feature disabled: 4/4
independent frozen cases completed, **0 logs** as expected. NO first-log
skill, no Any% win, no pretrained/human policy.

Critical performance insight from matched NO-TORCH original mod runs:
Gamescope **SDL visible backend conditional ~60-tick/s cap** despite
MCIO_UNLIMITED_FPS and advertised virtual refresh300: ~5.0s/300. SAME
unpatched jar via `GMAINCRAFT_GAMESCOPE_BACKEND=headless` recovered
**0.770s/300 warm**. Patched jar opt-in status OFF vs ON had
**0.714s vs0.723s warm**, tiny extra overhead. Minecraft still renders RGB
through NVIDIA inside nested offscreen Gamescope. Omitting backend var keeps
visible SDL for user's watching but sometimes caps throughput. Separate
4-chunk natural-world render distance lever remains ~2.68x over original
defaults in earlier measured idle natural seed. Keep exact profile/viewport
identical for comparisons, monitor each run and warmup, don't change host
settings. Gamescope wrapper139 AFTER child success remains; use separate
viewer sessions for train/eval, preserve original ignored worlds/weights.

Next: derive bounded *camera perceivable* block/entity candidate set with
empirical FOV/occlusion/light/projection and false-X-ray tests, then solve
sample-efficient continuous vanilla targeting/attack/pickup with reproducible
new/held-out seeds and negative controls. More PPO episodes without success
evidence are NOT an acceptable shortcut. Commit source/docs only; normal Git
push via Gentoo Shell available.

## CURRENT handoff (Nov 2, 2026): V4 actually trains on memory + RGB

Read `MEMORY_POLICY_V4.md` and `MINECRAFT_RENDER_PERF.md` before further
Minecraft work. Current opt-in *V4 `player_knowledge_v1` + RGB* agent passes
own position/facing/health/inventory/stats numerically during both actual
Minecraft CUDA training and normal frozen evaluation. Same validated
`MCioAgentBridge.multimodal_view()` path in both. New V4 feature schema
`factorized_v4_memory_v1_rgb2_6tick`/20 floats/976,430 parameters, separate
source modules/scripts and checkpoints; all V1/V2/V3 source and weights
remain preserved. `mempilot_nov03b` 4 train +3 frozen cases: no pickups;
`memfast_nov03c` 10 train +2 updates, 0/10 inventory and 0 true mined
events; separate `memfrozen_nov03e` 12/12 completed frozen evaluations,
**0/12** pickups. Natural seeds31415927 and246813579 each compared random
vs frozen V4 for180 ticks, all 0 logs at identical starts. Do NOT treat
these runs as learned tree-finding or even reliable log mining. SHA of
`memfast_nov03c` checkpoint c57d5aba99010a704fc332f5f5a7bf1dc598c8de6224a037613531781a30014f.

A natural-scene no-Torch random control also reproduced ~12.96 ms/tick
where warmed flat worlds were ~2 ms/tick: problem is not PyTorch. A
controlled **600 idle-tick natural-world seed246813579** test under
unmodified MCio options measured 4.535s vs 1.695s with
`renderDistance:4`, `enableVsync:false`, `maxFps:260` (2.68x). Intermediate
6 chunks 2.590s, 8 chunks 3.387s, uncapped alone 4.265s; cold first100
is substantially slower. V4 new training defaults to the 4-chunk LEGAL
Minecraft rendering mode; it reduces visible range, so paired training and
scored play must explicitly match it. The project `temporary_game_options`
restores the original bytes; no permanent host modifications. See
`MINECRAFT_RENDER_PERF.md` for caveats, timings, source/seed parity.

Gamescope can assert `vulkan_remake_swapchain()` and abort when launching a
second Minecraft in the same viewer (this happened immediately after
`memfast_nov03c` completed valid 10 episodes and saved its checkpoint).
One orphaned Python child was terminated via SIGTERM and it restored options.
Use `joint_log_memory.py --skip-eval`, then **new**
`memory_frozen_eval.py --model-tag TAG` in a separate viewer wrapper. The
first 12-case standalone eval succeeded. Do not confuse wrapper 139 after
child completion with failed Python tests. Preserve all local ignored runs.

**Actual next research gate:** installed twoturtles/MCio 1.6.1 source
(at ignored `mcio-data/mcio-upstream-inspect`) exposes own player state,
inventory and stats, but not player hunger or targeted block/entity, nor
perceivable block lists. Implement safe versioned engine-read status and
vanilla crosshair raycast first, then derive bounded block/entity eligibility
via real-game camera/frustum/occlusion/reach testing on held-out scenarios.
No hidden world map or privileged target as policy input. When that sensor
works, compare new state, RGB-only and numeric-ablated frozen agents under
identical settings and natural seeds; then pursue real unrestricted first-log
FIND/MINE/COLLECT, not massive unproductive PPO runs.

## Latest verified update: performance probe + structured player knowledge

`docs/MCIO_THROUGHPUT_2026_11_02.md` documents a new no-Torch real-game
probe (`experiments/mcio_throughput_probe.py`) with four unique disposable
worlds/launches and 10 measured 300-tick repeats. First each launch ~1.17–1.24
seconds; after warm-up **0.617–0.721 seconds** per 300 ticks, overlapping
old fast results. All tests used normal vanilla attack/pickup; each repeat
collected exactly one real log, with action/observation sequence parity and
no dropped frame sequences. The old ~4-second slowdown is therefore NOT
currently reproducible; its root cause remains unknown. Explicit FPS cap
change gave no reliable advantage and original instance options were fully
restored. Gamescope wrapper still reports post-child exit 139.

New `player_knowledge_v1` in `src/gmaincraft/player_knowledge.py` and
`MCioAgentBridge.player_knowledge()` offers immutable opt-in numeric
player position/facing, health, owned inventory and available own progress
stats. The 66-test pure suite and a real synchronized 300-tick smoke verified
memory packet/inventory agreement. **No trained policy consumes it yet**;
legacy RGB AgentView and saved checkpoints remain unchanged. Next: train/eval
parity wiring for a new from-scratch model, then scope and validate perceived
block/entity candidate filters (not raw world X-ray). Reinvestigate I/O if
historical regression returns; no host-level changes justified.

## Canonical user intent: direct bounded Minecraft memory observations

**Read [OBSERVATION_POLICY.md](OBSERVATION_POLICY.md).** The user corrected
our interpretation: the learner SHOULD receive **game-engine/memory-derived
stats and counters directly during both training and gameplay**—including
player position/facing, health, own inventory, progress and bounded visible
block/entity state. A skilled player could know these facts; the agent
should not have to infer them all from RGB pixels. Optional video/audio
remain useful but are not the primary required input. The CURRENT RGB-only
`AgentView` is a historical assistant choice, **not desired final design**.

Memory reads are expressly allowed; the scope boundary is what a player
could perceive or remember, not whether information came from game memory.
Unobserved hidden ore/structure maps and unvisited goal coordinates remain
outside the agent's knowledge budget. Visible/actionable-block eligibility
(FOV, range, occlusion, size, lighting, reach) is a FUTURE calibrated research
task; do not choose arbitrary thresholds now. The game controller may still
use hidden state for fixture reset and scalar training rewards. Scored
policy gameplay must use ordinary Minecraft actions, with the same bounded
structured observations it had during training.

**User's retained work order:** first diagnose/repair MCio throughput;
next expose player-knowable memory state and calibrate block visibility;
then resume from-scratch learning of unrestricted first-log acquisition,
followed by eventual Any%. Preserve old RGB results as diagnostics.

## HISTORICAL FIRST ACTION — superseded by the CURRENT V4 handoff

**The user wants the MCio performance regression investigated and fixed
FIRST. Read [`NEXT_SESSION.md`](NEXT_SESSION.md) before doing any model
experiments.** Do not resume training simply because the latest model had
restricted mining success. The current 300 game tick episode takes ~3.9-4.0
seconds (MCio game send/receive ~3.876s, CUDA inference ~0.09s, RGB resize
~0.015s), versus historical ~0.6-0.8s. A no-Torch control is ~5.008s;
small viewport/chat/6-vs-12 intervals did not restore speed. Root cause
not yet known, and historical comparisons require matched reruns. Trace
and reproduce the bottleneck first, report grounded before/after metrics,
preserve normal Minecraft sync and safe restricted sandbox. Use Git directly
from connected Gentoo Shell `/workspace`; code, docs and files are already
pushed in prior work. Then build **bounded, direct memory-derived** player-state
observations for training and gameplay and resume the corrected from-scratch
Any% course and real unrestricted first-log acquisition; details below.

The very first `## Newest milestone` paragraph above is **historical** and
predates V2/V3. Current next actions are ONLY the first-priority notes in
this section and `docs/NEXT_SESSION.md`/`AGENTS.md`/top of `TODO.md`.

## MOST RECENT HANDOFF — clean RGB, self-learning, stats repair, V3 (2026-11-02)

Read **`docs/VISUAL_SELF_SUPERVISION.md`** first for complete verified
findings, file paths, exact experiment tags and next research gate. Latest
code is currently on Git main (check `git status`/`git log`); use connected
Gentoo Shell for direct Git and real NVIDIA RTX 4070 SUPER CUDA tests. Do
not replace random-weight learner with human/external pretrained models.
No autonomous first-log ability yet.

The agent's 84x84 RGB captured huge red admin-reset chat: fixed by writing
`chatVisibility:0`, `chatScale:0.0` only in dedicated local MCio DemoInstance.
**Never** set `chatVisibility:2`: it consistently caused teleported yaw/pitch
to report zero despite correct position and would contaminate all episodes.
Real clean screenshot probe exists under Git-ignored `artifacts/`.

New randomized self-action RGB collection and CNN inverse-dynamics training:
13 valid train episodes / 507 frame-action pairs, 4 held-out episodes /136
pairs; camera42.6% vs majority23.5%, attack84.6% vs52.2%, movement15.4%
vs14.7%. Camera-fixed movement-only 38.5% vs14.7%. Separate causal GRU
reached 20.6% movement, 44.9% camera,81.6% attack on same joint validation,
**not integrated in online agent**. Raw self-trained visual encoder may
initialize V2/V3 via `--vision-init-tag`, no downstream benefit verified.

V3 separately schema-tags 12-tick decisions (V2 remains 6), actual GPU
`v3clean01` interrupted at 15 recorded episodes, valid checkpoint after
14, limited stationary training wins. **INVALIDATE `v3event01`:** cumulative
stat cache reset caused fake 1+2+3+4+5+6 event counts. Preserve invalid
checkpoint for forensic history but never warm-start/claim improvement.
**Correct FIX:** privileged Minecraft `minecraft:mined` and
`minecraft:picked_up` event cache updates every tick; shared
`CumulativeMinedTracker` persists between repeated episode bridge resets.
Real negative control `statfix01` gave **mine/no-attack/mine =1/0/1**.
Corrected `v3eventfixed01` from prebug clean checkpoint, 10 new CUDA train
episodes, 3 real mined blocks, **0 collected**. Paired frozen restricted
stationary: both mine4/4, prebug clean picks0/4 vs corrected picks2/4;
full unrestricted near remains 0/4 for both. These tests have different
training budgets, so don't attribute gains to event reward alone.
Natural seed31415927 paired spawn 0 wood for random and V3 in240 ticks.

**Unresolved performance:** 300 game ticks now ~3.9 s inside MCio action
send/receive, vs earlier ~0.6-0.8. CNN inference0.09s, resize0.015s,
no-neural control ~5s, small viewport and chat setting don't fix. Do not
speculate about driver, weaken sandbox, or run very large experiments.
Bridge shell actions time out at90sec and Gamescope wrapper often segfaults
139 AFTER child finishes; inspect actual JSONL/checkpoints and cleanup only
owned orphaned processes. Prior saves/weights and local frames all ignored.

**NEXT:** scientific review of action/visual representation + history for
unrestricted tasks, then real hold-out Minecraft first-log acquisition and
continuous wood-to-crafting; avoid adding many standalone toy subskills.

## CURRENT handoff: V2 factorized policy and scientific course correction

Read `docs/FACTOR_POLICY_REVIEW.md` and the top of `TODO.md` first. This
session implemented `vision_factorized.py` (separate CUDA-capable factorized
CNN), variable stationary/aim/near/broad head-masked curricula with
interleaving, fixed an inventory-reset race, verified continuous short hold
mining in Minecraft, and compared REAL 12-case baselines. Frozen models
`factorv2pilot01`, `attackrl01`, `aimnear01` all scored **0/12** on the
identical near seed3737 scenarios, versus scripted 8/12 and PRIVILEGED
oracle 11/12. `attackrl01` learned attack-only in restricted aligned fixture
(2/4 held-out) but failed when movement/looking were unlocked. Sequential
and interleaved curriculum did not fix unrestricted learning.

`natural_world_benchmark.py` now evaluates `--modes joint_factorized
--factorized-model TAG`. Real 1.21.3 seed246813579, 3 independent same-
spawn modes (random, forward_attack, V2) all 0/360ticks natural wood. This
proves test wiring only. `rollout_store.py` saves ONLY policy RGB/action/
reward/value/terminal data to Git-ignored `runs/`; `recordpilot01` real
CUDA pilot produced and successfully read 40 actual transitions.

Next: investigate image representation, action credit, memory, and
sample-efficient algorithms rather than burning huge CUDA time on zero-win
PPO. Run `tools/gpu-preflight.py` if CUDA disappears after user reboot.
The bridge `run_shell` has a 90-sec hard timeout; do short experiments and
check JSONL/checkpoint when Gamescope exits 139 or times out after child.
Git works inside `/workspace`, commit and push there. Never alter ignored
prior world saves/checkpoints/SSH secrets; avoid broad system mounts.

## Newest: CUDA training operational after bridge restart

CUDA now actually WORKS: UVM devices visible, `cuInit(0)=0`, Torch CUDA
allocation passed. A 25-step synthetic CNN batch-64 model benchmark measured
~6.82x GPU versus CPU update throughput. Real Minecraft `cuda01`: 12 joint
PPO training episodes / 3 GPU PPO updates, 4 frozen eval episodes, 220 ticks
each, 0/12 train logs and 0/4 eval logs. Both local model and JSONL saved;
Git-ignored. Frozen policy exhausted episodes in just 3 long actions,
indicating potential fixed 80-tick attack-hold macro exploitation; **next
investigate before large GPU experiments**. Gamescope post-child exit 139
remains. Historical CUDA-unavailable remarks below are now superseded.

## Latest checkpoint continuity check (2026-11-02)

`warmcheck01` successfully loaded the old joint model, trained one more
100-tick **CPU** Minecraft episode with 23 transitions, ran PPO's second
update and saved a valid **atomic** 11 MiB checkpoint with optimizer state.
No log acquired; this used repeated scenario RNG, so it is just an interface
smoke, not independent training progress. New code rejects source train/policy
seed repetition. For future work use `--init-model OLD_TAG` **with a new
`--run-tag`, `--train-seed` and `--policy-seed`**. This warms weights and Adam,
but is not an exact interrupted-episode restore. Preserve existing models
and follow `docs/JOINT_LEARNING.md` rather than rerunning a used tag.

## Newest handoff: joint visual PPO and RTX GPU, 2026-11-02

Source additions: `src/gmaincraft/vision_ppo.py`, `ppo_math.py`,
`joint_curriculum.py`, `experiments/joint_log_ppo.py`, optional
`requirements-gpu.txt`, C GLX probe, GPU preflight, CPU/GPU training
benchmark. Read `docs/JOINT_LEARNING.md` and `docs/GPU_ACCELERATION.md`
first. Joint agent uses two RGB84 screenshot frames, 888,236-parameter
CNN/PPO, 11 vanilla timed macros including a **fixed 80-tick mining hold**.
Not trained to craft, explore general terrain, or speedrun. Verified
CPU real-game pilot `jointsmoke01`: 2 training+2 heldout Minecraft log
scenarios with PPO gradient update, 0/2 logs each. Frozen natural-seed pilot
`jointtransfer01` seed 161803398, 0/120 ticks; no errors in Python child.
These runs establish pipeline only; large-scale RL research remains pending.

RTX 4070 SUPER hardware OpenGL **confirmed** via nested XWayland GLX probe
(`GL_RENDERER=NVIDIA ...`, 4.6 NVIDIA). Torch 2.15.0+cu132 installed only
into project `.venv` and works on CPU, but CUDA still **unavailable inside
current bridge**: `nvidia_uvm` host module loaded, no bound `/dev/nvidia-uvm`,
`cuInit(0)=999`. Host needs UVM node creation and bridge restart; instructions
in `docs/GPU_ACCELERATION.md`. Do not claim real GPU training completed.
Use `--device cuda` for fail-fast once devices are visible, verify actual
allocation first with `tools/gpu-preflight.py`. No modifications to host
launcher, Gentoo packages or broader sandbox permissions were made.

## NEW course correction (2026-11-02)

Read `docs/COURSE_CORRECTION.md` and the highest-priority block of `TODO.md`
first. Existing trained aim/pickup models remain valuable *diagnostic* baselines
but have not transferred to general Minecraft. New source modules include
`agent_interface.py` and `natural_eval.py`, plus the real seeded-world runner
`experiments/natural_world_benchmark.py`. A pilot natural Minecraft seed
314159265 was generated and run: 0 logs for idle, random, forward+attack and
frozen `legacy_probe` within 100 ticks; no Python/game test error. Pilot
spawn positions differed (original source default spawnRadius=10); **do not
claim paired benchmark validity**. New `--prepare` fixes source spawnRadius=0
and verifies natural noise terrain + exact worldgen seed. A SECOND real-game
smoke test on new seed 271828183 ran idle and forward+attack for 25 ticks
per mode; **both actual start positions matched exactly (0.5,77,0.5), yaw 0,
pitch 0**, and both recorded zero logs. This verifies paired starting pose
only for these two cases; longer multi-seed testing remains open. Gamescope still reports exit
139 after Python child completes. Use normal `git` from Gentoo Shell on
`/workspace`, commit/push; never expose scoped private Git credentials.

## First actions for the next assistant

1. Read `AGENTS.md`, `TODO.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/EXPERIMENTS.md`, inspect relevant code/tests and recent GitHub commits.
2. Recognize the last verified milestone: all three camera-relative numerical policies scored **50/50 after training and 50/50 on an additional unseen target/yaw set**. Each trained for 300 real Minecraft episodes with distinct policy/train/eval/extra/action seeds. Each training CSV shows **299/300 wins**.
3. **NEXT:** the separately learned W/A/S/D pickup skill is DONE (fresh fixture audit 30/32 p01, 24/32 p02 vs W21/32), and combining frozen aim+pickup collected real mined logs (p02 51/56, W49/56). **Variable trunk/player positions and an initial shared visual PPO controller now exist**, but only zero-win smoke tests completed; next train/evaluate actual improvements against baselines in independent natural worlds. Preserve old checkpoints. The user does NOT want human-looking camera behavior; odd gaze is fine if vanilla crosshair/reach and observation rules are followed. Retain no-attack/dropped-item cleanup safeguards.
4. Later remove privileged numeric goal-direction inputs for visual/perceivable goals. Independently seeded *Minecraft worlds* remain untested.
5. Commit code, tests and all status docs together to GitHub, then give the user safe `git pull --ff-only` commands. Preserve local ignored checkpoint and world artifacts.

## Real Minecraft orientation evidence (2026-11-02)

| Run | Old fixed yaw | Old random yaw | Fresh before | Fresh after 300 episodes | Extra unseen target/yaw set | Learning time |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 50/50 | 24/50 | 9/50 | **50/50** | **50/50** | 30.16s |
| `rep_b` | 50/50 | 16/50 | 7/50 | **50/50** | **50/50** | 26.67s |
| `rep_c` | 50/50 | 19/50 | 7/50 | **50/50** | **50/50** | 25.17s |

Seeds (policy/train/eval/extra/action): reference 2718/81003/60211/99501/771; `rep_b` 2719/81004/60212/99502/772; `rep_c` 2720/81005/60213/99503/773. Local-only valid outputs: `checkpoints/orientation_policy.npz`, `checkpoints/orientation_policy_rep_b.npz`, `checkpoints/orientation_policy_rep_c.npz`, three corresponding `logs/orientation_training*.csv` files (300 rows each) and three disposable `MDL_OrientationLab_v1*` worlds copied from the same `DemoWorld` template. All model weights are finite (4,3) matrices. This is **not** general Minecraft world-seed success, no end-to-end speedrun, no visual recognition, no camera-turning action.

`experiments/orientation_learner.py` now supports `--run-tag` with write-once world/log/model filenames, and `--policy-seed`, `--train-seed`, `--eval-seed`, `--extra-seed`, and `--action-seed`. Existing default run behavior remains unchanged. Five new seed/name tests supplement the five existing numerical policy tests.

## NEW learned RGB movement and integrated mining milestones (2026-11-02)

- New sources: `src/gmaincraft/log_pickup.py`, `experiments/log_pickup_learner.py`, `experiments/log_pickup_evaluator.py`, `experiments/log_mine_and_pickup_audit.py`, `tests/test_log_pickup.py`; common MCio action wrapper now maps W/A/S/D and attack. Python project unit suite expanded to **24 passing tests**; real-GPU Minecraft trials completed in this same MCP workspace.
- Two new independent 150-episode movement trainings: RGB screenshot → **14 handcrafted oak-item features** → random-weight 4×14 softmax network → real W/A/S/D actions. `p01` seeds policy/train/eval 4401/6601/8801, 123/150 training pickups, 14.59s learner time; `p02` seeds 4402/6602/8802, 109/150 training pickups, 15.56s. Training item is **controller-summoned as an in-game dropped item**; the subsequent corrected independent audit uses the updated zero-initial-Motion fixture; controller shapes scalar reward from item spawn position, which policy NEVER sees.
- **Clean independent spawned-item audit:** world `MDL_DropAudit_dclean02`, seed 8804, 32 trials/controller, no-movement negative 0, untrained 2/32, random 16/32, always W 21/32, trained p01 **30/32**, trained p02 **24/32**. `logs/drop_pickup_audit_dclean02.csv` 160 rows (ignored). First `dclean01` audit is INVALID/incomplete due conservative distance-from-moving-item guard; do not quote it as success. Reset kills old items before each test.
- **Combined actual block mining + policy-chosen collection:** pretrained `rgb02` selects camera/ATTACK from screen, fixed 100-tick Minecraft left-click mine, then frozen `p01`/`p02` chooses W/A/S/D from screen. **No scripted collection movement in neural-controller modes.** Two fresh test sets: seed 8816 (24 cases), 8817 (32 cases), all controller commands restricted to reset, score is real mined oak_log inventory. Over **56 identical cases per controller**: untrained movement **21/56**; scripted W **49/56**; trained p01 **43/56**; trained p02 **51/56**. No-attack negative control yielded zero in the second audit. In this fixed centered-trunk setup even trained p02 mostly chooses W and its 2-win edge over scripted W is **not a meaningful proven advantage**.
- **NOW:** initial variable-trunk/player RGB joint PPO prototype exists, 0/2 held-out successes after just two training episodes. Next evaluate actual improvement over baselines, stronger visual perception and general survival on distinct random-world seeds. Detailed data/limitations: `docs/EXPERIMENTS.md`.

## NEW learned RGB camera/attack milestone (2026-11-02)

- Two fresh 500-episode Minecraft training runs: `experiments/log_visual_aim_learner.py`, policy seeds 5501/5502, train seeds 7723/7724, original camera eval seeds 8813/8814, learner time 18.01s/28.71s. Models 5×12 linear softmax initialized from random weights; actions look left/right/up/down/ATTACK. Features come from **actual RGB screenshot via engineered oak-brown mask/spatial bins**; no target coordinates or player yaw/pitch reach the policy. Controller shaping uses **scalar privileged angular-progress reward** (not fair reward in free-form Minecraft). In the **original camera-only evaluation**, after agent-chosen ATTACK the harness held left button and moved forward with **scripted W pickup**. The newer integrated evaluator (above) replaces that scripted movement with a separately trained neural pickup policy.
- **Important measured evaluation bug:** initial soft resets cleared player inventory but left *dropped items from earlier trials*. Thus 12/14 initial scores (and nonsensical untrained 11/14) are INVALID. The reset now sends **controller-only** `kill @e[type=minecraft:item]` before restoring the target/teleport; real item drops from the current ATTACK are still obtained normally.
- Independent, **clean audited frozen evaluation** uses fresh copied `MDL_LogAimAudit_clean01`, new angle seed 8815, 20 cases/model, no-attack negative control 0, untrained model 0/20, trained `rgb01` **15/20** and trained `rgb02` **18/20** real oak-log inventory acquisitions. `logs/log_aim_audit_clean01.csv` (60 rows) and two 5x12 checkpoints are **ignored local files**. Trained models are saved in `checkpoints/log_aim_policy_rgb01.npz` and `log_aim_policy_rgb02.npz` (see ledger for SHA-256). Both original numerical orientation/navigation checkpoints remain unchanged. At this **earlier camera-only stage**, general visual perception and pickup movement were unlearned; the newer learned pickup and integrated evaluations are reported in the section above. Independent world seeds, general tree recognition and speedrunning are still untested.
- Local one-shot command for another **new** audit tag: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_evaluator.py --run-tag audit02 --models rgb01 rgb02 --eval-seed 8816 --eval 20 --visible`. This creates only a disposable copied world, never overwrites checkpoints, and performs zero training. Continue with ordinary Git operations inside the connected shell.

## Updated interaction curriculum and honest status

- **User camera rule (2026-11-02):** camera pose/appearance should never receive human-likeness rewards; the agent may stare at sky or move sideways/backward, but must still use vanilla first-person line-of-sight, crosshair reach and mouse-based interactions where needed.
- `experiments/log_interaction_probe.py`: **VERIFIED scripted vanilla log mining/pickup in actual Minecraft**, not RL. In a *new disposable copied world* (`MDL_LogInputProbe_pilot01`), the controller placed a log column, then no commands were sent after staging. MCio mouse deltas +120,+50 pixels changed yaw +18.00° and pitch +7.50°, inverse deltas restored camera. Holding left attack 100 ticks then walking forward 11 ticks yielded **1 real oak log** in observed inventory; 20.15s total launch+test. Gamescope exit 139 occurred only after printout. No `/give`. Four new pure-helper tests were added.
- **DONE (bounded scope):** two models learned camera-turn and ATTACK choices from screenshot color features (15/20, 18/20 versus untrained 0/20); **now a separate learned RGB-based pickup policy also exists** and an integrated camera-ATTACK-plus-movement audit works on real mined logs (p02 51/56 across two angle seeds). **FOLLOW-UP:** new variable-log joint CNN/PPO exists as an unproven prototype; do not claim tested general tree skill from its zero-win smoke tests. This is NOT an independent-world Any% agent.

## Runtime and local environment

- Gentoo Linux/OpenRC, Plasma Wayland, i7-14700KF, RTX 4070 SUPER driver 595.104.02, 32GB RAM, Python 3.14.10.
- Host project `~/Documents/minecraft-discovery` is exposed as `/workspace` through a custom Gentoo Shell MCP Bubblewrap sandbox. Host `/home` remains **not exposed**. Network namespace is now intentionally **shared with host** for GitHub HTTPS/SSH access, including possible LAN and host-loopback connectivity. Repo-specific SSH write deploy key under ignored `mcio-data/git-auth/` allows normal shell `git push`; HTTPS fetch also works. GPU/DRM devices, selected read-only sysfs and one host Wayland socket are available.
- MCio Python controller `mcio_ctrl==1.5.2` with `cbor2==5.9.0`, Minecraft Java **1.21.3**, Fabric MCio **1.6.1+1.21.3**, Java 21 MCio runtime. Instance/template are `mcio-data/instances/DemoInstance` and `DemoWorld`. MCio ZMQ ports 4001/8001 require a working private localhost `/etc/hosts`.
- Inside MCP, Minecraft uses nested NVIDIA Gamescope/Xwayland. **Use `tools/run-mcio-viewer.sh`**, which builds the SDL2 no-host-mouse library and filters real mouse motion, clicks and scrolling (not MCio agent actions), keeping your pointer free. Example unique disposable test: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/orientation_learner.py --visible --run-tag viewer_demo --episodes 8 --eval 5`. Local Xwayland wrapper has `-ac` only for its private X11 socket. SDL2 simulated-input test and Minecraft MCio smoke 10/10 passed; physical desktop pointer hover verification is pending. See `docs/DEVELOPMENT.md`.
- **Gamescope exits SIGSEGV 139 after each completed Python run**; test results/checkpoints/CSVs were written and validated first. Direct MCio NVIDIA headless without Gamescope failed with GLFW/EGL in this sandbox.
- All checkpoints, saves, Python virtualenvs, local MCP launcher helpers, huge raw logs and model files are git-ignored; only source/docs/tests belong on public GitHub. Never access other host folders, alter global Gentoo settings or add a daemon for these experiments.

## Scientific boundaries

- Privileged arena controller may use `/tp`, `/fill`, `/setblock`, `/kill`, `/summon` **only for setup/cleanup**. Earlier numerical navigation policy receives a hidden goal vector; the newer **learned camera/ATTACK and W/A/S/D pickup policies receive only handcrafted RGB pixel features** and *privileged scalar training reward*, no player/item world coordinates. Controller randomizes camera on reset; neural aim policy controls camera through real MCio mouse inputs during play. Fixed 100-tick mining hold and the two-stage handoff are still scripted scaffolding.
- The 3 independent seed sets are different *numeric RNG seeds*, not different Minecraft world seeds. All three use a small flattened arena. These are strong curriculum checks but do not prove survival, perception, crafting, Nether/End progression or cross-world generalization.
- A soft reset does not fully restore block/entity/RNG world state. Preserve saved checkpoints and `DemoWorld` across future experiments; avoid overwriting known-good assets.
