# AGENTS.md — project instructions for coding assistants

## NEWEST verified milestone: real crack-stage feedback and learned attack motor

**Start with [`docs/MINING_FEEDBACK_AND_ATTACK_LEARNING.md`](docs/MINING_FEEDBACK_AND_ATTACK_LEARNING.md).**
An opt-in direct game-memory observation now exposes THIS player's visible
vanilla mining crack stage and whether attack is active, with no hidden
block position or world oracle. Real Minecraft staged oak was seen to
progress 1..9, reset to inactive on release, and subsequently increase
actual mined stats. 9/9 live checks passed. Existing V5/Fabric flags and
legacy models were not changed. A narrow CUDA actor-critic attack-only
model learned from scratch on 28 actual Minecraft stationary oak cases,
with real mined blocks in 3/14 early vs 10/14 late cases. A separate
12-case frozen test gave learned **12/12 mined**, untrained **0/12**, random
**0/12**, never-attack **0/12**, and the always-attack baseline **12/12**.
This is a learned attack motor skill, NOT a learned general tree finder,
item pickup or Any% result. The player inventory stayed 0 after mined logs.
Next: test composed learned attack + independently learned pickup on real
mined items, then add read-only player-perceivable **item-entity** memory,
not an unseen world map, for future train/frozen parity and navigation.
Keep all worlds/weights/checkpoints Git-ignored; update TODO, experiment
ledger and handoff on any substantive work. Previous research on 13x9
versus 25x17 rays remains valid, no default silently changed.

## Earlier verified milestone: independent rendered-image surface calibration

Read **`docs/VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`** and
`docs/OBSERVATION_POLICY.md` FIRST. A real Minecraft evaluator-only
paired-RGB/engine-ray test uncovered and fixed asynchronous server fixture
updates (earlier no-detection data invalid). **23 settled positions**
confirmed that the 13x9 ray sensor missed visible off-center blocks;
explicit 25x17 fine samples improved strong changed-pixel recall from
13/14 to 14/14 in one lab sweep but STILL misses smaller candidates.
Stone occludes targets, glass/leaves create visible false negatives.
Three distinct natural noise seed sources were checked with matching
300-tick RGB and no admin play actions, all frame/action sequences correct;
fine throughput is scene-dependent. The Java vertical camera basis was
fixed for pitch +/-90 and passed 12/12 actual bridge/surface controls.
**92 pure tests pass.** The new full study and original ignored run tags
are documented; NONE of these scene samples are yet train/frozen policy
inputs. Default remains backward-compatible 13x9; 25x17 is research-only.
NEXT: natural FOV/lighting/occlusion/entity calibration, exact same bounded
memory in future V6 train/frozen, action-duration and mining progress, real
first-log inventory success. Preserve old binary backups, worlds, models.

## Earlier milestone: synced on-demand grouped first-hit surface samples

See [`docs/SURFACE_REQUEST_V1.md`](docs/SURFACE_REQUEST_V1.md). The opt-in
Fabric patch supports request-on-final-normal-action-tick samples, with
none on unrequested ticks and no stale world data. Eight real bridge checks,
87 pure tests, and a no-neural fixed-fixture 300-tick comparison passed.
Warm on-demand 50 samples/300 ticks took 0.7140s; same current mod with
sensor disabled 0.7302s; older continuous grouped 0.8898s (single
launches, not statistical guarantees). Keep V5 checkpoint and its player
status/crosshair unchanged. Further sensor *visibility calibration*, then
train/held-out parity for new model + learned sustained mining and pickup
are the next priorities. Keep all previous worlds/checkpoints/backup JARs.

## Earlier experimental milestone: grouped camera-ray surfaces (not V6 yet)

See **`docs/SURFACE_SENSOR_V1.md`** before modifying game memory or scaling
training. An opt-in, read-only 13x9 camera first-hit ray grid and a bounded
**block-position-grouped** format are installed only in the disposable MCio
DemoInstance, with ALL prior mod JARs backed up. They report off-crosshair
wood, preserve stone/glass occlusion and have verified one-to-one actual
Minecraft first-hit ray parity (6/6 live controls, 86 pure tests). This is
**NOT** a calibrated player-visibility detector, does NOT handle entities,
transparent surfaces or perceptual lighting, and is NOT fed to any V5 model.
A same-jar fixed-fixture warm baseline took 0.6746 s/300 neutral ticks vs
0.8898 s grouped (raw v0 0.9497 s under the earlier JAR). The next gate is
visibility calibration AND efficient on-demand sample cadence before V6
train/frozen parity and genuine unrestricted first-log learning. Do not
claim grouped samples are rendered pixels or learned success. Preserve
current ignored runs and weights; real test outputs in local `logs/`.

This file is the operating contract for anyone (including a future ChatGPT conversation) editing **polskiftw/gmaincraft**. Treat the GitHub repository, not the conversation transcript or an assumed local checkout, as the authoritative codebase.

## Historical verified milestone: V5 patched MCio, true crosshair/player status and headless speed

**Read `docs/MCIO_PLAYER_SENSOR.md` and `docs/NEXT_SESSION.md` FIRST.**
The user explicitly permitted changing the isolated, disposable MCio game
instance and asked that we monitor performance continuously. V5 now uses a
**real, opt-in, read-only Fabric MCio patch**: hunger/saturation/armor/air,
selected slot/dimension, and ONLY the vanilla Minecraft crosshair's actual
block ID/position/face/distance. This is not a loaded-chunk/X-ray map. The
tracked patch `patches/mcio-1.6.1-player-status.patch` is built from pinned
upstream `twoturtles/MCio` tag 1.6.1+1.21.3 using
`tools/build-mcio-player-status.sh`; original mod is backed up and restorable.
The binary itself, vanilla world saves, private credentials and checkpoints
remain Git-ignored. Do not remove old models or blindly overwrite the mod.

**Training/inference parity is REAL:** new schema
`factorized_v5_status_v1_player_v1_rgb2_6tick` trains from randomized
weights with player+status memory and optional RGB; frozen controlled and
natural-survival modes consume precisely the SAME status view in normal
Minecraft. Real 4-episode CUDA training/one PPO update produced 0/4 logs;
8-case independent frozen score was 0/8; natural seed31415927 paired random
vs V5 was 0/180-tick logs each. In a follow-up four-case frozen audit,
stationary model saw oak on 30/30 policy decisions but chose **zero attack**;
it is not a learned log-mining agent. Current priority is calibrated
perceivable (not only targetable) object state and sample-efficient action
credit/duration, before another expensive PPO scaling attempt. All earlier
V1-V4 checkpoint families are preserved and the V4 frozen compatibility
smoke passed with the patched mod installed but status flag OFF.

**New performance root cause identified:** SDL Gamescope can impose ~60
Minecraft ticks/s EVEN when MCio/FPS/VSync are uncapped; setting its virtual
refresh to300 did not help. On the SAME unmodified MCio jar a 300-tick
benchmark changed from ~5.00s under SDL to **0.77s warmed** under
`GMAINCRAFT_GAMESCOPE_BACKEND=headless`, with real RGB still rendered on the
NVIDIA GPU via nested Xwayland. The patched sensor ON vs OFF warm runs
were **0.723s vs0.714s** /300 ticks, a small observed overhead. For
high-throughput training, set `GMAINCRAFT_GAMESCOPE_BACKEND=headless` on
`tools/run-mcio-viewer.sh`; for watching gameplay, omit to retain visible SDL
with protected host mouse. The separately measured render-distance benefit
(4 chunks vs defaults) also remains relevant on natural worlds. Continue
logging startup vs active I/O, per tick, scene complexity and profile;
never claim all prior slowdowns have only one cause.

**Safety:** only edit/test inside `/workspace` and the user's disposable
DemoInstance. Original MCio jar available through
`tools/build-mcio-player-status.sh --restore`; do not alter host packages,
filesystem mounts, drivers, user game worlds or other service processes.
Before using the opt-in packet, register `gmaincraft.mcio_status` with CBOR
BEFORE first observation, and set game process env `GMAINCRAFT_STATUS_V1=1`.
Never let admin reset geometry or global chunk data reach the playing policy.

## Canonical design: direct Minecraft state, bounded by skilled-player knowledge

**Read [`docs/OBSERVATION_POLICY.md`](docs/OBSERVATION_POLICY.md) before making
any sensor, agent-input, fairness or training decisions.** The user explicitly
wants **direct read access to Minecraft memory/engine state during BOTH
training and gameplay** for facts a skilled player could know. Examples:
player XYZ and camera direction, health/hunger, inventory, item counts,
player statistics, and suitably filtered observable block/entity IDs.
Screenshots and audio are OPTIONAL useful perception, not a required path
for every fact. **Do not force the project back into RGB-only learning.**

Bound the supplied world information, **not the method of reading it**:
never give an unobserved hidden ore/stronghold map or unopened container
contents as an oracle. Determining which blocks are perceptible/actionable
and calibrating range, occlusion, angle and confidence are EXPLICIT FUTURE
RESEARCH TASKS, not fixed thresholds to guess today. Structured memory
observations should be available in the same way during training and scored
play. The present `AgentView` still returns only pixels: a **legacy
implementation limitation**, not the project design.

**Priority order unchanged:** fix/quantify MCio tick-throughput regression
FIRST, then build a versioned structured game-state observation interface,
then pursue visibility-threshold research and integrate learning toward real
unrestricted first-log success. Never damage old saved experiments or quietly
pretend the new sensors are already implemented.

## Latest verified milestone: V4 memory policy + controllable natural-world speed

**Read `docs/MEMORY_POLICY_V4.md` and `docs/MINECRAFT_RENDER_PERF.md` FIRST.**
A new opt-in **V4 `player_knowledge_v1` + RGB neural controller** genuinely
consumes player XYZ/facing/health/inventory/stats in BOTH from-scratch
Minecraft CUDA training and frozen Minecraft evaluation. It has a separate
checkpoint schema, 20-float numeric encoder and 6-tick factorized actions;
legacy V2/V3 checkpoints are untouched. Real 10-episode V4 training and
12 fresh frozen trials worked with **0 log pickups**, so do not claim general
learning. Paired natural seeds 31415927 and 246813579 also scored zero.

**Performance discovery:** natural-world Minecraft with default render
distance can take ~12–13ms per game tick even with a no-neural controller.
Controlled 600-tick idle real-world comparison on seed246813579 showed
**4.535s default versus 1.695s at renderDistance4 + VSync off/FPS260**;
8 chunks 3.387s, 6 chunks 2.590s. The first 100 ticks are significantly
slower than later ones. This is a genuine reproducible graphics-setting
performance lever, **not proof** that all older 4-second runs share a single
root cause. `src/gmaincraft/instance_options.py` temporarily overrides only
the disposable MCio instance and restores byte-exact prior options. New
V4 training defaults to 4 chunks. This reduces visible distance—record
profiles and match train/scored comparison settings.

**Gamescope caveat:** the second Minecraft launch inside one viewer sometimes
asserts `vulkan_remake_swapchain()` and aborts the wrapper; an abandoned
Python child needed SIGTERM before options were restored. For new V4 work
run `joint_log_memory.py --skip-eval` and call `memory_frozen_eval.py` in a
**separate** viewer session. The old usual 139 after successful child exit is
also common; always examine durable JSONL/checkpoint/child success. No
host GPU driver, filesystems, or sandbox mounts were changed.

**Next real research gate:** direct read-only *perceivable* block/raycast
state and player-owned missing status in MCio, then empirically calibrate
visibility/LOS/occlusion/range on held-out scenes before exposing bounded
block lists. The installed twoturtles MCio 1.6.1 observation packet does
not currently provide block IDs/hunger; source was inspected in ignored
`mcio-data/mcio-upstream-inspect`. Do not give unobserved hidden chunks.
Then test paired actual unrestricted first-log FIND/MINE/COLLECT with frozen
models and RGB/memory ablations. Do not burn thousands of PPO episodes
without evidence of improved real-scene success.

## Live update: recovered fast MCio ticks and opt-in player-state memory

`docs/MCIO_THROUGHPUT_2026_11_02.md` records four actual no-Torch
Minecraft launches, including one with the opt-in `player_knowledge_v1`
adapter: ~0.62–0.72 s/300 ticks after warm-up, real vanilla log pickup,
no missing frames. First repeats are ~1.17–1.24 s. The historical ~4 s
regression was **not reproduced**, not conclusively fixed; retain it as an
open root-cause question if it returns. Do not change host drivers or
Minecraft options on speculative grounds. The new
`src/gmaincraft/player_knowledge.py` reads own XYZ, camera, health,
inventory, known stats directly from MCio. Legacy RGB `AgentView` is
untouched; **new memory view has NOT yet been wired into training/eval**.
Calibrating block visibility and identical train/eval policy inputs are next.

## Historical MCio speed-regression runbook (now measured/narrowed)

**Start with [`docs/NEXT_SESSION.md`](docs/NEXT_SESSION.md), not a new PPO
run.** The user explicitly requires **diagnosis and remediation of the
Minecraft/MCio game-step throughput regression BEFORE resuming AI training**.
Earlier normal real-game runs took ~0.6-0.8 s/300 ticks, current ~3.9-4.0
s/300; profile showed ~3.876 s inside MCio game send/receive, ~0.09 s CUDA
inference and ~0.015 s RGB conversion. A no-Torch control is also slow;
small viewport, 6-vs-12 game-tick action intervals and changing chat scale
did not fix it. The comparisons must be matched before claiming causality.

**Order of operations:** connect to Gentoo Shell `/workspace` -> inspect
repo/docs and GPU preflight -> reproduce/trace slow per-tick Minecraft I/O
with minimal no-ML control -> fix/quantify and rebenchmark under matched
conditions -> update evidence/docs/tests -> only THEN implement player-knowable,
engine-memory-backed structured observations and investigate perceptible
block-state thresholds -> resume integrated from-scratch RL toward real
unrestricted first-log mining + inventory pickup. Do not spend hours on model training first.
If root cause is unresolved, preserve precise profiler results and request
a user decision before accepting the slower speed as the new baseline.

Use direct `git` from the shell (including commit/push), keep all ignored
worlds/checkpoints/data unchanged, and never widen the restricted sandbox or
modify host drivers without demonstrated need and user agreement. New
conversation must have the Gentoo Shell bridge connected/authorized; do not
claim it is connected if it is not. Update `TODO.md` and `docs/HANDOFF.md`
with each verified milestone. See `docs/VISUAL_SELF_SUPERVISION.md` for valid
vs contaminated experiments; **`v3event01` is INVALID evidence**.

## Historical November 2 research snapshot — visual experiments and mining events

**Read `docs/VISUAL_SELF_SUPERVISION.md` BEFORE the next feature or training
run.** Connected Gentoo Shell `/workspace` has GPU CUDA and direct Git;
keep using the shell, not the GitHub connector. All previous ignored worlds,
checkpoints, datasets and user files must remain intact. The active mission
remains from-scratch learned Any% on unseen Minecraft Java seeds, not scripted
completion. **No unrestricted first-log skill has been demonstrated.**

The original actual 84x84 RGB observations contained large **red MCio
command-chat overlays**. `vision_settings.ensure_clean_visual_options` now
atomically sets `chatScale:0.0` with `chatVisibility:0` in dedicated local
`DemoInstance/options.txt` before MCio launch: keeps HUD and normal camera
controls. Do NOT use `chatVisibility:2`: Minecraft teleports then consistently
reports pitch/yaw=0, invalidating seeded resets. `sendCommandFeedback false`
did not fix error chat. The user's own Minecraft options are untouched.

New `experiments/visual_dynamics_collect.py` captures the agent's OWN random
normal Minecraft action/RGB pairs only, locally, with distinct train/validation
scenario+action RNG seeds. 13 valid training episodes (507 before/after pairs)
and 4 held-out validation episodes (136 pairs) were obtained; collector
initially planned 14 train episodes but timed out after 13, so **do not claim
14**. New fully random-weight CUDA inverse visual predictors achieved held-out
**camera 42.6% (majority 23.5%)**, attack **84.6% (baseline 52.2%)**,
movement **15.4% (baseline 14.7%)**. Camera-locked movement-only dataset
achieved **38.5% vs 14.7%**; a causal GRU action-effect diagnostic achieved
movement20.6%, camera44.9%, attack81.6% on the mixed dataset. **Offline
self-supervised diagnostics are NOT yet a recurrent RL policy or first-log
success.** `--vision-init-tag` in V2/V3 training supports self-collected
raw-RGB CNN initializer, not outside pretrained models; downstream learning
advantage remains unverified.

Minecraft mining statistics (`minecraft:mined` and `minecraft:picked_up`) now
have private evaluator caching at EACH MCio game tick in `MCioAgentBridge`;
policy `AgentView` stays only RGB. **CRITICAL BUG HISTORY:** `v3event01`
training result INVALID: per-episode stats cache began at 0 and then received
cumulative absolute mine counts, producing fake 1+2+3+4+5+6=21 mined counts.
DO NOT treat `v3event01` as clean evidence or warm start from it. Fixed with
`CumulativeMinedTracker` scoped to a shared world arena; verified actual
mine/no-attack/mine **1/0/1** delta across resets in `statfix01`. A clean
`v3eventfixed01` warm-start from prebug checkpoint `v3clean01` ran 10 CUDA
training episodes, **3 real mined oak blocks, 0 inventory logs**. Frozen
held-out paired restricted stationary audit: `v3clean01` mined4/4, picked0/4;
`v3eventfixed01` mined4/4, picked2/4. Fully unrestricted near is still 0/4
for both. Different training budgets prevent causal attribution to reward.
New natural seed31415927 with identical spawn: random and frozen V3 each
0/240-tick logs; pipeline verified, no general-world transfer proven.

`--decision-ticks 12` is separately schema-tagged V3 (6 is V2); old V1/V2
checkpoints remain supported. `v3clean01` started 28 episodes but timed out
after 15 recorded; LAST CHECKPOINT was after 14. Never claim 28 complete.
An unexplained MCio I/O **slowdown to ~3.9s per 300 game ticks**, independent
of Torch, viewport, chat scale or 6/12 action interval, is under investigation.
Profiling shows MCio game I/O, NOT GPU inference (~0.09s). Do not blindly
change host services, driver, sandbox mounts or GPU settings. When 90-second
bridge calls time out, inspect valid JSONL+atomic checkpoints and terminate
ONLY orphaned processes owned by the experiment.

**NEXT:** pursue evidence-backed sample-efficient visual motion modeling,
credit assignment and genuine unrestricted first-log acquisition. Compare
against matched scripted 8/12 controlled-near baseline, then unseen genuine
natural survival seeds; no infinite PPO blindly. Use own RGB/action data,
not human/external pretrained models. Document every result in the ledger,
commit+push code/docs, NEVER Git-add local models or world saves.

## Historical November 2 research snapshot — factorized CNN and baselines

**Read `docs/FACTOR_POLICY_REVIEW.md` BEFORE any more training.** V2 is a
new, checkpoint-INCOMPATIBLE 888,750-weight CUDA RGB PPO actor with 3
categorical heads (movement/look/attack) controlling ordinary vanilla
Minecraft inputs simultaneously every 6 game ticks. V1 code/checkpoints
are preserved. The new `experiments/continuous_mine_probe.py` demonstrated
a real log after 20 consecutive 6-tick holds; no hidden commands during
scored play. Stale-inventory contamination at reset was found by the positive
controls, then FIXED: `VariableLogArena.reset` waits for both correct pose
and zero wood inventory, with 2 extra settled ticks.

**Evidence:** fair/simple same near-scenarios (seed 3737) 12-case control:
attack-then-forward **8/12**, random **0/12**, privileged geometry oracle
**11/12** (NOT a fair policy). Frozen V2 models `factorv2pilot01`,
`attackrl01`, `aimnear01` each **0/12** on these EXACT same cases. A
restricted attack-only curriculum produced **3/32** training and **2/4**
held-out logs when look/movement were disabled, but it does **not** transfer
to unrestricted play. `aimnear01` sequential and `mixstable01` interleaved
curriculum runs also failed fully unrestricted near tests; learned policies
often endlessly turn or walk away. Do not claim that more PPO epochs fixes
this or that CUDA speedup implies Minecraft skill.

New `experiments/joint_log_factorized.py` supports staged or interleaved
curriculum, privileged SCALAR shaping only in the placed-log training lab,
and optional `--save-rollouts`. Rollout data are validated, compressed,
versioned, and stored in **ignored local** `runs/joint_factorized_TAG/`,
containing only agent RGB, normal actions, scalar rewards/values,
head masks and terminal flags; **NO hidden coordinates**. Do not blindly
reuse stale PPO rollouts as new on-policy experience. `recordpilot01` did
real CUDA rollout, saved/read a 40-transition 135670-byte local dataset.

Frozen V2 can now be evaluated with `natural_world_benchmark.py --modes
joint_factorized --factorized-model TAG`. Actual natural seed 246813579,
all modes start (16.5,124.0,0.5), 0/360-tick logs for random,
forward-attack and V2 `mixstable01`. Checkpoints hashed in output. This is
pipeline proof only, NOT world transfer or speedrunning competence.

**Next scientifically justified step:** use held-out paired logs and
recorded agent visual data to investigate perceptual features, temporal
memory and more sample-efficient off-policy/representation learning BEFORE
scaling new GPU training. Keep actual game success as the gate; if basic
first-log learning fails, DO NOT proceed to piles of crafting subskills.
Never overwrite existing ignored files; Git fetch/commit/push via isolated
Gentoo Shell `/workspace`, no GitHub connector necessary.

## Historical November 2 snapshot — CUDA bridge restart

User created host NVIDIA UVM device nodes and restarted Gentoo Shell. GPU
preflight **passed**: `cuInit(0)=0`, CUDA tensor allocation on RTX 4070
SUPER and local PyTorch 2.15.0+cu132. `tools/benchmark-training.py` measured
**879.69 CUDA vs 128.99 CPU updates/s** for synthetic batch 64 (6.82x);
this is model training only, not Minecraft environment speed.

Actual CUDA real Minecraft `cuda01`: 12 training episodes x220 ticks,
3 GPU PPO updates, 4 frozen held-out x220 ticks, **0/12** and **0/4** log
acquisitions. All completed, local checkpoint/JSONL saved. Frozen policy
used only three long actions per episode, suggesting action-duration collapse
toward the fixed 80-tick mining macro; investigate before expensive long
training. Current CUDA works but UVM devices may need recreation after
future host boot. See `docs/GPU_ACCELERATION.md` and `docs/JOINT_LEARNING.md`.

## Historical November 2 snapshot — durable checkpoints

`joint_log_ppo.py` now atomically writes local `.pt` checkpoints including
Adam moments and source metadata. For continuation, use `--init-model` with
**new** `--run-tag`, `--train-seed` and `--policy-seed`; this is new-world
warm-start, not exact mid-episode replay. `warmcheck01` CPU smoke completed a
100-tick episode / second PPO update and saved optimizer state; **0 logs**,
scenario RNG repeated earlier seed in this *pilot*, so not fresh evidence.
Later code now rejects repeated train/policy RNG seeds. See
`docs/JOINT_LEARNING.md` for canonical commands. Never delete or overwrite
older saved checkpoints to make a rerun work.

## Historical November 2 snapshot — joint PPO + NVIDIA

**Read `docs/JOINT_LEARNING.md` and `docs/GPU_ACCELERATION.md` before continuing.**
A new 2-RGB-frame CNN actor-critic (888,236 parameters) trains random weights
from full collected Minecraft trajectories with timed-action GAE / clipped PPO.
11 actions jointly select movement/look/jump/attack; 80-tick mining hold is
still a **scripted motor primitive**, not learned duration. Controller resets
can plant logs and supply scalar geometric shaping ONLY in the **training
arena**. Frozen natural-world evaluation is a separate `joint_ppo` mode.

Verified real game pilot `jointsmoke01`: 2 CPU-train + 2 held-out episodes at
140 ticks; **0/2 success each**, but actual Minecraft rollout, PPO update,
local checkpoint and independent evaluation all completed. Another natural
seed `161803398` ran frozen `jointsmoke01` for 120 ticks with 0 logs. **Do NOT
claim successful transferable learning or Any% progress from these smoke tests.**
Existing old checkpoints and saves are intact; local outputs stay ignored.

GPU probe inside the same nested Gamescope/XWayland GLX display actually
reported `NVIDIA GeForce RTX 4070 SUPER` OpenGL 4.6 / driver 595.104.02,
**not CPU llvmpipe**. System game simulation still uses CPUs. PyTorch
2.15.0+cu132 is installed **project-locally** in `.venv` and CPU forward/
backward was tested. Host `nvidia_uvm` module was loaded but **current bridge
still lacks `/dev/nvidia-uvm`**; `cuInit(0)=999`, `torch.cuda.is_available()`
false. User received root host-side `mknod` commands and must restart the
bridge to bind devices once created. DO NOT claim or silently assume CUDA
training worked; use `tools/gpu-preflight.py`, and `--device cuda` for strict
failure until genuine GPU tensor allocation succeeds. Never widen filesystem
sandbox to fix CUDA. Original shell already passes UVM devices when available.

## Historical November 2 instructions — shell/Git access and first course correction

**Use Gentoo Shell `run_shell` for repository work when connected.** It
provides scoped `/workspace` access, working `.venv`, MCio, real Minecraft,
network and ordinary `git` fetch/commit/push via a **repo-scoped** SSH deploy
key. Run Git directly inside the shell rather than requiring the GitHub
connector. Always inspect `git status` first, preserve local ignored worlds,
model checkpoints, logs, private keys and user modifications. Do not display
private keys. Push commits normally and verify `git status --short --branch`.
If shell is disconnected, use GitHub connector only if explicitly available;
never pretend to have tested a runtime or pushed a commit when unable to do so.

**Read [docs/OBSERVATION_POLICY.md](docs/OBSERVATION_POLICY.md) and
[docs/NEXT_SESSION.md](docs/NEXT_SESSION.md) before new feature work.** The
older pixel-only generalization experiments below are historical baselines,
not the required future agent design. Fix MCio throughput first, implement
bounded memory-derived player-knowable state next, then resume generalization,
longer-horizon learning and contiguous wood-to-crafting progression. Do not add narrow skill scaffolding without a clear
route to a better held-out first-log or end-to-end survival score. Do not
replace the user's learn-from-random-weights goal with a scripted planner.

**Historical prototype code:** `src/gmaincraft/agent_interface.py` (the
currently deployed `AgentView` passes only immutable RGB to early visual
policies; this MUST be superseded by the new bounded player-state observation
contract, not kept as the permanent design; `AgentAction` uses normal
key/button/mouse input), `src/gmaincraft/natural_eval.py`, and
`experiments/natural_world_benchmark.py`. The controller may privately inspect
inventory to score after real gameplay. There is no admin/cheat channel in the
policy action contract. The interface is not an untrusted-code security
sandbox. Pilot natural-world result was 0 logs in 100 ticks for all 4 modes;
old source spawnRadius=10 made that smoke test **unpaired**. New world
preparation fixes spawnRadius=0, requiring another actual test. Never treat
old linear policies as a scalable fully autonomous agent; the `legacy_probe`
mode explicitly uses scripted mining timing and phase switching.

## Start every new task this way

1. Read `README.md`, **`docs/OBSERVATION_POLICY.md`**, `docs/NEXT_SESSION.md`, **`TODO.md`**, `docs/HANDOFF.md`, `docs/ARCHITECTURE.md`, and (for measured claims) `docs/EXPERIMENTS.md` from the current branch. Inspect relevant source files, their imports, and recent commits/PRs before proposing changes. Do not claim to have read them if the connector is unavailable.
2. Determine the *last verified* milestone, the *current pending* test, and the next unchecked item in `TODO.md`. Distinguish user-supplied actual run output from forecasts, synthetic tests, and code review.
3. Take user-provided run results seriously; extract meaningful summaries and errors. Do not assume a test passed merely because source exists, syntax compiles, or a toy simulation worked.
4. Make the smallest coherent code change. Preserve confirmed behavior and keep test worlds, production inputs, and model weights safe.
5. **Update `TODO.md` in the same change whenever progress, next steps, blockers, or scope change.** Update `docs/HANDOFF.md` whenever the restart instructions or latest status change. Update `docs/EXPERIMENTS.md` when a run is actually measured or an earlier claim is corrected. Update `README.md` when user-facing commands or layout change. These documents must not contradict each other.
6. Verify with appropriate unit/syntax tests; explicitly note runtime tests you cannot perform on the user's Gentoo machine.
7. Commit/source-control the changes on GitHub when authorized and give the user `git pull --ff-only` or other precise Git commands to bring their working copy up to date. Do not default to scattering downloadable Python attachments.

## Goal and honest scope

Long-term: Minecraft Java Any% agent that learns from random weights, discovers behavior through experience, and generalizes to unseen random seeds, without a scripted solution. The current experiments are **scaffolding**, NOT a pretrained or general speedrunner.

Current proven stages (2026-11-02): real MCio controller, benchmarks, reward checks, numerical camera-relative navigation (3 independent policy/train RNG seeds, each 50/50 on held-out and extra yaw-target tests); **learned RGB-feature mouse/ATTACK** (2 fresh 500-episode training runs, fresh-world audit 15/20 and 18/20 real mined oak logs, untrained 0/20); **learned W/A/S/D pickup** (2 fresh 150-episode training runs, separate zero-drift spawned-item audit: untrained 2/32, random 16/32, always-W 21/32, learned p01 30/32, learned p02 24/32). **Two-policy real mined-log combination** (two separately seeded camera-angle audits, 56 trials/model): untrained movement 21/56, fixed W 49/56, learned p01 43/56, learned p02 51/56. The two neural stages are NOT jointly trained and pickup chooses movements only **after** an agent-selected ATTACK; attacking is held for a fixed 100 ticks. Fixed-centered trunk in flat copied world, handcrafted brown-color RGB features, privileged controller-only scalar shaping and reset commands. Better than always-W on off-center spawned-item pickup, **not a proven advantage over W in integrated centered-tree tests**. No learned general tree recognition, cross-world generalization or Any% gameplay. **NOW:** a variable-player/trunk joint visual PPO prototype exists and is smoke-tested, but first-log success remains unproven; next validate real learning on held-out natural worlds and CUDA training. Never reward human-like camera aesthetics. Preserve local old navigation/aim and new pickup checkpoints. See `docs/EXPERIMENTS.md` and `TODO.md`.

Important distinctions:
- **Camera rules (user decision 2026-11-02):** never reward natural-looking camera movement, looking forward, staying level or other aesthetics. Allow strange angles, walking backward/sideways and fast turns if they help. Camera orientation matters only because it changes perceived pixels and determines actual vanilla raycast interactions; the agent must still aim to break/place/interact, within vanilla reach. Use normal MCio mouse/cursor inputs for agent actions, never controller teleport commands for agent turning.
- The `navigation_learner.py` policy receives a numeric goal direction and assumes a fixed camera yaw. `orientation_learner.py` also gets a privileged goal vector in camera-relative coordinates. Both are carefully limited **curriculum experiments**, not vision or real discovery.
- The new `log_pickup_learner.py` uses a **controller-summoned dropped item** only to train/evaluate the movement subskill; it must not be counted as natural wood gathering. The combined `log_mine_and_pickup_audit.py` uses a controller-placed *block* and real vanilla mining and pickup, no `/give` or scripted movement during scored phases. Both RGB encoders are **hand-engineered** and the trainer gets **privileged scalar progress**, not hidden coordinates as policy observations.
- The reward probe uses Minecraft `/give` to test its detector, not to assist an agent. Controller-admin commands `/tp`, `/fill`, `/setblock`, `/effect` are for episode setup/cleanup; a learning policy must never issue them during scored gameplay.
- A persistent-process **soft reset** restores only selected player properties and does not undo mined blocks, mobs, dropped items, exploration, or RNG/world state. Do not call soft-reset trials fully independent random worlds.
- A network's randomly initialized weights and actual Minecraft rollouts are central requirements. Don't replace experimental RL with a hardcoded policy/planner while claiming learning.
- A single pre/post evaluation on 20 targets is an encouraging smoke test, **not** statistical generalization proof. Use multiple runs/seeds and properly held-out scenarios before strengthening claims.
- The ultimate task is much harder than these prototypes; single-GPU training feasibility, sample complexity, and end-to-end completion remain uncertain.

## Environment and filesystem constraints

Verified user's machine: Gentoo/OpenRC, Plasma Wayland, XWayland compatibility path, NVIDIA RTX 4070 SUPER 12GB, i7-14700KF, 32GB RAM, Python 3.14.10. User prefers project-local setups in `~/Documents/minecraft-discovery`, no `sudo` (if elevated action ever needed, explain `su -c`), no background daemons, and no Portage/desktop integration unless explicitly approved. Do not request access to other local directories.

- Repository: `https://github.com/polskiftw/gmaincraft`, default branch `main`. **Normal Git works directly from shell**: HTTPS fetch and a repository-scoped write deploy key via `ssh.github.com:443` for pushes. Prefer `git fetch`/`pull --ff-only`, local commits and `git push` over relaying Git blobs through connectors. Credentials exist only under git-ignored `mcio-data/git-auth/`; never expose private key data or commit it.
- Local project root: `~/Documents/minecraft-discovery` (an existing working directory, may be attached to Git).
- Source scripts: `experiments/*.py`. Reusable code moves into `src/gmaincraft/` only when it earns its abstraction; tests go in `tests/`.
- Local-only instance/template: `mcio-data/instances/DemoInstance`, source `DemoWorld`. Scripts overwrite **only** their explicitly named disposable trial worlds; never change template `DemoWorld` or PrismLauncher worlds.
- MCio controller: `mcio_ctrl==1.5.2`, Minecraft 1.21.3 Fabric, MCio mod 1.6.1+1.21.3 observed in logs. Keep `cbor2>=5.7.1,<6` because CBOR 6.x has a known incompatible decoding behavior; document before changing.
- For the original non-sandbox Gentoo desktop, use per-process `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ...` to run MCio's GLFW viewer through XWayland; don't change global graphical settings.
- **Gentoo Shell MCP sandbox (verified 2026-11-02):** isolated `/workspace`, host home hidden; **network now deliberately shared with host to permit outbound Git/DNS; LAN/host-local services are potentially reachable**; read-only NVIDIA sysfs, GPU/DRM devices, one Wayland socket, private localhost-only `/etc/hosts`; accelerated Minecraft launched through local nested Gamescope/Xwayland with `--visible`. The local-only `mcio-data/xwayland-sandbox-wrapper.sh` sets `-ac` for nested Xwayland only. **Use `tools/run-mcio-viewer.sh`** to prevent physical desktop mouse motion/click/scroll from reaching the nested game's camera; it compiles a project-local SDL2 shim and unsets `LD_PRELOAD` before spawning the Python/Java game child. Synthetic SDL2 input test and MCio 10/10 smoke succeeded; live physical hover verification is still pending. See `docs/DEVELOPMENT.md`. Gamescope exits with code 139 after the experiment child shuts down; the Python experiment completed, logged and saved a valid checkpoint first. Direct MCio headless failed with EGL/GLFW in this sandbox. Do not weaken sandbox boundaries or expose other host folders to fix these issues.
- Running headless on a separately configured host is allowed, but still needs a working graphics stack; benchmark active simulation versus full launch/exit overhead.
- Model checkpoints, worlds, local logs, large videos, profile files and binary outputs are not source code and remain Git-ignored. Don't add them or API tokens to the public repo. Never commit raw huge console logs full of irrelevant Java stack traces.

## Development standards

- Keep runnable entrypoints documented with exact commands and flags. Avoid breaking the existing root-relative runtime paths when moving files.
- Prefer simple, inspectable, reproducible code; use explicit RNG seeds and record Minecraft/mod/Python versions, training/evaluation seeds, action spaces, reward definitions, sample counts and end-to-end timings.
- Keep evaluator separate from trainer; freeze policy during evaluation; use held-out targets/world seeds and ensure train/test leakage checks.
- Start with source-only or isolated unit tests; full Minecraft launches and GPU integration tests must be labeled and run only when requested/available. No CI job should require Mojang downloads.
- Never silently change the objective from learn-from-scratch to scripted speedrun or from fair observations to hidden-world state.
- Treat user feedback and factual results as primary. If a GitHub write fails, report it; do not claim the repo was updated.

## Completion checklist for every substantive change

- [ ] Code added/updated in the correct directories, not only a chat attachment.
- [ ] Syntax/targeted unit tests run or limitations explicitly stated.
- [ ] `TODO.md` updated in the same commit to reflect the new state.
- [ ] `docs/EXPERIMENTS.md` updated if actual measured outcomes changed.
- [ ] `docs/HANDOFF.md` updated for the next fresh-chat continuation.
- [ ] Documentation/commands updated if paths, dependencies, or flags changed.
- [ ] Commit/PR linked and user given safe local Git synchronization commands.

**In a new chat:** do not ask the user to retell the entire project. Read the repository and request only a specific missing run result or environment detail.
