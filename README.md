> **Latest (Nov 4): V1 200-world training is COMPLETE, but its low-level
> wood gathering regressed (only 3 actual log pickups).** New strictly
> versioned V2 learns rare **player-visible** wood progress from its own
> previously recorded Minecraft actions, retains original successes during
> resumed self-play, and has obtained real logs on multiple frozen unseen
> seed/RNG tests. Three separate completed real V2 training worlds acquired
> another FIVE logs in total. Neither reliable wood collection, crafting,
> skill composition nor Any% is solved. Full measured results, failures,
> versioned CLI and lineage: [V2 rare-success and forgetting audit](docs/V2_RARE_SUCCESS_REPLAY_AUDIT_2026_11_04.md).
> **Active V2 long training:** `qwood_v2long1104` launched and verified
> with up to 100 NEW natural survival worlds starting at TRAIN index206;
> its FIRST world actually acquired 5 logs, with 32 GPU updates,
> checkpoint cursor207 and new world207 already launched. Use
> `tools/run-replay-q-collection.sh status qwood_v2long1104` and
> [current handoff](docs/NEXT_SESSION.md) for status, safe stop/resume
> and eventual frozen heldout gate, not the old v1 commands below.

> **Latest executable work (Nov 3 local / Nov 4 UTC):**
> [Real replay-Q self-play, online CUDA training, independent heldout audits](docs/REPLAY_Q_LIVE_LEARNING_2026_11_04.md).
> New from-scratch Q learning reuses actual player-only V9 action experience,
> with no teacher, hidden seed map, auto-target or scripted strategy. A real
> Q rollout collected 2/1 logs on the same unseen seed under two action RNGs,
> but had no pickups on two other unseen seeds. The separate self-play
> collector restarted across two REAL new Minecraft worlds, saved replay,
> performed 16 additional CUDA updates and independently acquired one dark
> oak log. These are useful experiments, **not** a proven first-log agent
> or autonomous Any% speedrunner. See the guide for the new bounded,
> stoppable longer-training commands; previous V2-V9 work is preserved.
> **Historical V1 200-world run COMPLETED**: `qselfplay_v1_1104`.
> Its archived initial launch is no longer an active process. The
> versioned new V2 continuation is in `docs/NEXT_SESSION.md`.

# gmaincraft

> **Latest executed V9 research:** [V9 real-game temporal target, CUDA,
> replay and goal-transfer audit](docs/V9_TEMPORAL_TARGET_AND_REAL_GOAL_AUDIT_2026_11_04.md).
> New V9 remembers its own recent crosshair, attack and visible mining
> progress without changing the V8 reward or normal motor. Two fresh
> survival training worlds plus resume ran 1,200 units and 15 CUDA updates,
> acquiring one gravel item and zero wood. A separate self-goal learner
> trained on that genuine gravel gain but failed gravel acquisition
> on frozen unseen-world evaluations; incidental wood gains were RNG
> dependent. Independent goal-action RNG fix is verified. No general
> wood mastery or learned full-game strategy is claimed.


> **New executable research infrastructure:** [Durable self-play replay,
> real-GPU resume, goal-conditioned skill learning and frozen practice
> evaluation](docs/DURABLE_EXPERIENCE_AND_SELF_GOALS.md).
> Unlike the long-term architecture plan, these tools now RUN. Actual
> player-only experience was captured in two newly generated vanilla
> worlds (432 ticks, 19 neural choices and 4 CUDA PPO updates). The
> replay/goal training toolchain does not confer a reliable wood skill,
> learned skill manager, crafting or full-game Any% competence.


> **Latest research direction (Nov 3, 2026):** The real natural survival
> trainer has completed V7/V8 controlled CUDA runs, but does NOT yet
> acquire wood reliably on unseen seeds/alternate action RNGs. The new
> **[bootstrapping-to-Any% design](docs/BOOTSTRAP_TO_ANY_PERCENT.md)**
> specifies how learned skills should eventually compose into a
> self-directed strategy without scripted human routes, hidden-world
> observations, global item-reward ladders or evaluator-selected skill
> handoffs. It is a DESIGN, not an implemented multi-skill agent.
> Read [V7/V8 results](docs/V7_V8_WOOD_LEARNING_FAILURE_DIAGNOSIS_2026_11_03.md)
> and [current handoff](docs/NEXT_SESSION.md) before using the historical
> V4 training suggestion below; that V4 run was already completed.


## Current milestone — first autonomous log in natural survival

**November 3, 2026:** The new from-scratch **four-head memory-first V4
agent** learned to choose ordinary movement, camera, attack AND how long
to hold each action (6, 24 or 72 game ticks). In a real, entirely
noise-generated survival world with no placed tree, scripted handoff or
admin gameplay, it **mined a wood block and actually picked up the drop
into its inventory**. A frozen four-update checkpoint also collected a
real log on **1/4 independent held-out seeds**, versus **0/4** for a
same-action, same-time uniform-random control. This is a promising
individual success, **not a demonstrated reliable learned wood skill**,
and Any% is much farther away. Read the complete evidence and constraints:
[First natural wood inventory + versioned action-duration V4](docs/NATURAL_WOOD_FIRST_LOG_PERSISTENCE_V4.md).

**Archived historical V4 training command (already completed; do not re-run):**

```bash
cd "$HOME/Documents/minecraft-discovery"
tools/run-fresh-treadmill.sh resume persistpilot1103 24000 600 wood_persistence_v4
```

Its original 600-unit checkpoint has been independently preserved for
frozen comparisons. All old model families, natural worlds and research
patches are retained. **134 unit tests pass** and real Minecraft
start/resume, same-seed evaluation and frozen old-model compatibility
have been independently exercised.

**NEW PRIMARY TRAINING KIT: [endless nonrepeating random-seed survival treadmill](docs/FRESH_SEED_TREADMILL.md).**
`tools/run-fresh-treadmill.sh start mytrain01 forever` now automatically
generates a NEW real vanilla noise Minecraft seed per episode, trains ONE
random-initialized memory-first actor continuously, saves safe checkpoints,
prunes old owned disposable world saves and supports `stop`/`status`/`resume`.
Unlike the old eight-world loop, it never deliberately revisits a consumed
training seed. Frozen neural and uniform random baselines use provably
disjoint held-out seeds and the **same initial spawn location** per seed.
Real tests covered independent new-world generation, training resume,
in-game stop, aborted-world quarantine and independent frozen evaluation.
**Earlier treadmill launch tests passed; current full suite: 134.** Current MCio still restarts the Minecraft
client per seed (~7s world generation plus9-14s client startup); hot world
switching is a future independently tested optimization, not promised.
The subsequent V4 duration-policy pilot HAS now mined and collected one
real natural-world log, but this remains an early, unproven skill. See the
linked guides for the current foreground training command.

**Primary development track: [real natural survival online learning](docs/NATURAL_SURVIVAL_ONLINE_TRAINING.md).** The project now has a working, resumable **from-scratch memory-first Minecraft PPO trainer**, running in actual vanilla-noise survival worlds. Movement, camera and attack are all learned actions; no player teleport, preaimed log, world cheat command or scripted mining/pickup handoff occurs during training. The run includes world rotation, local model/optimizer/exploration checkpoints, clean stop/resume and independent frozen/random controls. The first v2 pilot completed 720 normal-world decisions across three seeds and encountered 366 genuinely first-hit visible wood surfaces, but **has not yet mined or collected natural wood**. A held-out random policy still beats the immature learner on short exploration benchmarks; don't conflate a successful training pipeline with a successful speedrunner. To actually train unattended, see the linked guide for `tools/run-long-natural.sh` start/resume/stop commands.

**Earlier controlled resource-acquisition milestone:** [Frozen learned mining + pickup on genuine Minecraft drops](docs/COMPOSED_MINE_PICKUP_AUDIT.md).
Two scratch-trained policies have been composed without privileged actions
while playing: a Minecraft-client mining-progress motor mines an oak log,
and a separate RGB-only learned movement component walks to pick up its
physical vanilla drop. In 12 independent pre-aimed held-out scenarios,
**11/12 real inventory logs** were collected with the `p02` pickup model;
a second set yielded 9/12. Simple always-hold attack and often forward
movement remain competitive. **Still NOT end-to-end or a natural Any%:**
controller stages/aims the tree BEFORE play, and a scripted real mined-event
handoff chooses when pickup starts. The next goal is calibrated memory-
visible dropped items and autonomous phase selection in shared train/frozen
observations. Checkpoints/worlds remain local and Git-ignored.

**New actual learned Minecraft behavior:** see
[`docs/MINING_FEEDBACK_AND_ATTACK_LEARNING.md`](docs/MINING_FEEDBACK_AND_ATTACK_LEARNING.md).
An opt-in real engine-memory mining feedback sensor now exposes the current
player's visible breaking stage. A separately trained, from-scratch
six-tick vanilla attack motor matched an always-hold baseline (**12/12**
held-out real Minecraft pre-aimed oak blocks mined), while untrained and
random controls broke none. This is NOT tree finding, log pickup or Any%.
Original V5 trained checkpoints and research modes remain intact.

**Earlier sensor research:** [Real Minecraft visibility calibration](docs/VISIBLE_SURFACE_CALIBRATION_2026_11_03.md)
compares memory-derived block-surface candidates with independent actual
rendered RGB across 23 lab fixtures, three natural survival seeds and
night/occlusion controls. Fine 25x17 rays are an explicit measured research
option; 13x9 remains the default. The engine-only sensor is NOT YET a
trained policy input or a general log-finding skill; no claims of Any%.

**Updated:** The opt-in camera-ray sensor now supports synchronized,
on-demand last-action-tick observations with real Minecraft verification;
see [`docs/SURFACE_REQUEST_V1.md`](docs/SURFACE_REQUEST_V1.md).
Original V5 model weights and scoring are unchanged.

**Previous sensor experiment (Nov 2026):** See
[`docs/SURFACE_SENSOR_V1.md`](docs/SURFACE_SENSOR_V1.md). A reversible,
opt-in Fabric patch now samples first-hit camera rays and groups visible
block-position candidates; actual Minecraft off-crosshair and occlusion
controls passed. These are **not** calibrated visual/perceivable thresholds
and are **not** used by the current V5 trained model. Sample frequency
must be optimized (~32% current warm overhead on one matched fixture)
before a new observation-policy version or large training runs.

**A Minecraft learning-to-speedrun research project.** The long-term goal is
an agent that learns from randomly initialized weights to complete Minecraft
Java Any% on unseen random survival seeds. It can **directly read structured
Minecraft engine/memory data that a skilled player could know** (player
position/facing, inventory, health, counters, observable block/entity IDs,
etc.) during both training and gameplay. A screenshot may help, but should
not be forced to teach facts Minecraft can provide numerically. Neither
expert demonstrations nor a scripted speedrun are the intended solution.

**Design contract:** [docs/OBSERVATION_POLICY.md](docs/OBSERVATION_POLICY.md).
Visible/actionable-block eligibility and field-of-view/occlusion thresholds
are future experiments, **not something to guess before measuring**.
**Latest verified progress:** a small, opt-in Fabric MCio patch now reads
Minecraft's **actual vanilla crosshair-targeted block ID, coordinates, face
and distance**, plus own hunger/armor/air/slot/dimension, directly from game
memory. The 4/4 oak/turn/occlusion controls passed in actual Minecraft.
Empirical target reach/transparent-glass checks are documented (this is an
*actionable* block raycast, not a calibrated list of all visually perceivable
blocks). New from-scratch **V5 neural actor** feeds that status along with
own XYZ/facing/inventory/counters and optional RGB to the SAME train/frozen
policy. It ran 4 real CUDA training episodes,8 frozen cases and a paired
real survival seed; **no successful log pickup yet**. Target telemetry showed
30/30 oak hits in one stationary frozen scenario, but the young model chose
no attack, so fixing long-horizon action credit is a next research gate.
Read [source patch, calibrated tests and V5 results](docs/MCIO_PLAYER_SENSOR.md),
[canonical observation policy](docs/OBSERVATION_POLICY.md) and
[latest handoff](docs/NEXT_SESSION.md).

**Performance:** reproduced two independent bottlenecks: larger natural
render distance slows MCio (600-tick seed test **4.535s default vs 1.695s
at four chunks**) and Gamescope's **SDL visible backend can cap at ~60
game ticks/s**. High-throughput, GPU-rendered offscreen mode via
`GMAINCRAFT_GAMESCOPE_BACKEND=headless` recovers ~**0.77s/300 ticks** on
original MCio without losing internal RGB. A patched, opt-in status packet
added only ~0.009s/300 warmed ticks in one matched control. Keep SDL visible
for watching the agent, accepting its conditional slowdown. All disposable
instance options are backed up/restored; nothing on the user's normal
Minecraft/host needs modification. Details [performance A/B](docs/MINECRAFT_RENDER_PERF.md).

**Historical experiments (2026-11-02; NOT the target sensor design):** Minecraft Java 1.21.3 + MCio and three seeded numerical-navigation runs are verified. Two random-initialized, RGB-feature mouse+ATTACK models learn to aim at a controlled oak trunk (audited 15/20 and 18/20 real Minecraft logs). Two independently trained RGB-feature W/A/S/D policies learn to approach a **dropped** oak log: in a fresh 32-case offset-item audit, the best reaches **30/32**, versus **21/32** for just holding W. Combining the learned aiming and collection policies yields **51/56 real mined-log inventory pickups** over two fresh camera-angle sets for pickup model `p02`, versus **49/56** with scripted W. This does **not** establish a meaningful advantage in the centered-tree setup, and it still uses hand-engineered pixel masks, controller-only scalar rewards, fixed 100-tick attack holds, a single flat-world template, and separate training for the two neural stages. No general tree recognition or Any% speedrunning.

## Sensor intent versus the historical pixel-only prototype

**Actual user intent:** read skilled-player-knowable facts *directly from game
memory*, not through screenshot reconstruction. For example, full own
inventory and mined-block counts can be numeric inputs; XYZ/yaw/pitch can be
used directly. A block list must be limited to what the player could
reasonably perceive or remember, using visibility research rather than an
arbitrary fixed radius or free X-ray. Screen images are an optional source
of additional information.

**Implemented in V5:** base MCio provides own XYZ/facing/health/
inventory/progress counters; the opt-in patched Fabric mod additionally
reports own hunger/saturation/armor/air/dimension/selected hotbar slot and
exact **vanilla crosshair** block hit. A frozen V5 model receives these same
versioned fields as it did during its Minecraft training; the older RGB-only
`AgentView`/V4 features and saved weights remain unchanged. **There is NOT
yet a calibrated broader visible-object/entity list**, and the model has not
learned unrestricted log acquisition or Any%. [Full sensor design and actual
Minecraft evidence](docs/MCIO_PLAYER_SENSOR.md).

## Current runnable V5 memory+targeted-block training/evaluation

With a pinned upstream MCio build and a backed-up original in the isolated
`DemoInstance`, the opt-in `--with-status` mode uses the new read-only Fabric
status. Legacy V4 remains the default mode if the flag is omitted. Every
run needs a unique tag. Use one Gamescope session for training and a
**separate** session for frozen evaluation, as the nested viewer can crash
when restarting a second Minecraft. `--visible` still renders actual game
RGB even with the no-window Gamescope backend.

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python -m unittest discover -s tests -q
# Install patched MCio only in the disposable DemoInstance, backed up:
tools/build-mcio-player-status.sh --install
# Real raycast, occlusion, glass and distance fixtures (no neural model):
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/mcio_target_block_probe.py --run-tag NEWTARGET --visible --calibrate
# NEW V5 weights from scratch; game options revert after exit:
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_memory.py --run-tag NEWTRAIN --with-status --episodes 8 --update-every 4 --max-ticks 240 --device cuda --visible --skip-eval
# FROZEN eval in a different game/viewer session, no optimizer update:
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/memory_frozen_eval.py --run-tag NEWEVAL --model-tag NEWTRAIN --with-status --eval 4 --max-ticks 240 --device cuda --visible
```

For exact paired natural survival seed evaluation see
[`docs/MCIO_PLAYER_SENSOR.md`](docs/MCIO_PLAYER_SENSOR.md). The results
so far are still **zero unrestricted acquisitions**; the next research
step is valid wider *perceivable* block/entity eligibility and better
recurrent action duration/attack persistence. Do not turn a tiny PPO smoke
into a supposed competent agent. Historical stages are below.

## Historical November 2 experiment — self-collected RGB learning and mining events

We inspected real policy screenshots and found that MCio reset error chat was
obscuring the tree. The **dedicated MCio test instance** now uses
`chatScale:0.0`, preserving crosshair/HUD and normal camera behavior without
changing your everyday Minecraft settings. **Those historical RGB-only
experiments** did not receive numeric or world state; V4/V5 now do.

New CUDA experiments learn visual action effects from the agent's **own
random Minecraft inputs**, with no human demonstrations or externally
pretrained weights. Held-out same-arena action-prediction accuracy for camera
was 42.6% versus a 23.5% majority baseline; attack 84.6% versus 52.2%; and
movement only 15.4% versus 14.7%. With camera fixed, movement prediction
improved to 38.5%. A separate GRU slightly improved the diagnostic, but is
**not** yet integrated into the live RL agent. This is perceptual learning,
**not** Minecraft playing competence.

Minecraft's real mined-block and pickup counters are now available as
**evaluator-only** learning signals. A cumulative-counter reset bug invalidated
one experimental reward run; we kept it excluded, then verified mine/no-
attack/mine actually scores **1/0/1 new blocks** in the real game. Clean
follow-up training mined three genuine logs but collected none. A paired
restricted frozen test yielded **2/4 inventory logs** for the corrected-
event checkpoint, versus **0/4** for the earlier checkpoint; fully free
movement/look still scored **0/4**. Genuine random-seed survival world
transfer remains **0 logs**. The end-to-end Any% goal is still unsolved.

See [docs/VISUAL_SELF_SUPERVISION.md](docs/VISUAL_SELF_SUPERVISION.md) for the
complete evidence, reproducible commands, tests, known MCio throughput
regression and next research gate. All trained models, raw screenshots and
worlds stay local/Git-ignored. Existing checkpoint families are preserved.

## Latest research gate: V2 factorized control and measured limitations

The model now chooses **movement, camera, and attack simultaneously every 6
game ticks**, rather than treating an 80-tick mining hold as one decision.
The vanilla-game positive control confirmed that repeated 6-tick attack
selections genuinely mine and collect a log. Three different V2 frozen CNN
checkpoints then scored **0/12** on a matched nearby-log scenario set, while
a simple attack-then-approach diagnostic scored **8/12**. A privileged
geometric aiming oracle scored **11/12** but is **not** a fair learned policy.
A stationary restricted RL warmup scored 2/4 held-out logs, but that does
**not** transfer to free camera/movement. A newer 3-mode seed-246813579
survival-world benchmark ran successfully with **0 wood for every mode**.

We are **not** scaling the same ineffective CNN/PPO indefinitely. The next
research step is diagnosing visual representation, credit assignment,
exploration and longer-term memory using reproducible paired benchmarks and
**opt-in local replay-data recording** (`--save-rollouts`). All runs use
working CUDA; previous checkpoint files and worlds are preserved. See
[docs/FACTOR_POLICY_REVIEW.md](docs/FACTOR_POLICY_REVIEW.md) for actual
scores, methods, scripts and recommended next experiments.

## Live NVIDIA CUDA now operational (post-bridge restart, 2026-11-02)

The isolated Gentoo Shell now has **working RTX 4070 SUPER CUDA tensor
computation and actual GPU-backed neural-network training**. Benchmarked
~6.82x faster network-only updates than CPU (synthetic batch-64 test).
The real-game GPU pilot `cuda01` completed 12 training episodes/3 GPU PPO
updates plus 4 frozen eval episodes, but scored **zero logs**; the long
80-tick mining macro appears to dominate frozen decisions and needs an
action-space correction before expensive large-scale training. See
[docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md) for all measurements,
and [docs/JOINT_LEARNING.md](docs/JOINT_LEARNING.md) for outcomes.

## New joint learning and RTX GPU diagnostics (2026-11-02)

The experimental course correction now includes a **single RGB CNN
actor-critic (PPO with timed game-step advantages)** controlling look,
movement, jumping and a fixed-duration mining macro. `jointsmoke01` verified
its real Minecraft training/update/frozen evaluation and `jointtransfer01`
verified natural-world checkpoint loading. All successes in those short
smokes were **0 logs**; this is plumbing, not evidence of learning to
speedrun. Models are local and Git-ignored. See
[docs/JOINT_LEARNING.md](docs/JOINT_LEARNING.md) for the exact scope.

The nested NVIDIA OpenGL renderer is **confirmed hardware-accelerated** on
the RTX 4070 SUPER (not software CPU rendering). Project-local PyTorch
2.15.0+cu132 was initially installed while CUDA access was blocked by
missing UVM nodes; the user subsequently enabled those nodes and restarted
the bridge. **CUDA is now working**, as confirmed above. See
[docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md) for verification,
strict CUDA diagnostics and measured CPU training reference. No Gentoo-wide
package or driver changes were made by the agent.

## Research course correction (2026-11-02)

**Do not equate toy-arena successes with progress toward Any%.** We now have a
real seeded-survival-world first-log transfer benchmark and a **historical
pixel-only prototype** policy/action interface. The target observation model
is now explicitly structured and memory-derived; see
[docs/OBSERVATION_POLICY.md](docs/OBSERVATION_POLICY.md).
The earlier linear policies are retained as diagnostic
baselines, not the final learning architecture. The initial 100-game-tick
pilot on natural seed `314159265` produced zero logs for all four controls;
that pilot used default randomized spawn positions and is **not paired
performance evidence**. New benchmark world preparation fixes spawn scatter.
See [docs/COURSE_CORRECTION.md](docs/COURSE_CORRECTION.md) for the research
plan and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for real run commands.

This repository is the canonical home for code and continuity across ChatGPT conversations. Start with [AGENTS.md](AGENTS.md), [TODO.md](TODO.md), and [docs/HANDOFF.md](docs/HANDOFF.md).

## Repository layout

```text
gmaincraft/
├── AGENTS.md                    Agent workflow and rules for future chats
├── TODO.md                      Living, ordered project task list
├── docs/
│   ├── OBSERVATION_POLICY.md    Canonical bounded memory-state observation design
│   ├── NEXT_SESSION.md         Priority-ordered new-chat performance handoff
│   ├── ARCHITECTURE.md          Architecture and historical experimental contracts
│   ├── DEVELOPMENT.md           Installation, commands, testing, migration
│   ├── EXPERIMENTS.md           Verified experiment history and limitations
│   └── HANDOFF.md               Copy/paste new-chat resumption guide
├── experiments/
│   ├── random_agent.py          Random actions + clean-world restarts
│   ├── benchmark_agent.py       Windowed and headless full-restart benchmarks
│   ├── persistent_benchmark.py  Fast soft resets in one Minecraft process
│   ├── reward_probe.py          Per-episode first-seen item rewards
│   ├── navigation_learner.py    First real in-game policy-gradient learning
│   ├── orientation_learner.py   Camera-yaw generalization experiment
│   ├── log_interaction_probe.py Vanilla mouse/mine/pickup instrumentation test
│   ├── log_visual_aim_learner.py RGB-feature policy learns look/attack in toy arena
│   ├── log_visual_aim_evaluator.py Fresh-world audited frozen camera policy
│   ├── log_pickup_learner.py     RGB-feature W/A/S/D pickup reinforcement learning
│   ├── log_pickup_evaluator.py   Fresh-world spawned-item pickup audit + controls
│   └── log_mine_and_pickup_audit.py Audited real mined-log two-policy combination
├── src/gmaincraft/              Home for reusable code as it matures
├── tests/                       Minecraft-independent unit tests
└── requirements.txt             Pinned MCio compatibility dependencies
```

`LICENSE` is the existing Apache-2.0 license, preserved from the original repository.

## Quick start

Prerequisites: Linux with a working OpenGL/NVIDIA configuration, Python 3.12+, and a Java-capable Minecraft installation. The verified machine uses Gentoo Linux, Plasma Wayland with XWayland, Python 3.14.10, an RTX 4070 SUPER, and an i7-14700KF. MCio downloads/uses a separate Minecraft Java **1.21.3** Fabric client; the user's normal PrismLauncher installation is separate and untouched.

For a **new directory**:

```bash
git clone https://github.com/polskiftw/gmaincraft.git
cd gmaincraft
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/mcio demo --mcio-dir "$PWD/mcio-data"
```

The initial MCio demo provisions `mcio-data/` and creates the `DemoInstance` and `DemoWorld` used as templates by these experiments. It may download the Minecraft client/runtime on the first run. Press Q in the demo's viewer to exit it.

**Already have `~/Documents/minecraft-discovery` from this project's earlier experiments?** Do **not** delete it or clone over it. Instead attach that existing directory to this repository using the safe commands in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). This preserves `mcio-data/`, `.venv/`, `checkpoints/` and `logs/`.

### Next learning goal: acquire a log, not a human-looking camera

Our camera rule is **functional, not cosmetic**. The agent need not look forward, keep its head level or move its view smoothly. It may face skyward while traveling; it must use legitimate mouse aiming and vanilla reach mechanics when breaking, placing, interacting or fighting. That historical experiment used rendered-pixel observation → selected
camera/movement/attack → actual oak-log pickup. The next generation should
also read skilled-player-knowable state directly from Minecraft memory;
undiscovered goal coordinates remain outside its observations.

The **scripted mechanics probe** verified vanilla mouse and log mining. The **learned camera/ATTACK** stage now chooses where to look and when to attack from engineered screenshot features, and a **separately learned pickup** policy chooses W/A/S/D from item-appearance pixels. Both train with controller-only *scalar* shaping while the learned policies see no hidden target positions. In two fresh integrated tests, combining `rgb02` aiming and `p02` pickup obtained **51/56 genuinely mined logs** with *no scripted post-attack movement*; simple always-forward pickup scored **49/56**, so a meaningful improvement for the centered trunk remains unproven. Off-center dropped-log tests show a clearer advantage for learned movement. The camera is free to look bizarre as long as the game rules are followed. See [the experiment ledger](docs/EXPERIMENTS.md); we now have an unproven joint visual PPO prototype varying trunk/starting positions; next verify meaningful learning and natural-world transfer, not merely more toy tasks.

### Run an experiment

After the MCio demo has finished and you are back at a shell:

```bash
cd ~/Documents/minecraft-discovery
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/navigation_learner.py --visible
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/orientation_learner.py --visible
```

**For the Gentoo Shell MCP sandbox**, use `tools/run-mcio-viewer.sh` rather than calling Gamescope directly. It builds a project-local SDL2 shim that keeps the real desktop mouse **unlocked and unforwarded**: mouse motion/click/scroll events are filtered before reaching the nested Minecraft window, while MCio agent actions still work. Example disposable run: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/orientation_learner.py --visible --run-tag viewer_demo --episodes 8 --eval 5`. Test the shim alone with `tools/test-gamescope-mouse-filter.sh`. Physical hover behavior still needs user verification; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Run **one** MCio experiment at a time: they use the same local action/observation ports and disposable Minecraft worlds. The orientation experiment requires a local `checkpoints/navigation_policy.npz` from an earlier navigation run. It does **not** include that checkpoint in Git. The original host can use headless NVIDIA mode, but the isolated Gentoo Shell MCP environment was verified with **nested Gamescope + `--visible`**, not MCio direct headless mode; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Default reruns overwrite `checkpoints/orientation_policy.npz`. New reproducibility flags `--policy-seed`, `--train-seed`, `--eval-seed`, `--extra-seed`, and `--action-seed` require a unique `--run-tag` to protect that checkpoint. Tagged runs create separate worlds, checkpoints and CSVs, and refuse existing tags.

Each experiment is a direct-entry Python prototype, not a production package. Its runtime paths are relative to the repository root even though the scripts live in `experiments/`.

### Verify or repeat visual aiming

Inside the Gentoo Shell sandbox, use a unique run tag and the mouse-safe viewer. This creates a separate disposable world and outputs only git-ignored model/CSV data:

```bash
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_learner.py --run-tag rgb03 --episodes 500 --eval 14 --policy-seed 5503 --train-seed 7725 --eval-seed 8816 --visible
```

To **evaluate existing checkpoints without further training**, use another new tag:

```bash
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_evaluator.py --run-tag audit02 --models rgb01 rgb02 --eval-seed 8816 --eval 20 --visible
```

The evaluator creates a fresh copied world, clears dropped items before every case, verifies no inventory pickup without an attack, and requires a real observed oak log for success. These are still toy single-trunk tests, not independent Minecraft world seeds.

### Train/audit pickup and the full two-policy interaction

Use a unique tag for every run. In the MCP sandbox, all three examples launch the actual Minecraft game through the mouse-safe viewer; model files, logs, and worlds remain git-ignored:

```bash
# Train a NEW RGB-feature W/A/S/D policy on controller-summoned dropped logs
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_pickup_learner.py --run-tag p03 --episodes 150 --eval 20 --policy-seed 4403 --train-seed 6603 --eval-seed 8805 --visible
# Audit existing pickup policies against untrained, random, and always-forward
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_pickup_evaluator.py --run-tag dclean03 --models p01 p02 --eval-seed 8806 --eval 32 --visible
# Real mined-log combined test: agent selects camera+attack, then agent selects movement
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_mine_and_pickup_audit.py --run-tag f03 --aim-model rgb02 --pickup-models p01 p02 --eval-seed 8818 --eval 32 --visible
```

The pickup-only trainer **summons a dropped item as a controller-owned test fixture**; only the *combined* audit obtains a real mined log. Re-running a used tag is refused. No model weights are included in the public repository.

## Verified milestones

- Random-action baseline controlled the actual game and restored the starting world between episodes.
- Full-restart benchmark: 195.0 active steps/sec windowed; 185.5 active steps/sec headless (two small trials each).
- Persistent-process benchmark: 284.5 active steps/sec over 5,000 steps; 123.5 steps/sec measured end-to-end; soft resets around 0.08–0.11s. **Soft resets do not restore the world.**
- First-seen item reward detector passed all five controller-injected item tests. **Not a learned item-acquisition policy.**
- Simple navigation learner: 7/20 successes before training, 20/20 afterwards on the same 20 held-out targets, after 120 training episodes in 12.8 seconds. Fixed yaw, privileged goal direction, no vision.
- Vanilla log interaction probe (2026-11-02): camera deltas worked, oak log mined via held left mouse button, item entered observed inventory after moving closer; 1 log verified, no `/give`. Controller placed trunk in disposable world; scripted test, NOT learned perception.
- **Learned W/A/S/D collection** (2026-11-02): two fresh 150-episode Minecraft trainings. On an independent 32-case random-offset dropped-item test: untrained 2/32, random 16/32, fixed-forward 21/32, p01 **30/32**, p02 **24/32**. The policy receives only engineered RGB pixel features; the trainer uses privileged item-location scalar progress, and items are controller-summoned fixtures.
- **Combined real mined-log policy stages** (2026-11-02): `rgb02` chooses camera/attack; pickup policy chooses movements, **no scripted W**. Over two frozen camera-angle audits (24+32 cases), untrained pickup 21/56, fixed W 49/56, trained p01 43/56, trained p02 **51/56** real Minecraft inventory log pickups. In centered-tree setups the W baseline is already excellent; two extra wins do not prove a robust advantage.
- **Learned camera/attack curriculum** (2026-11-02): two random-weight linear RGB-feature policies trained for 500 episodes each, chose look/attack, then earned **15/20 and 18/20 real oak logs** on the same independent 20-camera-angle set in a freshly reset/loot-cleaned arena, versus 0/20 untrained. A stale-item contamination in the initial evaluation was diagnosed and fixed before these audited results; see `docs/EXPERIMENTS.md`.
- Camera-relative orientation learner (**real Minecraft**, 2026-11-02): initial 9/50, 7/50, 7/50 across three independently initialized/trained numerical policies. All three reached **50/50 held-out and 50/50 extra unseen** target/yaw scenarios after 300 training episodes each (30.16s, 26.67s, 25.17s learner time; 299/300 training wins per run). Policy never moved the camera or saw visual goal pixels; privileged numeric goal direction was supplied.

All measurements are preliminary and hardware/setup specific. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the actual scope and caveats.

## Safety, scope, and version control

`mcio-data/`, `.venv/`, `logs/`, `checkpoints/`, recordings, Minecraft worlds and model binaries are ignored by Git. The Python source and documentation are the source of truth; **Git alone cannot recreate an uncommitted trained checkpoint or world save**. Preserve those locally or arrange an explicit backup if needed.

No daemons, startup services, `sudo`, Portage changes, or unsolicited filesystem access are needed. Minecraft `/give`/`/tp`/`/fill` commands are for controller-led setup and testing, not for the learning policy. Avoid accidentally turning privileged game state into agent observations.

For exact commands, local migration, unit tests, and known Linux compatibility notes, consult [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
