# Architecture and research direction

## Objective

Create a from-scratch reinforcement-learning agent that eventually completes Minecraft Java Any% (Ender Dragon defeated and exit achieved) on **unseen random seeds**. This is a long-term research target, not a promised deliverable or a claim of present capability.

**Current design authority:** [OBSERVATION_POLICY.md](OBSERVATION_POLICY.md).
Direct skilled-player-knowable Minecraft memory reads are expressly allowed
in BOTH training and normal gameplay, with a strict distinction from
unseen chunk maps, target oracles and admin commands. Legacy RGB-only
`AgentView` remains for baseline comparability. V4 adds own-state numeric
inputs; **V5 adds a pinned opt-in read-only Fabric MCio patch** and a new
994,686-weight factorized random-initialized neural model consuming own
hunger/status and the actual vanilla crosshair-targeted block ID/coords/
face/distance in addition to own XYZ/inventory/stats and RGB. Both training
and scored evaluation consume the same V5 schema. The targeted raycast
reports **one actionable hit, not all blocks visible on screen**; extended
frustum, lighting, transparency, occlusion and entity perception still
needs empirical, bounded research. Real early V5 tests obtained zero
inventoried logs; learned general first-log and Any% remain unsolved.
See [MCIO_PLAYER_SENSOR.md](MCIO_PLAYER_SENSOR.md) and
[MINECRAFT_RENDER_PERF.md](MINECRAFT_RENDER_PERF.md).

There are two separate roles:

- **Experiment controller (privileged):** starts or restores worlds; sets up controlled curriculum arenas; applies resets, time limits and scoring; extracts results and holds experimental labels. It may issue administrative commands only in this role.
- **Learning policy (bounded player-knowledge observation):** receives an
  explicitly documented mix of structured memory-derived information and
  optional images/audio, plus scalar training rewards. It chooses ordinary
  Minecraft keyboard/mouse actions and learns from random weights. It **may
  directly read Minecraft engine memory** for facts a skilled player could
  know. This is distinct from giving it unobserved world seeds, hidden ore
  locations or oracle goals. No admin commands as scored gameplay actions.

## Current MCio+V5 data flow

Minecraft 1.21.3 Fabric MCio emits standard RGB/own-state and, with its
opt-in read-only gmaincraft patch, own hunger/armor/air and the actual
vanilla crosshair hit. The Python bridge exposes only bounded immutable
player-knowledge and currently targeted block to the V5 neural learner;
controller-only target geometry is converted into scalar reward in
controlled labs and never becomes a policy input. Frozen natural-world
scoring uses the same observation method with ordinary inputs and real
inventory. Historical numerical/RGB models below remain separate baselines.

## Historical prototype data flow

```text
Minecraft 1.21.3 + Fabric MCio mod
      ⇅  local synchronized ZMQ observation/action packets
MCio Python controller
      ↓                           ↑
Observation processing       Action mapping (W/A/S/D, eventually mouse/UI)
      ↓                           ↑
Policy (NumPy linear model) ──────┘
      ↑
Scalar reward (distance progress / terminal bonus; controlled arena only)
      ↑
Experiment controller + independent scorekeeper
      └── world setup, success checks, logs, checkpoint and resets
```

Prototypes are each self-contained in `experiments/`. Later, extract reusable environment, observations, actions, curriculum, reward, trainer, evaluator and recorder code into `src/gmaincraft/` without silently changing the meaning of earlier benchmarks.

## Camera use: effectiveness over human resemblance

The agent does **not** have to keep its head level, appear human, point the camera along the travel direction, or make smooth-looking movements. It may walk sideways while facing the sky if that is useful. Camera actions are available as needed for pixel observations, raycast targeting, mining, placing, interacting or combat. **The game engine's normal crosshair/reach/survival limits still apply.** Never teleport camera orientation as a learned action, directly edit blocks for the agent, grant inventory items, or hand the agent unseen-world coordinates. During research, the privileged controller may build a simple arena and assign scalar feedback, but the policy's input and outputs must be logged explicitly.

## Memory readings, observability and evaluation integrity

The original numeric navigation experiment passed a hidden goal vector; that
is a clearly labeled historical teaching shortcut, **not** the intended
future policy interface. The early RGB CNN explicitly omitted XYZ/inventory
and other state, but that omission is now recognized as **over-restrictive
for the user's project**. Treat those benchmarks as historical baselines.

Future policies should receive player-knowable numeric state **directly from
the Minecraft game engine**, not necessarily parsed from HUD or screenshots:
position, yaw/pitch, survival state, own inventory, stats/counters, and a
bounded set of observable blocks/entities with names/properties. The engine
might know every loaded block, but the agent should only receive things it
can see or already learned. **The actual FOV, occlusion, visibility, range,
entity and interaction threshold selection is a research question**; see
[OBSERVATION_POLICY.md](OBSERVATION_POLICY.md) for the planned validation
method. Never invent final visibility constants without test evidence.

Game-state memory access itself is allowed in **both training and scored
play**. Keep a versioned observation schema and matched train/evaluation
knowledge filter. The privileged controller may privately use extra data
for controlled resets, scalar shaping and scoring. Avoid granting unseen
ore/stronghold maps, oracle routes, or unopened containers to the policy;
this is about skilled-player-equivalent *knowledge*, not mandatory pixels.

## Learning stages

1. **Working interface and measurement (done):** controllable Minecraft, video/state observations, clean-world restarts, soft-reset benchmark, reward detector.
2. **Basic learning (controlled arena, verified):** fixed-yaw learner (7/20 to 20/20); camera-relative learner with controller-randomized initial yaw and numeric target direction replicated across 3 independent policy/training/evaluation RNG configurations (before: 9/50, 7/50, 7/50; after: 50/50 each; additional unseen target/yaw sets: 50/50 each, 2026-11-02). Still one saved-world template: no world-seed generalization, learned yaw control or visual recognition.
3. **Autonomous perception and actions (bounded evidence):** camera+ATTACK learned from 12 handcrafted RGB features (15/20 and 18/20 audited, untrained 0/20); **W/A/S/D pickup learned separately** from 14 RGB features with controller-summoned dropped-log fixtures (30/32 p01 and 24/32 p02 vs always-W 21/32). Separate frozen policies combined into vanilla mined-log evaluation (p02 51/56 vs always-W 49/56, a small unproven edge; p01 43/56). Controller provides only scalar shaping while training, never hidden coordinates to policies. All trials use one saved-world flat-arena type and a fixed centered trunk for combined evaluation. A later experimental joint RGB CNN/PPO now varies log/player positions, but it has no proven successful held-out learning yet. Replace handcrafted masks in legacy evaluators only if an evidence-backed natural transfer test merits doing so. The camera is **functional, not aesthetic**; never penalize nonhuman pose or movement. Controller camera teleports are for resets only.
4. **Scale-up:** reproducible isolated episodes, offline/online experience buffers, checkpointing, held-out evaluation, graphics throughput, GPU-aware visual representation/world-model RL (DreamerV3 or a measured alternative), hierarchical skill reuse if useful.
5. **Long-term Any%:** survival/resource economy, Nether/End progression, battle, exit detection and test-seed generalization.

Success requires measuring completed runs on seeds never used for training, with clear criteria for timeouts/deaths, in-game ticks, wall-clock resources and confidence intervals. Catastrophic failures and reward hacking count as findings.

## Historical RGB-only log curriculum observation and reset contract

- **Policy input:** only 12 engineered pixel features from the current RGB screenshot (five horizontal and three vertical color-mask bins, pixel centroid X/Y, mask fraction, constant bias). No player coordinates, yaw/pitch, Minecraft block IDs or hidden target coordinates are passed to the learned policy.
- **Aim-policy actions:** choose look left/right/up/down via MCio cursor pixels (45 pixels, approximately 6.75° per action) or choose ATTACK. The test harness then holds vanilla left button for a fixed 100 game ticks to complete a mine; this timing is still scripted.
- **Collection-policy input/actions:** separate newly randomized 4×14 softmax policy. Its only input is 14 hand-engineered RGB features of a cropped oak-colored item (7 horizontal + 3 vertical density bins, 2 centroid coordinates, mask strength, bias). It chooses W/A/S/D at 5-tick intervals; no player location/yaw, item coordinates, or hidden entities enter policy input. The fixture trainer privately knows the item spawn point to compute **scalar distance-progress reward** and adds an inventory-acquisition bonus.
- **Combined scored evaluation:** after the RGB aim policy selects ATTACK and the controller holds it, the RGB collection policy (not a hardcoded W finisher) selects normal Minecraft movement until the log enters inventory or time runs out. This is a **two-stage policy handoff**, not a joint neural system learning when to transition between all possible actions. A forward-only control and untrained control are required. In a centered trunk setup W already works almost perfectly, so the learned movement policy has not established a meaningful advantage over W there.
- **Privileged controller (training only):** creates a simple oak trunk, teleports player to random starting yaw/pitch, computes a scalar improvement/attack-proximity reward from the *known* log location and player yaw/pitch. These privileged values must not be put in policy features, even by accident.
- **Evaluator:** separate fresh copied world, independently sampled camera angles, frozen policy, no-angle-hint rewards; normal Minecraft left-click+survival break and inventory observation determine success. A zero-attack negative control must collect zero.
- **Soft reset integrity:** dropped item entities persist without explicit cleanup, falsely inflating success. The controller runs `kill @e[type=minecraft:item]` *before* every trial, clears inventory and restores either the trunk (integrated) or one fresh dropped-item fixture. Items may drift from the spawn point, making naive position-from-spawn audit guards unreliable; pickup fixtures now explicitly set initial item `Motion:[0.0d,0.0d,0.0d]` while retaining ordinary gravity/pickup. A negative no-action/no-attack test must collect zero, and the independent audit scores **actual observed inventory**. World geometry and RNG are not fully restored; no world-seed generalization is claimed.

## Reset and throughput cautions

The full-restart benchmark copies the saved `DemoWorld` template before each run, making a clean starting snapshot but paying a ~tens-of-seconds startup cost. The persistent benchmark teleports/heals/clears the player within one live world in ~0.1s; blocks, mobs, dropped items and RNG persist. It must never be advertised as independent-world random-seed training.

Separate active simulation throughput from end-to-end throughput and from neural training throughput. Rendering at 300 steps/s does not imply an agent can train at that rate after adding a neural model.

## Dependency baseline

MCio controller `mcio_ctrl==1.5.2`, Minecraft Java 1.21.3 Fabric/MCio, Python 3.14 tested. The `cbor2<6` override is essential on the current released controller because newer CBOR decoding caused blank viewer observations. The GitHub project is its own research harness; it does not vendor Minecraft, Fabric, MCio, Mojang runtimes or user saves.

## Historical first course correction (2026-11-02; sensor contract superseded)

See [COURSE_CORRECTION.md](COURSE_CORRECTION.md). The **historical V1/V2 prototype** policy boundary is an immutable copy of
rendered RGB (`AgentView`) and normal Minecraft input (`AgentAction`) via
`MCioAgentBridge`. **This policy observation restriction is superseded by
[OBSERVATION_POLICY.md](OBSERVATION_POLICY.md) as the future design.**
`natural_world_benchmark.py` now evaluates actual natural Minecraft survival
worlds rather than planted toy objectives. Privileged worldgen/NBT/seed/
health/inventory fields remain evaluator-side; policies receive only `view`.
`legacy_probe` remains a scripted two-stage *diagnostic*, not a general
learner. Current one-step REINFORCE is insufficient for the ultimate
long-horizon objective: next learners need trajectories, temporal processing,
long-horizon credit assignment and periodic true-world evaluation. The old
stage table above documents history, not a promise that stacking the stages
will produce an Any% agent.

## Historical visual-PPO prototype (2026-11-02)

[JOINT_LEARNING.md](JOINT_LEARNING.md) documents the implemented two-RGB-frame
convolutional actor-critic, 11 joint Minecraft control macros, full
trajectory/tick-aware GAE and clipped PPO. This *replaces no legacy
experiment*; it provides the first research path beyond hand-engineered RGB
masks and immediate-reward single-layer REINFORCE. The actor gets only
screen-pixel history; the teacher controls variable log/player fixtures and
scalar shaping. Natural-world `joint_ppo` evaluation loads frozen weights and
never receives target geometry. It is **not** a complete long-horizon
agent: temporal context is only two frames, no crafting or inventory UI,
fixed 80-tick mining action, and no demonstrated success on first-log
retrieval. NVIDIA GPU/OpenGL and CUDA distinctions are in
[GPU_ACCELERATION.md](GPU_ACCELERATION.md).

## Historical V2 factorized RGB action head and experience capture (2026-11-02)

`src/gmaincraft/vision_factorized.py` is a **separate** 888,750-parameter
RGB CNN actor-critic with independent discrete movement (6), camera (5),
and attack (2) logits; total joint action is realized as a normal vanilla
`AgentAction` every 6 simulation ticks. Correct factorized joint
log-likelihood and entropy are used for trajectory PPO. Curricula may
temporarily mask movement/look heads with the correct masked PPO
likelihood, but **frozen unrestricted scores enable all heads**. Existing
V1 model/code remains independently loadable; its checkpoint is NOT weight
compatible. `src/gmaincraft/rollout_store.py` can capture opt-in versioned
compressed RGB history/action/reward/duration and terminal transitions;
no privileged world-state tensors were stored **in those historical RGB
rollouts**; future policy datasets should version the permitted structured
memory-derived observations as well.

This is a functional architecture/science scaffold, NOT successful general
visual control: repeated real Minecraft experiments reported 0/12
unrestricted near-log successes for all three evaluated V2 checkpoints.
See `docs/FACTOR_POLICY_REVIEW.md` before tuning or stacking extra stages.

## Historical V3 research additions: RGB learning and event reliability

See [`VISUAL_SELF_SUPERVISION.md`](VISUAL_SELF_SUPERVISION.md) for the
scientific ledger and measurable limits. The dedicated MCio instance now
uses `chatScale:0.0`, keeping the normal HUD/crosshair and hiding giant
admin-reset chat text that had obscured policy RGB frames. This does NOT
change user's own Minecraft game options. `AgentView` remains RGB-only.

Own-action visual data: `visual_dynamics_collect.py` saves only actual
Minecraft RGB/action pairs in ignored local NPZ files, split by complete
train/validation episodes. `inverse_dynamics.py` learns camera/attack/move
effects from random weights and the agent's own actions; `temporal_pixels.py`
offers a pixel-only delta mode; `recurrent_inverse.py` adds causal prior-
action GRU memory for offline diagnostic accuracy. None has replaced the
online Minecraft RL actor. `joint_log_factorized.py --vision-init-tag TAG`
can initialize its visual CNN only from the user's self-collected raw RGB
encoder; action/policy heads remain random. Downstream first-log gain is
not demonstrated.

Action time is separately schema-tagged V2(6) or V3(12) game ticks; older
models remain intact. `MCioAgentBridge` privately accumulates Minecraft
actual mined/picked-up stats on EVERY game tick, and arena-scoped
`CumulativeMinedTracker` corrects newly observed cumulative stat values
across per-episode cache resets. This fixed a major experiment-invalidating
mined-block reward bug; stats were evaluator-only in the historical prototype; the new design
permits skilled-player-knowable counters directly as policy inputs. Corrected real
Minecraft test mine/no-attack/mine=1/0/1; frozen V3 can mine placed logs
with camera/movement locked, but unrestricted first-log acquisition is
still zero in measured tests. Model weight size/architecture alone is not
proof of useful learned Minecraft ability.
