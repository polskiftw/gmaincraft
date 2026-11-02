# Architecture and research direction

## Objective

Create a from-scratch reinforcement-learning agent that eventually completes Minecraft Java Any% (Ender Dragon defeated and exit achieved) on **unseen random seeds**. This is a long-term research target, not a promised deliverable or a claim of present capability.

There are two separate roles:

- **Experiment controller (privileged):** starts or restores worlds; sets up controlled curriculum arenas; applies resets, time limits and scoring; extracts results and holds experimental labels. It may issue administrative commands only in this role.
- **Learning policy (restricted):** receives an explicitly documented observation vector or image and numerical reward, chooses allowed Minecraft keyboard/mouse actions, and updates random-initialized weights from experience. It must not see cheat commands, world seeds, arbitrary hidden blocks, goals read from world internals, or direct labels of the right action during normal speedrun evaluation.

## Current prototype data flow

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

## Fairness and leakage

The existing navigation experiments pass the exact target direction to the policy as a numeric vector (absolute or camera-relative). This is a **teaching shortcut**, legitimate for learning game control, but impossible to assume for arbitrary hidden Minecraft objectives. The 2026-11-02 orientation success demonstrates handling a controller-randomized **initial** yaw, not camera turning chosen by the agent. Do not claim it learned to recognize a block or navigate from pixels.

A later configurable observation contract should distinguish:
- Human-equivalent: RGB video, visible HUD, visible GUI, inventory contents available by opening it, and self-state data that a player can directly obtain under chosen rules.
- Privileged controller-only: complete block layout, hidden inventories/entities, ore distribution, seed, stronghold coordinates, direct unseen target coordinates, internal success flags except for reward and termination.

Define this contract *before* comparing curriculum scores to open-world speedrun results.

## Learning stages

1. **Working interface and measurement (done):** controllable Minecraft, video/state observations, clean-world restarts, soft-reset benchmark, reward detector.
2. **Basic learning (controlled arena, verified):** fixed-yaw learner (7/20 to 20/20); camera-relative learner with controller-randomized initial yaw and numeric target direction replicated across 3 independent policy/training/evaluation RNG configurations (before: 9/50, 7/50, 7/50; after: 50/50 each; additional unseen target/yaw sets: 50/50 each, 2026-11-02). Still one saved-world template: no world-seed generalization, learned yaw control or visual recognition.
3. **Autonomous perception and actions (bounded evidence):** camera+ATTACK learned from 12 handcrafted RGB features (15/20 and 18/20 audited, untrained 0/20); **W/A/S/D pickup learned separately** from 14 RGB features with controller-summoned dropped-log fixtures (30/32 p01 and 24/32 p02 vs always-W 21/32). Separate frozen policies combined into vanilla mined-log evaluation (p02 51/56 vs always-W 49/56, a small unproven edge; p01 43/56). Controller provides only scalar shaping while training, never hidden coordinates to policies. All trials use one saved-world flat-arena type and a fixed centered trunk for combined evaluation. Next vary real trunk positions, replace handcrafted visual masks, and learn joint action selection. The camera is **functional, not aesthetic**; never penalize nonhuman pose or movement. Controller camera teleports are for resets only.
4. **Scale-up:** reproducible isolated episodes, offline/online experience buffers, checkpointing, held-out evaluation, graphics throughput, GPU-aware visual representation/world-model RL (DreamerV3 or a measured alternative), hierarchical skill reuse if useful.
5. **Long-term Any%:** survival/resource economy, Nether/End progression, battle, exit detection and test-seed generalization.

Success requires measuring completed runs on seeds never used for training, with clear criteria for timeouts/deaths, in-game ticks, wall-clock resources and confidence intervals. Catastrophic failures and reward hacking count as findings.

## Log curriculum observation and reset contract

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

## Course-corrected architecture (2026-11-02)

See [COURSE_CORRECTION.md](COURSE_CORRECTION.md). The **new** common policy
boundary is an immutable copy of rendered RGB (`AgentView`), and an explicit
bounded normal Minecraft input (`AgentAction`) via `MCioAgentBridge`.
`natural_world_benchmark.py` now evaluates actual natural Minecraft survival
worlds rather than planted toy objectives. Privileged worldgen/NBT/seed/
health/inventory fields remain evaluator-side; policies receive only `view`.
`legacy_probe` remains a scripted two-stage *diagnostic*, not a general
learner. Current one-step REINFORCE is insufficient for the ultimate
long-horizon objective: next learners need trajectories, temporal processing,
long-horizon credit assignment and periodic true-world evaluation. The old
stage table above documents history, not a promise that stacking the stages
will produce an Any% agent.
