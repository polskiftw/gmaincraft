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

## Fairness and leakage

The existing navigation experiments pass the exact target direction to the policy as a numeric vector. This is a **teaching shortcut**, legitimate for learning game control, but impossible to assume for arbitrary hidden Minecraft objectives. Do not claim it learned to recognize a block or navigate from pixels. Similarly, it did not learn the camera-movement controls yet.

A later configurable observation contract should distinguish:
- Human-equivalent: RGB video, visible HUD, visible GUI, inventory contents available by opening it, and self-state data that a player can directly obtain under chosen rules.
- Privileged controller-only: complete block layout, hidden inventories/entities, ore distribution, seed, stronghold coordinates, direct unseen target coordinates, internal success flags except for reward and termination.

Define this contract *before* comparing curriculum scores to open-world speedrun results.

## Learning stages

1. **Working interface and measurement (done):** controllable Minecraft, video/state observations, clean-world restarts, soft-reset benchmark, reward detector.
2. **Basic learning (partly done):** a randomly initialized linear policy learned a fixed-yaw target-navigation toy task (7/20 to 20/20); yaw-generalization test pending real results.
3. **Autonomous perception and actions:** learn camera turning, block visual targeting, mouse interaction, inventory and crafting inputs; combine event rewards without admin-command leakage.
4. **Scale-up:** reproducible isolated episodes, offline/online experience buffers, checkpointing, held-out evaluation, graphics throughput, GPU-aware visual representation/world-model RL (DreamerV3 or a measured alternative), hierarchical skill reuse if useful.
5. **Long-term Any%:** survival/resource economy, Nether/End progression, battle, exit detection and test-seed generalization.

Success requires measuring completed runs on seeds never used for training, with clear criteria for timeouts/deaths, in-game ticks, wall-clock resources and confidence intervals. Catastrophic failures and reward hacking count as findings.

## Reset and throughput cautions

The full-restart benchmark copies the saved `DemoWorld` template before each run, making a clean starting snapshot but paying a ~tens-of-seconds startup cost. The persistent benchmark teleports/heals/clears the player within one live world in ~0.1s; blocks, mobs, dropped items and RNG persist. It must never be advertised as independent-world random-seed training.

Separate active simulation throughput from end-to-end throughput and from neural training throughput. Rendering at 300 steps/s does not imply an agent can train at that rate after adding a neural model.

## Dependency baseline

MCio controller `mcio_ctrl==1.5.2`, Minecraft Java 1.21.3 Fabric/MCio, Python 3.14 tested. The `cbor2<6` override is essential on the current released controller because newer CBOR decoding caused blank viewer observations. The GitHub project is its own research harness; it does not vendor Minecraft, Fabric, MCio, Mojang runtimes or user saves.
