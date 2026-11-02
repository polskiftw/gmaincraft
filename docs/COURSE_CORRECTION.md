# Course correction: toward an actual from-scratch Any% agent

**2026-11-02 decision:** preserve all earlier learning experiments and their
checkpoints, but do not extrapolate toy-arena success into general Minecraft
skill. Make transfer into untouched, naturally generated survival worlds the
primary external benchmark. Every future curriculum improvement must be
compared against this benchmark at explicit fixed evaluation intervals.

## What has been implemented

1. `src/gmaincraft/agent_interface.py` supplies a consistent, narrow policy
   contract: an immutable copy of the actual RGB screenshot (`AgentView`) and
   bounded keyboard/mouse actions (`AgentAction`). No hidden target coords,
   block maps, seeds, teleport commands, inventory state or controller scoring
   are included in the policy-facing object. A bridge maps normal input into
   MCio packets with an empty administrative command list. This is **not** a
   security sandbox for arbitrary untrusted Python policies. Future human-like
   HUD, inventory UI and status observation rules require separate decisions.
2. `experiments/natural_world_benchmark.py` provides an actual distinct-world
   *first wood* benchmark. `--prepare` generates normal Minecraft 1.21.3
   survival worlds using a real 64-bit seed, verifies `minecraft:noise` terrain,
   checks world-generation seed, and writes `spawnRadius=0` into **only its
   newly generated benchmark templates** to prevent spawn scatter between
   policy modes. Every policy launches its own fresh copy of the same source
   seed; the source world itself is never played. No level editing, admin
   commands, cheats or teleports occur in scored gameplay.
3. Diagnostic modes: `idle`, `random`, `forward_attack`, and `legacy_probe`.
   The last uses the previously trained, frozen camera and pickup weights but
   still has the **scripted** 100-tick mining-hold and two-stage handoff. It
   is not a general RL agent. All modes receive the same pixel-only view.
4. Score is actual Minecraft inventory acquisition of overworld logs/wood or
   Nether stem/hypha items. Stop on success/death/tick budget and record game
   ticks, decisions and wall seconds in local JSONL. A Minecraft *game tick*
   is a simulation tick, **not** necessarily 1/20 wall-clock seconds with MCio.
5. Existing experiments remain runnable and unchanged. Checkpoints, local
   worlds and CSV/JSONL files remain ignored by Git. Unit/CI tests validate
   the contract and planning logic; actual end-to-end benchmark must be tested
   separately on the Gentoo GPU.

## Measured pilot (before spawn fairness fix)

A normal noise-generated survival world (seed `314159265`, not the old flat
DemoWorld) was created and verified by the Minecraft 1.21.3 server. A
100-simulation-tick comparison of all four modes returned **0 natural logs**
for every mode, with **no game errors**, and saved local
`logs/natural_world_naturalpilot01.jsonl`. All four completed the end-to-end
Minecraft, RGB policy, normal input and inventory-check pipeline. This is a
**short smoke test**, not a statistically meaningful first-log benchmark.
The original source world had Minecraft's default `spawnRadius=10`, so the
fresh comparison copies started at slightly different coordinates; **do not
interpret this pilot as a precisely paired evaluation**. Future `--prepare`
sets `spawnRadius=0` and the evaluator refuses sources without it. The
generator server is configured to bind only to `127.0.0.1`; no LAN exposure
is required. Gamescope
still exits 139 after the successful child benchmark completion.

## Fixed-spawn verification (completed after the pilot)

World seed `271828183` was generated as verified normal `minecraft:noise`
terrain. `spawnRadius=0` was written only into that NEW source template.
Real Minecraft/MCio then ran **two distinct fresh copies** of this seed:
`idle` and `forward_attack`, each for **25 synchronized game ticks**.
Both started at exactly **(0.5, 77.0, 0.5), yaw 0, pitch 0** as recorded
from Minecraft's privileged observation and saved only in local evaluation
JSONL. Both returned 0 natural logs (expected for such a short smoke test).
Results are in ignored `logs/natural_world_fixedspawn01.jsonl`. The result
verifies fixed matching spawn *for these two modes on this world*, not across
all seeds or all game versions. More seeds / longer episodes are still needed.
As before, Gamescope exited 139 after the Python child completed.

## CUDA GPU is confirmed available after bridge restart

The actual NVIDIA GPU is now usable for learning, not just OpenGL rendering.
`cuda01` real Minecraft trained 12 episodes with 3 successful CUDA PPO
updates and evaluated 4 frozen held-out episodes: **0/12 and 0/4** log
acquisitions. A synthetic CNN batch benchmark shows ~6.82x faster CUDA
model updates, but not an established end-to-end training speedup. Frozen
cuda01 decisions appear to overuse 80-game-tick mining actions; focus on
agent action/exploration design before scaling training compute.

## New joint PPO prototype (built after the first transfer gate)

See [JOINT_LEARNING.md](JOINT_LEARNING.md) for architecture, code and exact
smoke results, and [GPU_ACCELERATION.md](GPU_ACCELERATION.md) for verified
NVIDIA hardware OpenGL / blocked CUDA access. A single randomly initialized
CNN policy can now choose joint movement/look/mine actions and update from
complete real-game trajectories with timed PPO. Two train plus two frozen
held-out episodes completed, 0 logs; one frozen natural-world evaluation
completed on seed 161803398, 0 logs. These results only validate the
pipeline, not successful learning/generalization. The fixed 80-tick mining
macro and two-frame history remain major limitations. Do not mark the
long-term joint control or first-log milestones complete.

## Research architecture / honest boundaries

These parts are **NOT** implemented:

- A learned joint movement + look + mine + interact + UI controller.
- Robust learned visual encoder, temporal history, recurrent memory or learned
  world representation.
- Proper multi-step credit assignment (PPO/GAE, recurrent actor-critic,
  world-model RL or similar). The old linear REINFORCE update uses immediate
  rewards; it is an educational prototype, not the planned final learner.
- Portable general-tree recognition, survival/food/crafting/combat/navigation,
  Nether/End progression or dragon fight.
- Validated success on several untouched random world seeds with meaningful
  confidence intervals, much less a complete Any% speedrun.

Design the next learner around one common policy boundary and trajectory data:
`RGB frames -> learned encoder -> memory/state -> policy/value or world model
-> Minecraft actions`, with full episode experience, reward/termination events
and training/evaluation isolation. Benchmark simple actor-critic/PPO versus a
world-model family **based on measured environment throughput and available
single-GPU memory**, rather than implementing a large algorithm on faith.

## Evaluation standard

- Record distinct **Minecraft world** seeds (not just Python or camera seeds),
  commit/version, model checkpoint hash, tick budget, action space, frame size,
  training seed, and all error/death/timeout outcomes.
- Never train on the final test-seed set. Use distinct train/validation/test
  world-seed splits, and rotate only validation during research. The pilot seed
  is already viewed; don't treat it as a pristine final held-out test.
- Compare the **same saved checkpoint** periodically against idle/random and
  suitable scripted diagnostic controls, using equivalent tick budgets.
- Do not report progression merely because reward improves in an artificial
  arena; require increased real-world success or convincingly diagnose failure.
- Score whole episodes (eventual wood -> crafting -> exploration -> Nether ->
  End -> exit) rather than isolated independent subskills as the objective grows.
  Track actual in-game time for speedrunning *after* establishing reliability.
- Use no-aid assessment: controller can prepare independent worlds and read
  outcomes, but can never summon target resources, teleport, pause threats,
  use `/give`, or alter terrain for the policy during a real-world score.
- Always distinguish instrumentation, scripted diagnostics, teacher-shaped
  learning and fair held-out world evaluation.

## Next work, in order

1. [DONE, limited two-mode pilot] Verify a freshly prepared **spawnRadius=0**
   world in actual game with matching starting coordinates and record observed
   spawn. The runner also rejects mismatched starting poses between modes of
   the same seed. Still validate future seed/mode comparisons.
2. Finish a small randomized-position **controlled-log** comparison. Move
   actual trunks *and* the player so fixed-W is not sufficient; replay old
   frozen models, compare independent baselines, and report failure modes.
   This is a transfer diagnostic, not another indefinite toy-arena detour.
3. Extract reusable episode recording and reward bookkeeping, including
   full trajectories, step/event timestamps, world-seed partitions, checkpoint
   metadata and outcome guards. Include event reward only from the evaluator.
4. Build a minimal **joint pixel-driven** look/move/attack policy with real
   trajectory-level credit assignment; vary natural scenes early. Keep
   independent untouched-world evaluation on every milestone.
5. Extend one continuous run from wood acquisition to crafting before adding
   isolated sophisticated subskill suites. Keep uncertainty and single-GPU
   compute limits explicit.
