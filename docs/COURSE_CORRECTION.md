# Course correction: toward an actual from-scratch Any% agent

**Current user intent is clarified in
[OBSERVATION_POLICY.md](OBSERVATION_POLICY.md):** a speedrunner learning from
random initialization with **direct engine-memory reads of player-knowable
stats, inventory, position, and calibrated visible blocks during training
AND gameplay**. RGB may supplement that state; it is not a project-wide
requirement. The following experiments document the earlier RGB-only
prototype, which was an assistant-imposed restriction, NOT user intent.

**2026-11-02 research decision:** preserve all earlier learning experiments
and their checkpoints, but do not extrapolate toy-arena success into general
Minecraft skill. Make transfer into untouched, naturally generated survival worlds the
primary external benchmark. Every future curriculum improvement must be
compared against this benchmark at explicit fixed evaluation intervals.

## Historical pixel-only prototype: what was implemented

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

## New visual-memory / real-event research gate (Nov 2, 2026)

The latest scope and definitive run ledger are documented in
[`VISUAL_SELF_SUPERVISION.md`](VISUAL_SELF_SUPERVISION.md). Actual policy
screenshots were obscured by command chat; the dedicated MCio test instance
now hides that overlay through `chatScale:0.0`, leaving vanilla HUD and camera
working. It does not modify user's normal Minecraft options.

CUDA models learned to identify camera and attack effects from the agent's
own random normal Minecraft actions, using only RGB before/after frames.
Camera 42.6% vs a 23.5% majority baseline, attack 84.6% vs52.2%, but
movement15.4% vs14.7% in the simultaneous-action scene. With camera held
still, movement recognition improved to38.5%. A causal GRU and pixel-delta
encoder were tested, but are still offline diagnostics, NOT deployed online
memory or general-world first-log learning.

Per-game-tick Minecraft mined/picked-up statistics are now evaluator-only
signals; policy remains pixel-only. **Do not use `v3event01` as scientific
evidence**: reset of per-episode MCio statistic caches gave fake cumulative
new-block rewards. A shared world-session counter fixed it, verified with
real mine/no-attack/mine=1/0/1. Valid `v3eventfixed01` trained 10 new CUDA
episodes (3 true mined,0 collected); paired frozen *restricted pre-aimed*
test: mined4/4 for both old/new checkpoints, collected0/4 vs2/4;
unrestricted movement/look still0/4 for both. Natural Minecraft seed31415927
also gave0/240-tick logs for random and V3 at same spawn. Training-budget
differences prohibit attributing better pickup to the event bonus itself.

**Do not blindly scale PPO.** The major unresolved operational problem is
MCio synchronous I/O throughput (~3.9s/300ticks) versus CUDA policy
inference~0.09s, and no-ML controls are also slow. Investigate separately
without weakening host sandbox or guessing at NVIDIA driver changes.
Focus on disentangling camera/movement perception, temporal control, and
reliable unrestricted *actual inventory* pickup before adding crafting stages.

## Latest hard evidence: independent controls beat the learned V2 policy

V2 now has simultaneous movement/look/attack selections every 6 game ticks;
actual vanilla 20x6 continuous mining was confirmed. Nonetheless three
frozen V2 RGB PPO checkpoints each returned **0/12** real logs on the same
held-out nearby trunk cases where a simple scripted attack-and-approach
baseline scored **8/12**. An explicitly privileged geometry oracle scored
11/12, validating fixtures. A restricted attack-only policy achieved 2/4
held-out when look/movement were disabled, but progress **did not transfer**
to unconstrained trials. Neither blocked staging nor interleaved staged
training with sustained-aim scalar shaping has fixed full-control learning.
A newly prepared survival seed 246813579 returned 0 logs for frozen V2 and
both random/forward controls, with strictly identical spawn positions.

Consequently **stop scaling this same PPO CNN on faith**. Diagnose visual
representation and credit assignment using the opt-in versioned local
RGB/action trajectory recording. Evaluate memory and more sample-efficient
algorithms against the same real inventory, normal-action, held-out seed
gates. See [FACTOR_POLICY_REVIEW.md](FACTOR_POLICY_REVIEW.md) for complete
methods, reproducible commands, frozen SHA256 and limitations.

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
`bounded Minecraft memory state + optionally RGB/audio + learned history
-> encoder/state -> policy/value or world model -> vanilla Minecraft actions`, with full episode experience, reward/termination events
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

## Historical next-work list (superseded by NEXT_SESSION and OBSERVATION_POLICY)

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
4. **After addressing MCio throughput**, integrate skilled-player-knowable
   memory-derived numeric/player/inventory state into the policy and measure
   calibrated visible-block/entity thresholds. Then build a joint learned
   look/move/attack/pickup policy with real trajectory-level credit assignment,
   and vary natural scenes early. Keep
   independent untouched-world evaluation on every milestone.
5. Extend one continuous run from wood acquisition to crafting before adding
   isolated sophisticated subskill suites. Keep uncertainty and single-GPU
   compute limits explicit.
