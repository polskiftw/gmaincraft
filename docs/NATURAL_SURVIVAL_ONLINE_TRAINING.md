# Natural survival learning is now the primary roadmap

**November 3, 2026 — from-scratch, unscripted Minecraft Java survival PPO.**
This supersedes earlier instructions to spend most development time on
pre-aimed oak trunks and scripted collection arenas. Those experiments
remain useful *unit diagnostics*, but they are not the training objective.
The user specifically wants to leave a Minecraft agent training for a long
period in ordinary worlds, with game-memory observations rather than the
historical RGB-only approach. This document describes an actually run and
resumed end-to-end natural-world learning path, not a simulated training
plan or proof of a functional speedrunner.

## Get it training on your Gentoo sandbox

From an ordinary terminal with the Gentoo Shell's `/workspace` sandbox
available, run in `/workspace`. It only edits the disposable MCio data,
local models/logs and uniquely named natural-world copies. It does NOT
install a service, daemon, global Minecraft mod, shell startup hook or
host performance changes.

1. Prepare a cohort of **eight genuine, independently seeded vanilla noise
   survival worlds**, once (safe to repeat with the same parameters after
   interrupted world generation). Minecraft's vanilla server creates the
   terrain; the preparation tool verifies seed, generator and a
   deterministic zero spawn-radius; no trees/items/teleports are staged.

   ```bash
   cd /workspace
   .venv/bin/python tools/prepare-natural-training-worlds.py --tag longworlds01 --count 8 --seed 917211
   ```

2. Start a **foreground** unattended training run. You may leave this
   terminal open while it plays, or use a terminal multiplexer you already
   trust. The `100000` is the total count of **six-game-tick decisions**;
   no estimate of real-world completion time is guaranteed. Actual game
   simulation is much faster than real-time in the tested headless backend.

   ```bash
   tools/run-long-natural.sh start longtrain01 longworlds01 100000
   ```

3. Follow progress from a second terminal (the rotating latest-console
   symlink is in Git-ignored logs; the full JSONL is also persisted).

   ```bash
   tail -f logs/natural_console_longtrain01_latest.log
   ```

4. Request a **graceful stop** from another terminal, even if Minecraft is
   currently running. The trainer finishes its current normal input tick,
   exits Minecraft normally and saves its **model, optimizer, RNG, explored
   places, cumulative player-owned counters and world progress**. It
   removes the one-time stop marker when successfully paused.

   ```bash
   touch logs/natural_online_longtrain01.stop
   ```

5. Resume the exact SAME run and its evolving survival worlds, optionally
   raising the **total** decision target (not adding that many decisions).
   The optimizer is NOT reinitialized, and previous trained weights and
   source-world saves are NOT replaced.

   ```bash
   tools/run-long-natural.sh resume longtrain01 longworlds01 150000
   ```

Other useful files (all local and Git-ignored):
`checkpoints/natural_online_<tag>.pt` stores the policy plus optimizer and
memory; `logs/natural_online_<tag>.jsonl` is the append-only training
journal; `mcio-data/instances/DemoInstance/saves/MDL_NaturalOnline_<tag>_XX`
are the actual evolving survival saves. `logs/natural_sources_<cohort>.json`
records source names and exact vanilla seeds. No binary checkpoint or world
file is pushed to the source GitHub repository.

**Crash limitation:** A clean stop or scheduled world rotation has been
verified; the checkpoint is committed after Minecraft has shut down and
saved the current world. The runner deliberately does NOT claim lossless
recovery after sudden power failure, SIGKILL or process hard-crash in the
middle of a segment; the evolving world may have auto-saved separately
from the last model checkpoint. Preserve the journal and worlds if this
happens, reconcile before resuming, and do not count orphaned journal
entries as valid training. This is a near-term hardening goal.

Nested headless Gamescope on the current Gentoo setup sometimes exits
**139 during its own teardown after the Python training child has saved
complete valid results**. `tools/run-long-natural.sh` checks that the
journal actually grew and ends in a valid `completed` or `paused` record
instead of reporting success merely because the wrapper exited. A Python
traceback, missing/unchanged journal or failed tests are still failures.

## What the agent actually does

`src/gmaincraft/natural_online_policy.py` implements a freshly randomized
**50,913-parameter** PyTorch actor-critic, trained on NVIDIA CUDA. On
**every six Minecraft game ticks**, it chooses together:

- one ordinary movement: idle, W, S, A, D or W+SPACE;
- one camera delta: none or six discrete mouse directions;
- whether to keep ordinary left-click attack pressed or release it.

There is NO setblock/teleport/admin action/skill switch/known route in the
policy or training rollout. A frozen evaluator uses identical inputs and
normal actions, with separate never-trained seeds; selectable greedy,
policy-sampled, random-action and independently untrained policy controls.

**Current policy observation** (versioned
`natural_memory_online_v1_first_hit_objects_ppo6`):

- Own exact Minecraft XYZ/yaw/pitch, health, hunger/saturation, air, selected
  hotbar, accessible player inventory counts and remembered action/ticks;
- Minecraft's real vanilla crosshair target and distance, plus the current
  player-owned crack stage (unknown = -1, even while attack is active);
- **Top 24 first-collision camera-visible block surfaces** from the existing
  opt-in 13x9 first-hit ray grid, with relative positions, ray coverage,
  screen coordinates, lighting and full registry IDs learned through
  trainable UTF-8 byte embeddings. The game reports these only after an
  explicit synchronized last-action request. No seed, loaded-chunk dump,
  hidden block lookup, reward geometry, whole-world entity list or global
  destination is a model input. RGB is intentionally not required.

**Visibility caveat:** the first-hit surface sampler is bounded, geometric
and direct-memory, but **not yet a fully calibrated human-perceivability
threshold**. 13x9 misses some genuinely rendered small/off-centre blocks;
glass/leaves, darkness, motion and fine transparency still need research.
The 24-record budget is an architectural bound, NOT a declaration that
other visible objects are impossible to see. It does not yet include
nearby dropped item entities. The previous calibration research remains
valid and is now an incremental improvement to the natural training
pipeline, not a reason to defer natural-world learning indefinitely.

`src/gmaincraft/natural_online_session.py` remembers visited 2x2-block
cells and already observed first-hit block-ID/position records **per world**
(with finite memory bounds), plus actual player's cumulative Minecraft
mined/picked stats and current own inventory, not a hidden map. It grants
ordinary exploration feedback for first-time cell/surface discovery.
The `wood_discovery_v2` reward adds one-time feedback when the camera
**actually first sees a wood-block surface**, instead of waiting for a
statistically rare full chop-and-pickup. The player still must learn how
to approach, aim, mine and collect with its own actions; nothing is placed
for it. Separate small rewards follow real mined block/wood events and
**actual wood inventory acquisition**, not synthetic success claims.

The original `explore_v1` reward is preserved for reproducing its local
checkpoint `natreal_1103a`. Do NOT silently change reward profiles on
resume. New long runs use explicitly versioned `wood_discovery_v2`.

## Actual live experiments, not demonstrations

All numbers below are from **real Minecraft Java 1.21.3 survival** with
NVIDIA-rendered MCio observations, no training-world block edits, no
stage/special-action controller commands and no pretrained weights. The
new pilot checkpoints and original diagnostic worlds are preserved locally.

### Pipeline reliability

- `natpilot_1103a`: two verified natural seed worlds, 20 real decisions and
  four PPO updates; after fixing a CUDA checkpoint RNG-device mismatch,
  successfully **resumed 20 -> 40 decisions and four -> eight updates**
  on the same evolving world saves and remembered cells.
- `tools/run-long-natural.sh start wrappertest_1103a freshworld1103 10`
  exited **success after checking the durable journal**, 10 real game
  decisions and one PPO update. `resume ... 30` continued SAME model/save
  to 30 decisions and two PPO updates; one config record, not two new runs.
- A preexisting `logs/natural_online_stoptest_1103a.stop` made the helper
  pause cleanly after the Minecraft world started, save a resumable zero-
  decision checkpoint and clear the one-shot stop marker. SIGTERM handler
  also requests normal end-of-action stop.
- `tools/prepare-natural-training-worlds.py` actually generated and
  verified two wholly new normal survival seed sources, tagged
  `freshworld1103`: **-382102578238778906** and
  **2549380921493329309**. The generator is NOT a test arena builder.

### Unscripted training results

| Scratch training mode | Decisions | PPO updates | New cells | New first-hit surfaces | New wood surfaces | Actually mined blocks | Real logs acquired |
|---|---:|---:|---:|---:|---:|---:|---:|
| `explore_v1`, `natreal_1103a`, three survival worlds | 480 | 12 | 114 | 5,727 | not measured | 40 | **0** |
| `wood_discovery_v2`, `natwood_1103a`, three survival worlds | 720 | 12 | 136 | 7,499 | **366** | 21 | **0** |

All 21 mined blocks in the wood-profile run were **non-wood**. One earlier
exploratory Minecraft packet had active breaking with crack stage -1;
this was verified in raw game output (two occurrences across an earlier
240-decision run), and the Python adapter now correctly returns **unknown
stage** rather than rejecting the entire training session. No raw packets
were secretly rewritten and old client versions remain compatible.

Actual per-action Minecraft game I/O in warmed real natural survival
scenes often reached **about 2-3 ms per game tick** (roughly 330-500
Minecraft ticks per wall second on this NVIDIA/CPU/Gentoo environment),
with world-start overhead separately measured. This is not a promise for
every render profile, terrain density or user machine.

### First honest held-out evaluation

Two *never-trained* natural source seeds were scored for each comparison;
all with only ordinary 6-tick actions, **120 decisions total**. These
are short first baselines, NOT statistical performance proof.

| Frozen action source | New 2x2 cells | Seen first-hit surfaces | Seen wood surfaces | Actually mined blocks | Natural log inventory |
|---|---:|---:|---:|---:|---:|
| `explore_v1` trained, greedy | 11 | 1,503 | not measured | 2 | 0 |
| `explore_v1` uniform random | 20 | 2,437 | not measured | 1 | 0 |
| `wood_discovery_v2` trained, **greedy** | 2 | 875 | 42 | 0 | 0 |
| `wood_discovery_v2` trained, **sampled** | 24 | 725 | 44 | 5 | 0 |
| `wood_discovery_v2` uniform random | **28** | **1,575** | **59** | **15** | 0 |

The V2 held-out seeds were 2549380921493329309 and 246813579, completely
absent from that V2 model's training list. Greedy policy behavior currently
collapses into stationary/repetitive actions; action sampling, as used in
training, partially resolves this, but **uniform random still explored and
mined more in these short comparisons**. This is WHY we need long-running
real-world training and better intrinsic reward/action training, not a
reason to claim success or return to scripted logs. Nobody has measured
natural first-log success yet. The objective is actual survival inventory
wood and eventually complete Any% on held-out random seeds.

## Next agent priorities

1. **PRIMARY:** keep improving / exercising THIS resumable natural-world
   trainer, and validate learning curves against random/untrained actions
   on new held-out natural seeds. Investigate poor greedy exploration and
   hard-crash model/world alignment. Never treat greater novelty as
   proof of competent survival or speedrunning.
2. Add a bounded, player-visible dropped-item *memory* observation with
   actual FOV/occlusion/size limits and probe it in ordinary natural
   Minecraft; switch current policy schema in a new version rather than
   silently edit old checkpoints. Keep input train/frozen parity.
3. Expand inventory/equipment/crafting/use controls and action duration,
   and devise a non-scripted objective curriculum that can genuinely move
   toward collecting logs, making tools and later Any% completion. Wood
   observation and skill learning should guide the roadmap, not long runs
   of a frozen pre-aimed scenario.
4. Retain the old controller-placed log and dropped-item experiments as
   quick deterministic **diagnostics only**. Do not devote the majority
   of the project to repeating fixed fixtures without measurable transfer
   to fresh survival seeds.

No ongoing training daemon has been installed or scheduled by this
change. The user can choose when to run the **foreground** long job above
and stop/resume it without any automatic account-wide agent service.
