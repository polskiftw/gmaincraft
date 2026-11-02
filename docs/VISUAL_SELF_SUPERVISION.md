# Visual learning, temporal memory, and real mined-block rewards

**Research ledger — 2026-11-02.** Read with `FACTOR_POLICY_REVIEW.md` and
`COURSE_CORRECTION.md`. This project still targets learning to finish
Minecraft Any% on unseen seeds from RANDOM initial weights, **without
pretrained outside models, human demonstrations or game-state oracle inputs
at policy inference time**. The following are limited, actual experiments
and deliberately do not claim speedrunning competence.

## 1. Critical camera/HUD observation fix

We inspected the **actual RGB frames** saved in ignored local PPO replay
(`runs/joint_factorized_recordpilot01/`). The rendered 84x84 player
observation was partly covered by large **red Minecraft command-error chat**
from privileged setup (`clear @s` when empty, `fill` with no matched blocks,
`effect clear` when none). The chat obscured the lower world/crosshair region
and created unwanted changes unrelated to motor actions.

The training launcher now calls `ensure_clean_visual_options()` in
`src/gmaincraft/vision_settings.py` *before* opening the dedicated MCio
instance. It atomically and idempotently writes ONLY `chatScale:0.0` and
`chatVisibility:0` into **MCio DemoInstance** `options.txt`; all other settings
and the user's ordinary Minecraft instance are untouched. The crosshair,
hearts and hotbar remain in rendered RGB. Local actual screenshot
`artifacts/vision_frame_scalezero01.png` shows the clean view. A defined
205x65 lower-left red-pixel diagnostic went from **1646 to 418** (remaining
red includes normal HUD hearts). Never treat this pixel count as a general
model-performance statistic.

Important negative control: `chatVisibility:2` (hide chat) caused MCio's
teleport/camera reset observations to stick at yaw=0,pitch=0 even though
player position changed. This was **reproduced**, and code now deliberately
uses the working zero chat SCALE instead. `gamerule sendCommandFeedback false`
failed to remove error overlays and was removed. Existing pose/inventory
reset guards remain enabled.

## 2. Fully self-collected visual action-effect data

`experiments/visual_dynamics_collect.py` runs randomized ordinary Minecraft
movement/look/attack in controller-reset, physically real-block training
scenarios. The collector never passes world coordinates or hidden target
information to the model, and never uses actions taught by people. It
stores only policy-visible RGB, its OWN chosen normal action factors, and
ordinary timing/terminal flags in versioned `rollout_*.npz`, under ignored
local `runs/visual_dynamics_TAG/`.

- `dyntrain01`: **13 completed** episodes x 40 actions, of 14 planned. The
  bridge timed out after saving 13 files; the leftover owned child was
  terminated. **Do not claim 14 completed.** There are 507 valid RGB
  before/after action pairs after dropping terminal boundaries.
- `dynval01`: **four separate** episodes x 35 decisions, 136 valid pairs,
  with independent scenario and action RNG seeds. These are held-out from
  the training data but share the SAME flat-arena world/visual design.
- `move6train01` / `move6val01`: camera and attack controls disabled, random
  six-tick movements only. 351 train pairs from nine new episodes and 156
  validation pairs from four new episodes, different RNG seeds.
- All local datasets, checkpoints, game saves and screenshot files are
  **Git-ignored**. The Git repository contains reproducible code and docs,
  not personal files or local training media.

## 3. Which visual effects are actually predictable?

`src/gmaincraft/inverse_dynamics.py`, `temporal_pixels.py`,
`recurrent_inverse.py` and their CLI training scripts use initially random
weights; all training occurs on the RTX 4070 SUPER through PyTorch/CUDA. The
input is the rendered frame before and after the learner's OWN normal action;
target labels are solely the action chosen by that learner. This is
self-supervised action-effect learning, **not** a pretrained external model
or an expert-demonstration clone. No Minecraft world seed, target or player
coordinates appear in saved training inputs.

**Validation accuracy on separate, same-arena episodes (not unseen Minecraft
worlds):**

| Experiment / pixels | Movement (6 classes) | Camera (5) | Attack (2) |
|---|---:|---:|---:|
| Majority-label baseline | 14.7% | 23.5% | 52.2% |
| Feed-forward CNN, raw pair `visualinv01` | **15.4%** | **42.6%** | **84.6%** |
| CNN, explicit frame difference `visualinv_delta01` | 21.3% | 44.9% | 72.8% |
| Movement-first CNN, then joint training `visualinv_warmmove01` | 21.3% | 43.4% | 79.4% |
| GRU, causal previous action and multiple frames `recur01` | 20.6% | 44.9% | 81.6% |

In the **movement-only** test, where the camera does NOT turn concurrently,
movement recognition improved to **38.5%** (`move6inv01`) versus a **14.7%**
majority baseline. Frame-difference encoding in this setting scored **28.9%**
(`move6_delta01`), worse than plain RGB. The other two heads are constant
and **their accuracy is not a meaningful metric** in movement-only runs.

**Interpretation:** the same visual CNN can learn changes caused by camera
movement and attacking, and can detect movement when the camera is held
steady; it currently fails to reliably separate simultaneous movement and
camera motion from small, noisy image pairs. A small GRU with strictly
causal prior actions modestly improved movement prediction but did NOT
solve it; an explicit difference channel traded some attack accuracy for
slightly better motion perception. These are only 136 validation pairs
in four same-arena episodes: uncertainty is substantial, and no untouched
world test has been attempted for these *offline* models.

The `train_recurrent_inverse.py` GRU has memory during this **diagnostic**
but is NOT integrated into the Minecraft PPO player. Do not claim the live
policy gained long-term memory. The encoder-only `.pt` artifacts are local.
`joint_log_factorized.py --vision-init-tag TAG` can optionally initialize its
RGB CNN from a **raw-image** self-collected action encoder, with random
policy heads. This option is implemented and unit-tested, but downstream
Minecraft skill improvement from such an encoder has NOT been demonstrated.

## 4. Longer actions and authentic Minecraft event rewards

New factorized action interval flag: `--decision-ticks 6` keeps the old V2
policy; `--decision-ticks 12` is an explicitly distinct V3 action/checkpoint
schema. Every decision still contains independently selected movement,
look and attack, with a sustained mouse hold across consecutive choices.
Neither interval represents human demonstration, injected game state, or
a scripted speedrun. V1 and V2 model checkpoints remain loadable and
unchanged. New schemas are checked on frozen natural-world evaluation.

`v3clean01` began a 28-episode GPU training plan but the bridge timed out
after **15** recorded episodes. Its **last valid atomic checkpoint** was
saved after episode **14**, with two actual inventory-success stationary
training cases among the first five stationary scenarios. The partial run
is NOT reported as a completed 28-episode result; unrestricted cases still
had zero successful pickups. Preserve that checkpoint but do not inflate
its sample count.

`MCioAgentBridge` also now privately accumulates the actual `minecraft:mined`
and `minecraft:picked_up` event options from **every intermediate game tick**.
The unchanged `AgentView` provides **pixels only** to the model.
`eventstats01` confirmed a real oak log increments both counters by one
on ordinary mouse-input mining and pickup.

### Crucial cumulative-stat contamination fixed

Minecraft's `StatsUpdateOption` contains cumulative values and sometimes
omits them from new initial packets. Previously, each new per-episode bridge
could start at stat 0 and receive the absolute cumulative mined count N on
its first new mining event. The first attempted reward pilot `v3event01`
therefore falsely counted **1,2,3,4,5,6** newly mined blocks in six
successive stationary episodes—**21 reported** despite only three planted
logs per episode. **These event counts and reward effects are INVALID.**
The file and checkpoint are preserved for forensic continuity, but MUST NOT
be used as clean RL training or evidence of event reward improvement.

`CumulativeMinedTracker` in `src/gmaincraft/reward_events.py` is now owned
by the **world-session arena** across all per-episode bridge recreations,
and converts absolute stats to genuine new-block deltas. Tests cover
zeroed fresh packet caches, old cumulative counts, invalid decreases and
reward caps. A real Minecraft **mine -> do NOT attack -> mine** experiment
`statfix01` yielded exactly **1,0,1 newly mined blocks**, despite later
observation packets reporting cumulative totals: the no-attack episode
received zero reward. This is the clean, verified event-reward path.

`v3eventfixed01` correctly warm-started from the **pre-bug** `v3clean01`
checkpoint under new training/policy RNG seeds. It completed **10 new CUDA
training episodes**: five restricted stationary, two aim-only, and three
unrestricted near. It mined **three actual oak blocks**, with zero mined
counts exceeding the physically available three per fixture, but acquired
**0 inventory logs**. Results from `v3event01` are excluded from that run.
No meaningful sample-efficiency advantage over unbonused RL is claimed.

### Frozen paired outcome with the corrected stats

Both V3 checkpoints were frozen and tested on exactly the same new copied
arena scenarios, `v3fixedaudit01`, validation seed 3737:

| Frozen model | Restricted stationary: mined | Restricted stationary: inventory | Fully unrestricted near: inventory |
|---|---:|---:|---:|
| `v3clean01`, checkpoint after 14 clean training episodes (2 PPO updates) | 4/4 | 0/4 | 0/4 |
| `v3eventfixed01`, additional 10 corrected-event training episodes | 4/4 | **2/4** | **0/4** |

Different *training budgets* prevent attributing the pickup difference to
the new event reward. In restricted tests, the tree is already aimed and
look/move channels disabled; the model has NOT learned to independently
seek trees, hold aim during movement, or reliably approach drops. **Mining
must not be confused with first-log acquisition**, which specifically
requires an actual inventory item.

`natural_world_benchmark.py` also loaded the new 12-tick V3 checkpoint in
normal Minecraft 1.21.3 noise-generated seed **31415927**. Random baseline
and frozen V3 both started at exactly **(-15.5,76.0,16.5)**, yaw/pitch0,
and acquired **0 logs in 240 game ticks**. This verifies fair natural-world
integration, **not successful survival transfer**.

## 5. Significant MCio runtime regression (unresolved)

These tests also revealed an unexplained **real Minecraft I/O slowdown**.
Earlier V2 GPU training processed ~300 game ticks in ~0.6-0.8 seconds per
controlled episode; new runs processed ~300 game ticks in ~4.0 seconds.
Instrumentation in `joint_log_factorized.py` (`profile01`) recorded for
an ordinary 300-tick episode approximately:

- **3.876 seconds** sending normal vanilla game actions and receiving MCio
  observations (`game_io_wall_s`)
- **0.09 seconds** of the CNN policy inference on CUDA
- **0.015 seconds** of frame resizing/processing

A control using **NO Torch at all** took **5.008 s/300 game ticks** with
320x180 frames. Decreasing to **160x96** barely changed it (**4.892 s**).
Removing the temporary `sendCommandFeedback` gamerule made no difference;
reverting the chat-scale option also left model episodes at ~3.9 seconds.
Do not blame GPU inference, the smaller RGB encoding, or the 12-tick
controller without evidence: 6-tick and 12-tick versions BOTH measured
~3.9 s/300 ticks. The same NVIDIA hardware OpenGL and working CUDA were
verified. The cause could involve host/system state, Gamescope, the MCio
synchronization path, driver timing or other dependencies; **not yet known**.
Do not widen the user's filesystem sandbox or alter host services to guess.

The nested Gamescope process still often exits with status 139 **after its
Python child has completed**, and outer timeouts can exit 124 even when an
atomic checkpoint/JSONL had been saved. Always inspect actual child logs,
checkpoint metadata and per-episode outcome counts. If a timeout leaves
owned orphaned Minecraft processes, terminate ONLY those owned descendants;
do not kill unrelated host sessions.

## 6. Reproducibility and next research decision

All scripts operate in connected Gentoo Shell `/workspace`, Python `.venv`,
NVIDIA OpenGL and CUDA; existing worlds, older checkpoints and previous
experiments are never overwritten. Use **fresh** run tags. Examples:

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python -m unittest discover -s tests -q
# Record new random own-action RGB (use different train/validation seeds):
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/visual_dynamics_collect.py --run-tag NEWTRAIN01 --split train --scenario-seed 1234 --action-seed 12345 --episodes 8 --decisions 40 --visible
# Evaluate inverse dynamics; requires separate NEWVALID01 run first:
.venv/bin/python experiments/train_inverse_dynamics.py --run-tag NEWINVERSE01 --train-data runs/visual_dynamics_NEWTRAIN01 --validation-data runs/visual_dynamics_NEWVALID01 --epochs 30 --device cuda
# Restricted per-episode game-event verification:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/mine_stat_reset_probe.py --run-tag NEWSTATS01 --visible
# New V3 training with 12-tick interval and *verified* event reward:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_factorized.py --run-tag NEWV3FIX01 --decision-ticks 12 --episodes 10 --max-ticks 240 --update-every 5 --mine-event-bonus 6 --device cuda --skip-eval --visible
```

Before another large GPU study, measure MCio throughput and repair the
visual-action learning deficiency using **independent, agent-visible
experience**. Evaluate the same trained policy on paired held-out real
inventory tasks and on *genuinely unseen seeded natural survival worlds*.
Then add continuous wood pickup, inventory/crafting and later Minecraft
progression only after unrestricted **first-log** success is demonstrated.
The complete from-scratch Any% objective remains UNSOLVED.
