# Factorized RGB agent, paired controls, and learning-efficiency gate

**Last verified: November 2, 2026.** This is a research ledger, not a claim
that the agent can speedrun Minecraft or acquire wood autonomously in a new
survival world. See `docs/COURSE_CORRECTION.md` for the original mission.

## Why the action controller changed

V1's 11-way CNN actor selected one action at a time, including a **fixed
80-game-tick hold of left click**. Frozen policy `cuda01` exhausted every
220-tick evaluation episode in just three attack actions and collected zero
logs. This is an inefficient representation of simultaneous vanilla controls.

V2 `src/gmaincraft/vision_factorized.py` keeps its own **separate** randomly
initialized 888,750-parameter RGB convolutional actor-critic checkpoint.
Three categorical heads control movement (6 choices), camera (5), and
attack (2), recomputed every **6 synchronized game ticks**. For example,
the network can choose W + look right + LEFT_BUTTON together. The cumulative
log probability and entropy are calculated over the *active* heads; PPO
updates still use timed, terminal-bounded GAE across whole episodes.
Input is still **two full-RGB 84x84 frames**; no target position or inventory
is passed to the model. Old V1 scripts and checkpoints are preserved.

A separate **positive control** `experiments/continuous_mine_probe.py`
placed a log before scoring, aligned the initial camera, then executed **20
successive 6-tick held LEFT_BUTTON actions** through the real MCio bridge.
Minecraft maintained the hold and awarded **one genuine oak log in inventory**.
This test proves the new motor primitive can mine; it is not learned success.

## Controls that establish what the model should beat

`experiments/factorized_controls.py` ran the SAME 12 new camera/player/trunk
scenarios using held-out Python scenario seed **3737**. Commands are allowed
only for *fixture reset* between episodes; gameplay uses vanilla input.

| Controlled experiment | Real logs / 12 | Meaning |
|---|---:|---|
| Privileged geometry + mouse-aim + attack + approach | 11 | **Instrumentation upper bound**, cheating privileged policy information, NOT a fair visual agent |
| Attack for 120 ticks, then move forward | **8** | Simple unlearned control to beat |
| Forward 12 ticks while attacking, then stationary attack and approach | 3 | Another unlearned control |
| Random factorized normal actions | 0 | Random baseline |

The first `controlv2a` run correctly detected a reset error after a genuine
pickup: MCio showed old inventory before `/clear` had fully propagated.
`VariableLogArena.reset()` now requires **both** correct player pose and zero
observed wood inventory, settles two extra ticks, and refuses contaminated
starts. The valid 12-case data come from **`controlv2b`**, not the aborted
first run. All data are local Git-ignored JSONL.

## CUDA training: empirical results, not predictions

All following models were trained from their own RGB CNN weights (except
explicit warm-start `aimnear01`); PyTorch actually used **CUDA** on RTX 4070
SUPER, and checkpoint optimizer updates completed. Except where otherwise
stated, successful wood always means real Minecraft inventory acquisition.

| Model / run | Real training outcomes | Frozen test outcomes |
|---|---|---|
| `factorv2pilot01` | 40 nearby, 0 wins | 0/4 near, 0/4 broad |
| `attackrl01` (masked stationary warm-up) | 3/32 on stationary aligned tree | **2/4** attack-only restricted test, **0/4** unrestricted near and broad |
| `aimnear01` (warm-start `attackrl01`) | 1/36 aim+attack with movement locked; 0/20 full near | 0/4 stationary restricted, 0/4 aim restricted, 0/4 unrestricted near and broad |
| `mixstable01` (fresh, interleaved phases + sustained-aim teacher proxy) | 0/12 stationary, 0/14 aim, 0/22 full near | **1/4** restricted stationary, 0/4 restricted aim, **0/4 full near**, 0/4 full broad |

**`attackrl01` shows genuine but very narrow learning**: the network learns
to maintain an attack in a nearly aligned fixture when movement/camera heads
are temporarily disabled. This is no substitute for autonomous search/aim.
The staged model then lost that restricted skill during continued training;
we suspect training interference/catastrophic forgetting, but cannot claim
its mechanism has been proved. Interleaving all three tasks and rewarding
sustained geometry-aligned attack did not resolve the full-task failure in
this small experiment. Reward shaping in `mixstable01` is **not** a mined-
block-progress measurement and must never be represented as such.

### Rigorous paired frozen audit

`experiments/factorized_frozen_audit.py` loaded THREE frozen checkpoints on
the exact **same 12 near cases** used by the controls (seed 3737), under a
new copied world. All scored **0/12** versus the simple **8/12** scripted
attack-and-approach diagnostic. This audit also logs SHA256 checkpoint hashes,
minimum angular error, movement, camera choices, and attack duration.

| Frozen model | Mean ending tree distance | Mean best angular error | Problem |
|---|---:|---:|---|
| `factorv2pilot01` | ~12.97 blocks | ~9.05° | Predominantly backward movement; ~5 of 360 game ticks attacking |
| `attackrl01` | ~15.77 blocks | ~7.60° | Attacks continuously while moving and turning |
| `aimnear01` | ~2.61 blocks | ~5.55° | Stays near but **turns right every action** while attacking nonstop, so cannot mine stably |

This is why simply extending PPO from 40 to thousands of episodes has **no
justification yet**. The representation, exploration, reward design, and
cross-skill credit assignment need investigation. More CUDA compute is not
by itself a solution. The 12-case comparison is a development diagnostic,
not a final untouched benchmark, and confidence about the general
population is limited by the small number of cases.

## Independent survival-world gate

`natural_world_benchmark.py` now supports `--modes joint_factorized`
and `--factorized-model TAG`. Frozen checkpoint weights get only rendered
Minecraft pixels, never training masks or teacher rewards; all input packets
contain normal vanilla controls. A fresh noise-generated 1.21.3 survival
world with seed **246813579** was prepared with `spawnRadius=0`, using the
same saved-world template for all three independent copied-world runs:

| Mode | Wood acquired within 360 game ticks |
|---|---:|
| `forward_attack` | 0 |
| `random` | 0 |
| `joint_factorized` (`mixstable01` checkpoint) | **0** |

All three observed **identical spawn (16.5, 124.0, 0.5), yaw 0°, pitch 0°**.
The factorized agent made 60 decisions, finished without Python error and
its SHA256 was logged. This verifies the complete fair evaluation path but
is **not evidence of natural-world skill**. Results are local
`logs/natural_world_factortransfer01.jsonl`.

## Save agent-generated learning data for more sample-efficient research

`--save-rollouts` is an **opt-in** argument of
`experiments/joint_log_factorized.py`. It atomically writes compressed,
versioned `.npz` batches beneath ignored `runs/joint_factorized_TAG/`.
Only the agent-observable `RGB` state `(N,6,84,84)` and its selected
three-head actions, scalar reward, value, old log probability, duration,
terminal flags, and active-head masks are stored. Batch metadata include
schema, run tag and episode/stage order; they **do not include** hidden
world/target/player coordinates, server commands, NBT or personal files.

`src/gmaincraft/rollout_store.py` validates shapes/finite values, refuses
existing output filenames and loads with NumPy `allow_pickle=False`.
`recordpilot01` generated a **real CUDA-backed** 2-episode 120-tick training
rollout; the saved 135,670-byte `.npz` loaded back successfully with **40
agent decisions and 240 game ticks**. Two terminal transitions were present.
The model and data are local, not committed to GitHub.

These old-policy trajectories can be used for diagnosis, visual
representation training or future **properly off-policy** learning methods.
Do **not** repeatedly feed them into ordinary on-policy PPO as if they were
fresh experience: stale action probabilities invalidate the naive objective.

## Canonical commands inside Gentoo Shell `/workspace`

All examples require unique run tags; avoid reused worlds/results.

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python -m unittest discover -s tests -q
# Test that a genuine block can be mined via repeated 6-tick input:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/continuous_mine_probe.py --run-tag NEWMECHANIC01 --visible
# Paired controller baselines in a controller-placed log arena:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/factorized_controls.py --run-tag NEWCONTROLS01 --eval-seed 3737 --eval 12 --max-ticks 360 --visible
# New model, interleaved curriculum, with optional local experience capture:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_factorized.py --run-tag NEWV201 --episodes 48 --eval 4 --max-ticks 300 --update-every 8 --device cuda --curriculum staged --schedule interleaved --save-rollouts --visible
# Examine frozen models on exactly the same held-out near cases:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/factorized_frozen_audit.py --run-tag NEWAUDIT01 --models NEWV201 --phases near --cases 12 --eval-seed 3737 --max-ticks 360 --device cuda --visible
# Prepare a new natural world and then evaluate frozen V2 + controls:
.venv/bin/python experiments/natural_world_benchmark.py --run-tag NEWNATURAL01 --world-seeds 1357913579 --modes forward_attack random joint_factorized --factorized-model NEWV201 --max-ticks 360 --prepare
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/natural_world_benchmark.py --run-tag NEWNATURAL01 --world-seeds 1357913579 --modes forward_attack random joint_factorized --factorized-model NEWV201 --max-ticks 360 --device cuda --visible
```

The external bridge has a hard ~90-second command limit. Keep a single test
within that limit or use an appropriate *transient, self-terminating*
runner with durable local checkpointing; **do not leave an indefinitely
running training daemon** or promise future data. The existing nested
Gamescope wrapper can still exit status **139** after the Python child has
completed and written valid results; one interleaved run wrapper also timed
out at status **124 after the full Python log/checkpoint had been saved**.
Verify JSONL/checkpoint completion and actual game outcomes, not only wrapper
exit codes.

## Recommended next research gate

1. Stop blindly scaling this PPO CNN. The paired scripted `8/12` vs trained
   `0/12` first-log gap is an explicit unresolved failure.
2. Use recorded agent visual experience to diagnose visual discrimination,
   action-response times and camera stability. Evaluate improvements against
   **matched, frozen validation scenarios**, including target-free controls.
3. Compare a memory-capable agent and more sample-efficient learned visual
   representation/replay-based approach, under the same compute/tick budgets.
   Choose algorithms only after measured throughput + pilot results.
4. Establish real, unrestricted first-log acquisition in diverse scenarios,
   then repeat on untouched Minecraft world seeds; only then proceed to the
   same uninterrupted wood -> crafting episode. Never substitute scripted
   tasks for genuine progress toward learned Any% speedrunning.
