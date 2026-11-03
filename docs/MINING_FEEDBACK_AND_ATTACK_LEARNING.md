# Read-only Minecraft mining progress and a learned attack motor skill

## NEWER: real inventory acquisition and on-demand follow-up

The attack motor is now independently combined with the older trained
pickup controller on **genuine vanilla dropped items from real mined logs**.
On 12 held-out cases a frozen learned+learned pair picked up **11/12**
actual inventory logs, with further held-out runs9/12 and7/12 for
alternate pickup models. This requires an evaluator's SCRIPTED phase
handoff after a real mined stat and a controller-staged pre-aimed log,
so is NOT an autonomous first-log finder or end-to-end learned policy.
See [`COMPOSED_MINE_PICKUP_AUDIT.md`](COMPOSED_MINE_PICKUP_AUDIT.md)
for all controls, the cumulative-stats bug/repair, independent seeds,
checkpoint hashes, per-tick performance, and the new synchronous opt-in
on-demand mining feedback protocol. The older findings below remain
accurate historical evidence; any claim of **0 pickups** refers to the
initial attack-only lab, NOT this subsequent composed audit. The current
on-demand mod JAR has a newer hash than the historical initial v1 build.

**Real Minecraft Java 1.21.3, Fabric/MCio, November 2-3 2026.** This is a
narrow but **genuine from-scratch learned action-control component**. It is
not a full V6 speedrunner, a general tree finder, a learned pickup skill, or
an Any% completion. `docs/OBSERVATION_POLICY.md` remains authoritative.

## Why this matters

The historical V5 actor could see a real vanilla oak crosshair target for
30/30 decisions but chose **no attack ticks**. Its 4-episode training pilot
and held-out cases never acquired a log. A fundamental issue was that it
had no fine-grained *own block-breaking progress* in its state or an obvious
credit signal for holding attack across multiple 6-tick decisions.

We inspected the actual cached Minecraft 1.21.3 client class file (not an
assumed mapping). `net.minecraft.client.multiplayer.MultiPlayerGameMode`
exposes public `isDestroying()` and `getDestroyStage()`. These describe
this player's **visible vanilla crack animation**, not hidden world data.
The opt-in Fabric extension adds:

- `GMAINCRAFT_MINING_FEEDBACK_V1=1` to the isolated MCio game process.
- A typed `MiningFeedbackOption`, schema `gmaincraft_mining_feedback_v1`,
  with `actively_destroying`, `crack_stage` (-1 if inactive; otherwise
  the current vanilla stage) and `crosshair_on_block` (vanilla client's
  own real crosshair target test). **No target XYZ**, hidden block
  enumeration, world seed, privileged mining reward or controller commands.
- A fail-closed `MiningFeedbackV1.from_packet()` Python adapter, checked
  immutable shape/ranges and explicit unknown state on release. Register
  `gmaincraft.mcio_mining_feedback` before first packet. Old V5/other
  clients still work when this env flag is unset; the old sensor schema
  and its trained weights are unchanged.

## First live real Minecraft sensor experiment

`experiments/mcio_mining_feedback_probe.py`, fresh copied lab world
`MDL_MiningFeedback_miningfeedback_nov03a`, report
`logs/mcio_mining_feedback_miningfeedback_nov03a.json` (ignored locally):

1. Player stood still and targeted an actual oak log; idle feedback was
   inactive with stage **-1** and the oak crosshair target valid.
2. Four consecutive ordinary six-tick `LEFT_BUTTON` holds produced
   actual stages **1, 2, 3, 4**, without any mined counter increment.
3. Releasing attack for six ticks immediately reported inactive stage **-1**.
4. A subsequent uninterrupted vanilla hold advanced through stages
   **1, 2, 3, 4, 5, 6, 7, 7, 8, 9**, then on the next six-tick decision
   the actual `minecraft:mined` oak count increased **by one**. The crack
   stage disappeared and the real crosshair hit the bedrock behind it.
5. Turning with the regular mouse never reported a hidden oak target or
   ghost breaking progress. **9/9** independent sensor audit checks passed.

This real block was mined but never added to inventory during the probe
(after an ordinary short forward attempt). **Do not call it collected.**
No admin commands occurred during measured play; `/setblock` and `/tp`
were used ONLY by the experiment controller before gameplay to prepare
one disposable repeatable scene. No policy was trained in this first test.

Current installed isolated MCio mod JAR SHA256:
`7c41137b8789b0661700a322a90e4e7891752711b27786ac3fb5f750529d82c2`.
The previous visible-surface calibration JAR SHA256
`8cc58f899e7e824d0d67de7b74517b5390dbc3175375ec8c6b7560a2406135b7`
is preserved under ignored
`mcio-data/mod_backups/mcio-1.6.1+1.21.3.visible-calibration.jar`.
All older mod backups remain intact; the original pristine JAR can still
be restored through `tools/build-mcio-player-status.sh --restore`.

## New from-scratch learning component

`src/gmaincraft/mining_attack_policy.py` is deliberately narrow:
`mining_attack_motor_v1_own_crack_stage_interval6` controls **only** whether
vanilla `LEFT_BUTTON` stays held or is released for each six game ticks.
The policy receives precisely these six locally knowable values:

1. whether the vanilla crosshair currently hits any block;
2. the actual current crosshair hit distance (normalized);
3. whether its own player is actively breaking a block;
4. the visible crack stage (normalized, inactive separately marked);
5. whether **its own previous action** held attack;
6. elapsed own game ticks (bounded).

It sees **no** tree position, fixture coordinates, privileged teacher
bearing, target identity oracle, world seed, server commands, or mined
progress counters as input. A tiny 32x32 Tanh actor-critic starts with
random initialization and trains on actual normal Minecraft motor actions
on the NVIDIA RTX 4070 SUPER CUDA device. A separate independent evaluator
uses the identical schema/feature encoding and a **frozen checkpoint**.

An intentionally simple reward for the controlled curriculum combines:
observed positive change in vanilla crack stage, negative reward for
abandoning a partially broken block, a small normal per-decision cost and
a large **real** `minecraft:mined` event when the block genuinely breaks.
The event is allowed as a training/scoring outcome, not as an observation
or supernatural goal location. This is an actor-critic trained via real
repeated agent decisions, **not supervised demonstrations or an external
pretrained model**. The train/frozen runtimes have the exact same six-feature
input and opt-in game flags. `AgentAction` has no admin-command field.

### Verified first run: `motorlearn_nov03a`

`experiments/mcio_attack_motor_learner.py --phase train`, 28 separate
controlled oak scenarios, source fixture copied into a new disposable
Minecraft world, training controller-only resets between episodes.
Independent scenario seeds: train **381231**, frozen eval **673231**;
initial random neural seed **410233**. Six ticks per decision, at most
18 decisions (108 game ticks) per episode. Source logs and game worlds are
ignored, never overwritten or pushed to GitHub:

- Genuine `minecraft:mined` oak events in **13/28** training episodes.
- First half **3/14** and second half **10/14** reached a true break.
- Final checkpoint `checkpoints/mcio_attack_motor_motorlearn_nov03a.pt`
  SHA256 `4fe85be220306424f06e93ef7f8b0e76c6fbf4bd662da511ce028201be9bb6b8`.
- Every training episode made standard 6-tick real Minecraft press/release
  decisions; gameplay has no controller commands after lab staging.
- Each policy decision observed its own stage, real crosshair and action
  history. The checkpoint contains the schema and feature order; incompatible
  copies must be rejected, and old V5 weights are not migrated.

In a separately copied **fresh** Minecraft eval world on 12 held-out
stationary near-oak scenarios (`motoreval_nov03a`):

| Frozen policy/control | Real oak blocks mined out of 12 |
|---|---:|
| Learned neural attack controller | **12/12** |
| Separate untrained random-initialized neural controller | 0/12 |
| Randomly alternate hold/release | 0/12 |
| Never hold attack | 0/12 |
| Always hold attack | **12/12** |

Each independent condition used the same held-out cases and a new
reset/cleared local inventory; only the learner and control action source
varied. The learned policy consistently picked the **11 consecutive
six-tick attack actions** required to break a log; the untrained network
averaged only 8.67 attack decisions scattered over the 18-step horizon.
No control used an artificial mined-log reward during frozen gameplay.

**Interpretation:** the neural network **learned** sustained vanilla attack
control and matched a competent, extremely simple always-hold baseline.
It **did not beat always-hold**, and these are pre-aimed, reachable,
controller-staged oak logs on ONE flat-world layout with small jittered
camera pose. The network did NOT learn where to find wood, how to aim,
which material to break, when to switch tool, how to pick the item up or
how to speedrun. Inventory pickup after the learned attack was **0/12**;
that is the next missing skill, not a success. 12 held-out cases do not
give a generalization estimate across natural Minecraft seeds. The
increased training success is encouraging, not proof of universal learning.

## Reproduction and next decision

```bash
cd /workspace
.venv/bin/python -m unittest discover -s tests -q
tools/build-mcio-player-status.sh --build
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_mining_feedback_probe.py --run-tag FRESHMINING --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_attack_motor_learner.py --phase train --run-tag FRESHTRAIN --episodes 28 --max-decisions 18 --device cuda --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_attack_motor_learner.py --phase eval --run-tag FRESHEVAL --model-tag FRESHTRAIN --eval-cases 12 --max-decisions 18 --device cuda --visible
```

The nested Gamescope wrapper frequently exits **139 only after** the
Python child saved complete results; check `SAVED` and durable JSON/JSONL
before counting any pass. Missing report, child traceback or failed test
is a genuine failure. The shared disposable Java launcher now activates
mining feedback ONLY when explicitly requested by the research run; all
previous V5/V4 code paths retain their old observations.

**NEXT:** combine the learned attack controller with separately validated
pickup behavior on actual mined drops (not pre-spawned item entities),
while recording comparator always-hold attack + always-forward movement.
Longer-term add calibrated read-only visible **item entity** state from
the Minecraft client, then learn joint discover/approach/aim/mine/collect
using the same bounded memory inputs in training and frozen survival.
Do not substitute pixel-only learning as a project requirement or claim
full first-log success until a policy physically acquires inventory wood.

## Performance watchdog and legacy crosshair compatibility

The installed mining-feedback Fabric JAR passed the **old** status-only
`experiments/mcio_target_block_probe.py` smoke with the new mining env OFF:
`miningcompat_nov03a`: oak positive, mouse-away negative, restored aim
positive and stone occlusion negative, **4/4**. The previous V5 trained
checkpoint still has no new input or altered schema.

No-Torch, NVIDIA 320x180 real RGB, identical fixed fixture, idle 300 game
ticks, separate fresh Minecraft launches, two repeats per launch:

| Same JAR / test tag | First 300 ticks | Second warmed 300 ticks | Sequence gaps |
|---|---:|---:|---:|
| Status V1 ON, mining feedback OFF, `mineperf_off_nov03a` | 1.3825s | **0.7440s** | 0 |
| Status V1 ON, mining feedback ON every tick, `mineperf_on_nov03a` | 1.3424s | **0.8109s** | 0 |

The warm mining-on measurement was **0.0669s (~9%) slower** than off in
this one pair, even though the record is tiny. It emitted 300 fresh
readings, and the no-input scene correctly reported zero active mining
packets. This is NOT a robust causal estimate or a confidence interval;
Java/JVM warmup and host load can differ by launch. But it is enough to
flag performance for follow-up: request the current crack observation only
at the actual 6-tick model decision instead of sending it on *every* game
tick, then retest same-world train/frozen parity and I/O. Never silently
substitute a stale crack stage for a current observation.
