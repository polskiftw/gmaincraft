# speedrun01 — 24k-decision real Minecraft training audit and next experiment

**Verified November 3, 2026.** This report describes actual independent Gentoo/MCio
Minecraft Java 1.21.3 runs; it does not claim general natural wood pickup.
All run logs and trained weights are local/Git-ignored and preserved.

## Original training: fully committed

`speedrun01` finished **24,000** actual six-tick decisions, **250** CUDA PPO
updates and **10** distinct real vanilla noise seeds, with full journal and
checkpoint agreement (`next_episode=10`). Event totals: **5,149** new
2x2 visitation cells; **75,294** first-hit surface sightings; **759** newly
observed wood surfaces; **106** actual mined nonwood blocks; **zero** mined
wood; **zero** acquired wood in inventory; three deaths. Per-world block
breaks: `41, 7, 34, 12, 6, 1, 2, 3, 0, 0`. This is a real loss of mining
frequency, not merely loss of access to wood: the policy still observes
wood surfaces but almost never attacks. The first and last PPO update
entropies were approximately 4.42 and 1.86 across the three action heads;
this is a warning of action collapse, not proof of any particular causal
mechanism. Prior startup driver SIGFPE was recovered before the successful
resume; the underlying NVIDIA root cause remains unproven.

World-level real game-I/O measured 1.557–2.396 ms/simulated tick. No
new prolonged MCio hang was observed in this completed run, though
startup and graphics reliability still warrant monitoring. The
render/boot cost cannot be inferred from this tick-only number.

## Frozen matched-seed baseline — independent live game runs

The new **non-policy** `NaturalActionAudit` reports move/look/attack action
counts, wood vs nonwood vanilla crosshair targets, consecutive decisions
holding attack with no camera look, and own visible mining state. It does
**not** alter the actor's input, rewards, choices, optimizer or existing
checkpoint schema; same exact audit appears on both train and eval.

Commands actually completed (new fresh worlds, same heldout seed and spawn):

```bash
tools/run-fresh-evaluation.sh speedrun01 srvdiag_pol_1103 480 policy_sampled 240
tools/run-fresh-evaluation.sh - srvdiag_rand_1103 480 uniform_random 240
```

Each controller played **240 decisions on each of two held-out seeds**:
`-6248071613379014257` at `(48.5,64,32.5)`, and
`-1471411430336780962` at `(0.5,127,0.5)`. Both policy eval and random
finished normally, with no optimizer updates.

| Live two-seed totals (480 decisions) | Frozen sampled model | Uniform random |
|---|---:|---:|
| Forward+jump moves | 452 (94.2%) | 87 (18.1%) |
| Attack decisions | 12 (2.5%) | 254 (52.9%) |
| Attack with wood under crosshair | 0 | 4 |
| Attack + still camera + wood crosshair | 0 | 1 |
| Longest attack+still-camera run | 1 decision = 6 ticks | 1 decision = 6 ticks |
| Actual mined ordinary blocks | 0 | 2 |
| Actual wood mined / inventory pickup | 0 / 0 | 0 / 0 |
| Newly sighted wood surfaces | 70 | 115 |
| New visitation cells | 149 | 68 |

**Important:** 12/480 is 2.5% (an earlier interim observation of 9
attacks was not the final committed per-episode count). The model's jump
and exploration behavior is qualitatively distinct from successful mining;
novelty rewards do not imply block acquisition. Two held-out seeds are
only a diagnostic, not a statistically powered generalization test. The
random policy's 2 blocks do not prove it is a better Minecraft learner.

## Separate versioned trial: wood progress + anti-collapse

The new optional `wood_progress_v3` reward **retains** v2 novelty, genuine
block mining and actual inventory rewards, but additionally credits **0.30
per newly attained own-client visible wood crack stage**, restricted to the
*same consecutively observed vanilla wood crosshair target*. Per-target
max-stage high-water means repeating an animation reset cannot repeatedly
produce credit; a missing/unavailable stage, loss of target, first target
frame or unrememberable target produces no credit. This uses no chunk map,
hidden destination, privileged target position or scripted action. The
actor still freely chooses normal move/look/attack every six ticks.
The v3 model trains with a lower learning rate **0.00015** (was 0.00035)
and a stronger policy entropy term **0.04** (was 0.014) to investigate the
exploration/attack collapse. Reward profile is stored in exact checkpoint
metadata; **existing `speedrun01` v2 checkpoints still resume with their
original hyperparameters**. Since changing a reward under the same run tag
would destroy experimental comparability, v3 needs its OWN new tag,
starting with random neural weights rather than importing collapsed v2.

Actual Minecraft smoke: `tools/run-fresh-treadmill.sh start crackpilot1103
96 96 wood_progress_v3` completed one genuine vanilla survival seed,
96 decisions, 576 ticks, one CUDA optimizer update, 3 ordinary mined
blocks, 12 newly seen wood surfaces, 0 mined/picked wood; worldgen6.651s,
startup10.529s, real game I/O **3.539ms/tick** including cold game ticks.
The journal+atomic checkpoint verifier passed. This is a functionality
smoke, not evidence that v3 learns a better skill; a longer training
plus matched held-out evaluation is required.

## Recommended next user training experiment

```bash
cd /workspace
tools/run-fresh-treadmill.sh start woodv3_01 24000 2400 wood_progress_v3
```

This deliberately creates a NEW learned policy rather than modifying
`speedrun01`. It retains the endlessly unique Minecraft seed stream,
scoped NVIDIA workaround, process-death watchdog, on-disk checkpoint,
world-retention safeguards, and train/frozen observation parity. The user
can extend the total later with
`tools/run-fresh-treadmill.sh resume woodv3_01 48000 2400 wood_progress_v3`.
After training, compare frozen v3 against original v2/random on held-out
seeds (profile must match model):

```bash
tools/run-fresh-evaluation.sh woodv3_01 v3eval01 4800 policy_sampled 1200 wood_progress_v3
tools/run-fresh-evaluation.sh - v3random01 4800 uniform_random 1200 wood_progress_v3
```

Until real wood inventory events appear, do not call any policy a successful
natural-world log finder, let alone an Any% speedrunner. Next substantive
sensor work remains genuinely perceivable dropped item entities.
