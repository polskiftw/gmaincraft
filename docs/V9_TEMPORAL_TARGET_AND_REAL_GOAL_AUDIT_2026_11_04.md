# V9 real-game temporal target history and self-goal pipeline audit

**November 3, 2026 (Indiana local; archived UTC logs are Nov 4). Verified on real Minecraft Java 1.21.3 + MCio, and
RTX 4070 SUPER CUDA. This is a software/data and empirical audit, NOT a
claim of reliable first-log learning or an Any% agent.** All V2-V8
checkpoints, historical worlds, and experiment data remain untouched.

Read [BOOTSTRAP_TO_ANY_PERCENT.md](BOOTSTRAP_TO_ANY_PERCENT.md) for the
long-term learned-skill composition goal and
[DURABLE_EXPERIENCE_AND_SELF_GOALS.md](DURABLE_EXPERIENCE_AND_SELF_GOALS.md)
for the existing replay/goal-learning tools.

## The real problem addressed

V5-V8 often selected new actions without remembering that the previous
vanilla crosshair target was the same block or whether recent continuous
attacking caused visible progress. Before attempting many crafting/stone
networks, test whether an actor can learn to sustain a single observed
resource interaction from its own recent player-visible history.

**V9 observation schema:**
`natural_memory_online_v5_visible_target_temporal_history_v9`.
Its underlying **20-way V7 motor choices and V8 reward profile are
UNCHANGED**. A new scratch network has ten additional bounded numeric
features, calculated only from the same successive player-status and
mining-feedback readings, plus the actually issued ordinary player input:

1. Same vanilla crosshair-target block ID/position as last observation
2. Capped elapsed ticks during same sampled crosshair target
3. Previous sample's crosshair was wood
4. Last ordinary action pressed attack
5. Last ordinary action pressed stationary attack
6. Capped stationary attack time across samples of the same visible target
7. Current own break crack stage is actually known
8. Increase in current own crack stage for the same visible block
9. Highest crack stage actually seen for the current target
10. Capped duration without an observed crack-stage increase

The previous target's internal identity comes **only** from a genuine
vanilla player crosshair hit, not a scanned chunk, hidden block location,
server command, item pickup, structure map, or a privileged oracle.
Features are capped 0..1, and no target position is additionally injected
into the agent. A sampled same-target match does not prove the target
was continuously held during all six intervening ticks. Unknown crack
stage -1 remains unknown, not zero. The policy can choose any of the
same motor actions and may completely ignore the history; there is no
rule such as `if crosshair is wood: attack` or training-script handoff.

Each *new* natural world gets a freshly initialized history object, while
network/optimizer/replay continue across worlds. History was added via
an opt-in argument to the shared real-game `read_step()` path. Old V8
`observe()` and saved-model schemas do not silently change. V9 has strict
metadata including exact history feature names and refuses V8 checkpoint
resumes. The direct observed-player archive now recognizes the V9 schema.

## Actual same-world V8/V9 scratch smoke (not enough for superiority)

Two genuine fresh vanilla noise seeds, both **120 six-tick units**
(720 Minecraft ticks) per world. Each starts from a fresh random network
seed, identical motor action vocabulary, policy RNG seed and matching
Vanilla start per world; their randomly initialized network dimensions
necessarily differ. Both completed real CUDA training and checkpointed.

| Metric | V8 scratch | V9 scratch |
|---|---:|---:|
| Real training worlds | 2 | 2 |
| Six-tick game-time units | 240 | 240 |
| Neural motor choices | 83 | 84 |
| CUDA PPO updates | 7 | 6 |
| Wood items in inventory | 0 | 0 |
| Stationary attacks while crosshair on wood | 0 | 1 |
| Observable wood crack reward | 0 | 8.1 |

V9 saw 13 repeated-target samples and nine positive crack-stage changes
in its first seed, none in the second. This is a live integration proof
and a **chance outcome** in a tiny matched training sample, not evidence
V9 learned a transferable target-aware policy. Source files:

- V9: `logs/fresh_stream_temporal9pilot_1104.jsonl`
- V8: `logs/fresh_stream_matched8temporal1104.jsonl`
- Ignored snapshots and per-world replay are preserved under their own
  `checkpoints/` and `data-local/experience/` run tags.

## Longer actual V9 train/resume: valid data, still no wood mastery

**Run** `v9longpilot1104`, 600 six-tick units/world, PPO update every 32
sampled neural actions, real game + RTX CUDA, player-only replay enabled.
An independent first world was cleanly committed, then the SAME run was
resumed into an entirely different real noise seed. Both complete:

| Committed episode | Real motor choices | CUDA PPO updates | Other blocks mined | Wood mined / inventory | Other observed inventory gain |
|---|---:|---:|---:|---:|---|
| 0 (seed 1671001014072077565) | 219 | 7 | 24 | 0 / 0 | 1 gravel |
| 1 (seed -125943801469537593) | 230 | 8 | 0 | 0 / 0 | None |
| **Total** | **449** | **15** | **24** | **0 / 0** | **1 gravel** |

Total **1,200 six-tick units = 7,200 real Minecraft game ticks**.
The player's actual inventory (NOT a screenshot or mined-count guess)
reported one gravel gain. Both per-episode player-only archives were
committed, reopened, and verified (278,779 compressed bytes total).
Time per real Minecraft tick in these worlds: 2.468 and 2.641 ms of
synchronized MCio game I/O, not counting generator/client startup.

The history auditor counted 41 and 80 consecutive same-target sampled
observations, and one and nine crack-rise samples. Wood crosshair occurred
at seven neural decision boundaries in first episode and zero in second.
The combined snapshot after 15 optimizer updates **does not yet show a
learned conditional wood-mining skill**.

Source: `logs/fresh_stream_v9longpilot1104.jsonl`, local ignored model
`checkpoints/fresh_stream_v9longpilot1104.pt`, and read-only archive
inspection via `tools/inspect-experience.py v9longpilot1104`.

### Frozen offline history sensitivity on REAL saved player states

New `experiments/visible_target_policy_probe.py` loads only genuinely
committed V9 player observation archives and a frozen network. It
measures actual crosshair-wood prevalence and **artificially zeros only
the appended ten history features** to quantify policy response. This is
an OFFLINE ABLATION, not a plausible no-history gameplay rollout or a
performance claim. After the full 15-update run:

- 449 real saved neural decision states across 2 worlds
- 7 of those actual states had wood under the crosshair
- 55 had the same currently targeted block as the previous sample
- Mean stationary-attack probability 0.1250; on actual wood 0.1272
- Mean action distribution total variation when history zeroed: **0.00312**
  (~0.31%); mean absolute stationary-attack probability change 0.000477

The model is still almost insensitive to history, even though the sensor
and the archive are verified. Do **not** justify another months-long PPO
run solely on this sensor being present; the next research target is
sample efficiency and real learned credit for target-aware mining.

## The first genuine Minecraft -> replay -> goal-learner -> Minecraft test

The one real gravel inventory gain generated 12 hindsight-labeled
preceding ordinary motor choices (bounded lookback 12). A NEW scratch
small goal-conditioned policy `v9gravelgoal1104` was trained for 20 CUDA
epochs; in-archive action-imitation accuracy reached only **33.3%**,
with **no held-out training episodes containing this item**. This is NOT
reliable skill learning and cannot be considered a gravel-acquisition
strategy. No synthetic test examples entered this real dataset.

Its *frozen* policy received the explicit practice goal `minecraft:gravel`
on independent vanilla **held-out** Minecraft seeds. One initial test
`v9gravelhold1104` accidentally picked up **two real wood logs** in 1,800
Minecraft ticks, even though it did not gather the requested gravel.
This was an interesting pipeline event, NOT trained wood proficiency.

**Important evaluator RNG correction discovered during audit:** before
this commit, the goal-model loader initialized its frozen motor sampler
with the *training* seed instead of the requested *evaluation action RNG*.
This made supposed alternate-seed tests repeat action randomness. It was
fixed without changing the actual goal network/checkpoint. The original
two-wood evaluation is preserved as a **pre-fix fixed-training-RNG run**,
not a valid `--policy-seed 731117` result.

Re-evaluating the SAME heldout seed index 0 with genuinely independent
frozen action sampling and two matched controls:

| Controller / action RNG | Actual wood mined | Real wood inventory | Gravel goal achieved |
|---|---:|---:|---|
| Goal network, pre-fix fixed training RNG | 3 | 2 | No |
| Goal network, corrected RNG 731117 | 0 | 0 | No |
| Goal network, corrected RNG 813777 | 1 | 1 | No |
| Original frozen V9 flat motor, RNG 731117 | 1 | 0 | N/A |
| Uniform-random V9 motor, RNG 731117 | 0 | 0 | N/A |

At *different heldout seed index 1*, frozen goal network RNG 731117
also achieved **0 gravel and 0 wood**. Every frozen audit completed with
**zero additional optimizer updates**; these were real natural survival
worlds with normal movement/looking/attacking and no evaluator phase
switch. They do not establish policy skill, and the single gravel example
was insufficient. Score source: each immutable ignored
`logs/fresh_eval_<tag>.jsonl` and the frozen goal checkpoint under
`checkpoints/hindsight_v9gravelgoal1104.pt`.

## Correct live commands for future independent experiments

Start a **new** durable training tag (not either existing pilot):

```bash
cd "$HOME/Documents/minecraft-discovery"
GMAINCRAFT_RECORD_EXPERIENCE=1 tools/run-fresh-treadmill.sh start temporal_v9_new01 forever 600 wood_temporal_v9
```

Resume with all the SAME settings including the archive opt-in:

```bash
GMAINCRAFT_RECORD_EXPERIENCE=1 tools/run-fresh-treadmill.sh resume temporal_v9_new01 forever 600 wood_temporal_v9
tools/run-fresh-treadmill.sh status temporal_v9_new01
tools/run-fresh-treadmill.sh stop temporal_v9_new01
.venv/bin/python tools/inspect-experience.py temporal_v9_new01
.venv/bin/python experiments/visible_target_policy_probe.py temporal_v9_new01 --device cuda
```

The training job is **foreground**, not a background daemon. New worlds
are never deliberately repeated. A 600-unit horizon is a new-run choice,
not a fixed training requirement. The user is happy to provide long GPU
runtime, but no one should imply V9 is now validated simply because it
supports indefinite training.

Frozen new V9 versus same-seed random control (replace all eval tags):

```bash
tools/run-fresh-evaluation.sh temporal_v9_new01 v9_frozen_new01 1200 policy_sampled 600 wood_temporal_v9
tools/run-fresh-evaluation.sh - v9_random_new01 1200 uniform_random 600 wood_temporal_v9
```

Train a scratch **goal-conditioned imitation baseline** only after the
archive contains actual item gains; it is a local skill diagnostic, NOT
a high-level task manager:

```bash
.venv/bin/python experiments/train_hindsight_goals.py --source-run temporal_v9_new01 --run-tag v9_goal_new01 --epochs 20 --device cuda
tools/run-hindsight-evaluation.sh v9_goal_new01 v9_goal_hold_new01 minecraft:gravel 1200 600
```

Gravel is merely an illustrative practice item; pick a goal observed
in that run. After the frozen RNG fix, supplying distinct evaluation
RNG seeds actually produces independently sampled goal-policy actions.

## Stability and next verified gates

- No V2-V8 actor, reward, environment, checkpoint, or historical raw data
  was rewritten. V9 is new strict schema + same 20 vanilla motor and
  underlying reward coefficients as V8. A stored source seed is never a
  neural policy feature, history feature, or goal-learner replay label.
- The observed crack uses 6-tick on-demand actual client readings, not
  a block-mining command. Unknown stages and changed targets are explicitly
  handled. Real player-only item gain remains the success source of truth.
- The NVIDIA/Gamescope outer wrapper still exits with known **139** after
  some successful real Java runs. Inner Minecraft completion, journal,
  checkpoint and replay file were checked independently. Do not assume
  this means the PPO checkpoint failed, nor assume driver root cause fixed.
- Tests: V9 player-owned continuity, unknown stages, no hidden target
  input, replay shape, strict checkpoint incompatibility, V9-vs-V8
  underlying reward contract, sensitivity probe, and corrected frozen
  goal-action sampling. Full pure suite must pass before push.
- **Next work:** find a *realistic train-only* way to increase encounters
  with naturally occurring visible wood WITHOUT preaiming, placing trees
  or handing the policy target coordinates; investigate replay/off-policy
  goal credit and a richer item/drop sensor. Continue comparing unbiased
  heldout natural seeds, multiple policy RNGs, true inventory and actual
  in-game time. Only then revisit hierarchical skill composition.
