# V2: retain self-achieved wood progress without forgetting normal Minecraft interaction

**November 4, 2026; real Java 1.21.3 + MCio, RTX 4070 SUPER CUDA.**
The user still wants an agent that autonomously learns actual Minecraft Any%
from fresh natural survival worlds and eventually composes learned skills,
without a human-specified route, hidden-world/seed oracle, auto-aim or
scripted phase selection. This document measures **one low-level practice
skill**, not resource mastery or any End/dragon capability. Read the
[observation policy](OBSERVATION_POLICY.md), [Any% goal contract](BOOTSTRAP_TO_ANY_PERCENT.md),
[prior V9 audit](V9_TEMPORAL_TARGET_AND_REAL_GOAL_AUDIT_2026_11_04.md), and
[V1 replay-Q launch](REPLAY_Q_LIVE_LEARNING_2026_11_04.md).

## 1. The previous 200-world V1 training run is FINISHED and failed its skill gate

Tag `qselfplay_v1_1104`, real committed journal
`logs/fresh_stream_qselfplay_v1_1104.jsonl`, ignored atomic checkpoint
`checkpoints/fresh_stream_qselfplay_v1_1104.pt`:

- **200 independent new natural train worlds** (train indices2..201), 1,200
  six-tick units each = **1,440,000 Minecraft game ticks** in 7,221.72
  seconds wall-clock including separate Java client/world launches.
- **139,165 real neural ordinary-input actions**, 9,512 cumulative CUDA
  double-Q optimizer steps, 200 complete player-only `.npz` archives (about
  58MB), 2,260 ordinary broken blocks, six mined wood blocks, but ONLY
  **three actually acquired log inventory events**. All wood pickups came
  from train worlds in the FIRST 20; later 180 worlds acquired **zero**.
- Mean MCio synchronous I/O ~2.0–2.3ms per game tick across the run;
  no measured recurrence of the original MCio performance regression.
- Late worlds still saw actual wood; e.g. train indices180–199 had 467
  neural choices with wood under the crosshair and 102 attack choices at
  that target, but only two sampled crack rises and zero log pickups.
  This rules out a simplistic explanation of 'Minecraft stopped showing
  any wood'; it does NOT prove any specific root cause.

**Actual archived V9 player-state analysis:** Of 139,165 choices, only
**52 genuinely observed wood-targeted crack-progress outcomes** occurred,
spread across ten episodes. 3,017 other choices began with wood under
crosshair but did not produce this exact progress label; 136,096 began
without wood under crosshair. The OLD v1 learner rewarded any
observable crack rise, not just when the current same visible target was
wood, and its rolling last-32-world window eventually lost all the
original successful wood examples. Both label mismatch and forgetting
are experimentally plausible contributors, **not causal proof**.

### Frozen same-real-state Q policy behavior (NOT gameplay evaluation)

`experiments/audit_replay_q_collapse.py` evaluated different frozen
models on exactly 3,868 SAME original saved player observation states
from source train indices2,5,15,36,100,199, including 313 states with
real wood under the crosshair. Neither model received new optimizer
updates. The per-row *stationary attack* probabilities on these
wood-targeted observations were:

| Frozen model | Stationary attack on wood | Total attack on wood | Stationary attack outside wood |
|---|---:|---:|---:|
| Original Q `qcrack_v1a1104` (1,000 updates) | 11.30% | 42.77% | 9.13% |
| Completed 200-world Q `qselfplay_final1104` (9,512 updates) | **0.04%** | 16.75% | 5.84% |
| New target-specific V2 `qwood_v2anchor2_1104` | 45.50% | 93.06% | 13.57% |
| Online V2 after two further natural worlds, `qwood_v2smoke2f1104` | 52.54% | 94.22% | 14.35% |

The V1 model indeed became **almost completely unwilling to stand still
and attack wood**, even on identical remembered observations. This is
actual measured policy drift, not just a hypothetical PPO reward issue.
V2 responds strongly but also attacks more frequently outside wood,
so controlled state-response probabilities are only a diagnostic and
cannot prove good navigation or stable multi-seed play.

## 2. Implemented V2 learner (separately versioned, NO original overwrites)

New code:

- `src/gmaincraft/replay_q_rare_v2.py`: new checkpoint schema
  `player_known_rare_outcome_conservative_q_v2`, training practice goal
  `effect:visible_wood_crack_progress_v2`. Positive labels REQUIRE an
  actual **same visible crosshair target**, **current own wood type**, and
  **positive known crack-stage increase** in the existing already-legal
  V9 observations. No new gameplay sensors or privileged world scans.
  The goal describes a verifiable practice outcome, not a hand-coded
  command to aim or attack; the network chooses normal 20-way vanilla
  actions freely. It need not even know a future target location.
- Rare-success replay uses genuine actions from the complete committed
  200-world experience, balanced against **real** wood-target failures
  and general actions; no fabricated expert actions or positive outcomes.
  It includes successful-action likelihood, double-Q game-tick targets,
  conservative action weighting, and regularization to the agent's **own
  older self-trained Q** on observations without wood under the crosshair.
  No external pretrained model or expert demonstration.
- `experiments/train_replay_q_rare_v2.py`: crash-safe separate scratch-
  optimizer fine-tune from own previous network. The selected model
  `qwood_v2anchor2_1104` was trained for 20 epochs x 25 = **500 CUDA
  updates**, with nonwood KL anchor strength2 and success action weight0.6,
  from the 52 real wood progress samples and matching failure/other
  action states. An earlier exploratory strength0.2 model caused much more
  off-target attacking; separate tags preserve both.
- `src/gmaincraft/online_q_rare_v2.py` provides on-game-exit online Q
  replay with permanently protected actual success episodes from prior
  committed training generations plus the newer completed normal game
  episodes. Prior archived training streams are **not** held-out eval
  data; all source seed populations are train, exact same seed root,
  strict observation/action schema and committed-cursor checked.
- Training can be resumed only with the same exact versioned controller,
  reward goal, source seed root, optimizer and run options; old V1
  controller/checkpoint metadata remain unchanged. No accidental reuse of
  previous training world's index: V1 indices2–201, provisional V2
  smoke index202, valid two-world test203–204, chained smoke205, and
  next new long run begins at **206**.

## 3. Actual frozen V2 in Minecraft; score real log pickups, not visible wood

Same 3,600 Minecraft game ticks per case, real fresh survival and normal
input controls; none of these `eval`-namespace worlds was in training.
All controllers frozen and reported zero extra optimizer updates.

| Eval seed index | Model / action RNG | Real log pickup events | Notes |
|---|---|---:|---|
| 0 | Original V1 Q / 731117 | 2 | Previous historical comparison |
| 0 | Finished 200-world V1 Q / 731117 | **0** | Regression |
| 0 | New V2 (anchor strength2) / 731117 | **2** | Recovered on this seed |
| 6 | Original V1 Q / 731117 | **0** | Same train/eval split and spawn X/Z |
| 6 | New V2 / 731117 | **4** | 48 sampled visible wood crack rises |
| 6 | New V2 / 813777 | **3** | Independent stochastic action RNG |
| 7 | Original V1 Q / 731117 | **0** | 11 wood-targeted attack decisions |
| 7 | New V2 / 731117 | **1** | 21 wood-targeted attack decisions |
| 8 | New V2 / 731117 | 0 | No wood under crosshair in 600 units |
| 9 | V2 after two online train worlds / 731117 | 0 | No wood surfaces encountered |

Real archived evaluation journals are `logs/fresh_eval_<EVAL_TAG>.jsonl`:
`qfinal_hold0_1104`, `qwood_v2hold0_1104`, `qorig_hold6_1104`,
`qwood_v2hold6_1104`, `qwood_v2hold6rng2_1104`, `qorig_hold7_1104`,
`qwood_v2hold7_1104`, `qwood_v2hold8_1104`, `qv2grown_hold9_1104`.
A "log pickup event" is an actual positive OWN Minecraft inventory
change during the run, not necessarily logs still retained on death.
Some V2 worlds include deaths; death/reset and survival must be evaluated
separately before calling a skill reliable. Real Minecraft seed7 has
**different initially observed Y** despite the same seed and spawn X/Z
(original Q first Y72; V2 first Y69), so treat this case as same-seed
but not perfectly identical initial player state. Small sample sizes,
selected held-out worlds, action RNG variance and possible terrain
unavailability prevent a statistically strong transfer claim.

**Bottom line:** the new V2 policy acquired real wood on multiple unseen
worlds and different action RNGs where frozen original V1 did not, while
other worlds still failed to offer a visible wood target. Promising, but
NOT reliable natural wood collection on a broad unbiased holdout sample.

## 4. Real new V2 train/resume, plus durable multi-generation replay

The FIRST V2 collector smoke `qwood_v2smoke1104` (train index202)
identified an **outer-checkpoint schema mislabel** (v1 wrapper around v2
weights). Its actual game and optimizer outcomes were real but this
historical smoke is NOT a valid source for a new v2 frozen checkpoint.
It was **preserved unchanged** instead of reinterpreting or overwriting.
The wrapper now inherits the network's version and the collector kind
is separately versioned for V2.

**Correct real start+resume `qwood_v2smoke2_1104`:** independent train
seeds203 and204, 600 six-tick units each / 3,600 game ticks each,
**481 real neural actions**. First world mined and acquired three oak
logs; second mined/acquired one birch log (four genuine pickups total),
plus incidental wood-derived sapling and sticks. Real self-play archive
reopened and exact model+optimizer/RNG checkpoint cursor205 verified.
The first world yielded 21 actual achieved crosshair-wood crack actions;
combined two-world history yielded 23 actual achieved motor decisions
for new balanced training. **Eight plus eight** new CUDA Q updates:
500->508->516. Synchronous Minecraft I/O 2.65 and 2.22ms/game tick.

`tools/export-replay-q-snapshot.py` exported V2 frozen model
`qwood_v2smoke2f1104`; no live collector weights were changed.

**Next real chained smoke** `qv2chain_smoke1104`, TRAIN seed205,
600 six-tick units, mined two logs and **actually picked up one**, nine
new own wood-crack motor progress events; added eight more Q updates
516->524. It independently verified **75 permanent actual past success
examples** (52 original Q collector + 23 new V2 two-world source),
plus its own nine. Its valid versioned checkpoint cursor206 and real
replay are preserved.

**V2 multi-generation lineage:** A new frozen checkpoint
`qv2triplef1104` from the latest finished smoke carries the strictly
verified TRAIN source chain:
`qselfplay_v1_1104` (52 successes),
`qwood_v2smoke2_1104` (23), and
`qv2chain_smoke1104` (9): **84 total prior real action successes**.
The next collector can retain these while adding fresh real Minecraft
self-play experiences and online GPU updates. A source/checkpoint cannot
claim uncommitted episodes, include a heldout evaluation journal, or
silently create new observations/actions. No model chooses a hidden target,
teleports or executes a privileged mine command.

## 5. Workstation commands and next gates

Status after this audit: V1 long run completed, V2 start+resume and chained
smokes completed, frozen models exported and actual real-game evaluation
performed. See [NEXT_SESSION](NEXT_SESSION.md) for the exact latest new
long run tag, active lock/commit, and safe resume instructions.

```bash
cd "$HOME/Documents/minecraft-discovery"
# Inspect completed historical training and 200 owned archives:
tools/run-replay-q-collection.sh status qselfplay_v1_1104
.venv/bin/python tools/inspect-experience.py qselfplay_v1_1104
# Inspect separate real V2 smokes:
.venv/bin/python tools/inspect-experience.py qwood_v2smoke2_1104
.venv/bin/python tools/inspect-experience.py qv2chain_smoke1104
# Offline SAME-actual-observation Q behavior audit (NOT heldout gameplay):
.venv/bin/python experiments/audit_replay_q_collapse.py --models qcrack_v1a1104,qselfplay_final1104,qwood_v2anchor2_1104,qwood_v2smoke2f1104 --episodes 2,5,15,36,100,199 --device cuda
# Independent NEVER-TRAINED frozen real-game evaluation; change new eval tag:
tools/run-replay-q-evaluation.sh qv2triplef1104 brandnewqeval effect:visible_wood_crack_progress_v2 600 600 10 813777 q_sampled
```

**Before claiming progress toward Any%:** evaluate many unbiased disjoint
holdout seeds, independent stochastic action RNGs and model training seeds,
actual owned inventory per life and post-death retention, interrupted mining,
visible dropped-item perception, complete vanilla use/place/hotbar/crafting,
and autonomous learned goal initiation/switch/termination. A simple
`if wood then attack` rule or hardcoded wood->stone->End sequence remains
prohibited; V2's only visible-target rule is a transparent **training
outcome label**, while its live player still chooses neural vanilla actions.
