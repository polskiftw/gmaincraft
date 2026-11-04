> **Historical V1 foundation only — run now completed and behavior regressed.**
> The follow-up 200-world experiment `qselfplay_v1_1104` finished and
> collected only 3 real logs. The corrective V2 training, outcomes,
> genuinely verified goal and new safe continuation are documented in
> [V2 rare-success audit](V2_RARE_SUCCESS_REPLAY_AUDIT_2026_11_04.md).
> Do not treat the historical launch instructions below as an active job.

# Replay Q: player-only off-policy learning, real Minecraft audit

**November 3, 2026 Indiana local / November 4 UTC journals.** This is a NEW
experimental research baseline; the end goal remains an autonomous, general
Minecraft Java Any% policy from scratch, with learned skill composition rather
than hardcoded route/recipe/target. No claim of reliable wood collection,
learned planner, item-use/crafting, or completed speedrun follows from these
small tests. [Observe/cheating boundary](OBSERVATION_POLICY.md) and
[Any% bootstrap architecture](BOOTSTRAP_TO_ANY_PERCENT.md) remain authoritative.

## What was implemented and actually measured

**`player_known_conservative_goal_q_v1`** is an independent lightweight
PyTorch double-Q network with a conservative logged-action penalty and
semi-Markov discounting by real Minecraft game ticks. Like the V9 scratch PPO,
it receives ONLY the existing bounded own-player/crosshair/history/camera
first-hit block observations, with the same real-game 20-way ordinary-input
motor. No world seed, hidden-world object lists, trainer true mining stats,
preaimed teacher targets, scripted item pickup or task handoff enter the
policy. Its explicit *practice goal* is either an item ID previously observed
in own inventory (scored solely by **actual inventory increase**) or a
dedicated `effect:visible_crack_progress` goal, which is achieved ONLY when
the own V9 typed history has a real known same-crosshair crack rise. The
latter means observed block damage, **not** wood mined or even wood targeted;
all new scored successes are accounted independently.

`experiments/train_replay_q.py` reads ONLY already checkpoint-committed
original V9 real player-only `.npz` episodes (never heldout evaluation data),
and prioritizes scarce positive outcomes among the recorded ordinary actions.
One real historical archived run `v9longpilot1104` contained 449 ordinary
motor decisions and only **four** achieved crack-rise action transitions,
plus **one** gravel inventory pickup. From random scratch weights, the new
`qcrack_v1a1104` trained 1,000 CUDA Q updates / 40 offline epochs on these
limited self-play examples; separate safe tags preserve all V2-V9 models.
Offline loss is not a gameplay result or a claim of policy improvement.

## Frozen real Minecraft evaluation (3600 game ticks per case)

All below are **actual vanilla newly generated survival** starts using the
same seed stream root 917211 but disjoint `eval` namespace; `spawnRadius=0`
fixes same-seed starting position. Every frozen rollout had **zero** new
optimizer updates. Identical source game engine/input/perception boundaries.

| Heldout seed index | Controller | Independent action RNG | Wood actually mined | Logs actually acquired in inventory | Player-visible crack rises |
|---|---|---:|---:|---:|---:|
| 0 | new offline Q `qcrack_v1a1104` | 731117 | 2 | **2** | 52 |
| 0 | new offline Q `qcrack_v1a1104` | 813777 | 1 | **1** | 8 |
| 0 | original V9 PPO `v9longpilot1104` | 731117 | 0 | 0 | 13 |
| 0 | uniform-random same 20-action motor | 731117 | 0 | 0 | 37 |
| 1 | new offline Q | 731117 | 0 | 0 | 36 |
| 1 | original V9 PPO | 731117 | 0 | 0 | 26 |
| 2 | new offline Q | 731117 | 0 | 0 | 83 |
| 3 | **after** online Q training, separately exported frozen checkpoint | 731117 | 0 | 0 | 22 |

Source journals respectively:
`logs/fresh_eval_qcrack_hold0_1104.jsonl`,
`logs/fresh_eval_qcrack_alt0_1104.jsonl`,
`logs/fresh_eval_v9_qmatched0_1104.jsonl`,
`logs/fresh_eval_rnd_qmatched0_1104.jsonl`,
`logs/fresh_eval_qcrack_hold1_1104.jsonl`,
`logs/fresh_eval_v9_qmatched1_1104.jsonl`,
`logs/fresh_eval_qcrack_hold2_1104.jsonl`,
`logs/fresh_eval_qonline_hold3_1104.jsonl`.

**Interpretation:** Two different stochastic action sequences obtained logs
on *one* forest-capable heldout seed whereas matched V9/random did not. On
seeds 1 and 2 the Q policy had **zero wood under crosshair at neural action
boundaries**, so the missing capability remains natural resource approach,
perception, targeting and cross-seed generalization. Additional heldout
post-update seed3 had zero wood collected. Repeatedly sampling seed0 is NOT
independent multi-world evidence. This does not establish reliable transfer;
more blinded same-world controls, action RNGs and independent training seeds
are required before comparing model quality with statistical confidence.

## NEW: learn from its own next normal world (not only offline PPO replay)

A separate **`collect`** mode in `experiments/fresh_seed_treadmill.py`
loads a frozen Q snapshot initially, uses real ordinary vanilla inputs in
sequential NEVER-repeated train-seed worlds, records each complete world
under ignored `data-local/experience/RUN/episode_NUMBER.npz`, and **optionally
trains Q between worlds on its own recent actually archived real transitions**.
No teacher, no model-file overwrites, no user-provided route. Each world is
saved/exited and archive atomically published before Q gradient updates; the
new Q network/optimizer/RNG and stream cursor commit TOGETHER after updates.
An aborted world has no checkpoint and its ahead-of-checkpoint replay is
quarantined on resume. The initial stream checkpoint cursor correctly matches
its actual new-world seed index (not hardcoded zero). `--collect-seed-start 2`
avoids the two original source PPO training seeds0 and1. Up to 32 prior
recent real worlds are replayed, and, if none contains a real achieved goal,
the updater explicitly performs **zero** Q gradient steps rather than
inventing success. The policy never sees the seed/curriculum index.

**REAL start+resume integration, tag `qonline_smoke1104`:** New train seeds
4 and5, 600 six-tick units EACH, 3,600 real Minecraft ticks each, 245+225
= **470 neural choices**, 1,200 units / 7,200 ticks total. Both real
Minecraft launches saved and closed; complete per-world replay files
reopened; real own inventory in seed5 gained **one dark oak log** (and five
dirt items across run). The first episode had 10 real player crack-rise
outcomes, the next restored previous weights+optimizer+RNG and replayed 22
real positive outcomes total across both worlds. The session performed
**8+8 new CUDA Q updates**: model count 1,000 -> 1,008 -> 1,016. Verified
exact training checkpoint cursor **6**, units **1,200** and 2 archived worlds.
Per-tick MCio game I/O 2.33 and 2.71 milliseconds excluding worldgen/client
start; no novel performance regression. Output:
`logs/fresh_stream_qonline_smoke1104.jsonl`,
`logs/qonline_smoke1104_start.summary`,
`logs/qonline_smoke1104_resume.summary`. A separate genuine observed-item goal learner `qdarkoak1104` was also
trained from just ONE real collected `minecraft:dark_oak_log` outcome in
the two collector archives (470 actual motor actions; 500 scratch CUDA
updates). Frozen item practice on disjoint eval index4 acquired **zero**
actual dark oak; see `logs/fresh_eval_qdarkoak_hold4_1104.jsonl`. This proves
the generic item goal evaluation path runs, NOT item-goal skill competence.
The separate immutable
`checkpoints/replay_q_qonline_frozen1104.pt` was exported from the committed
trained collection checkpoint and verified compatible with frozen game eval
on seed index3; **that eval collected zero logs**. Training wood is not proof
of heldout competence.

The known outer headless Gamescope process can exit with `139` / segfault
AFTER the Java child, completed journal and saved checkpoint; all wrappers
independently verify the actual Minecraft world closure and checkpoint/JSONL
cursor before reporting PASS. This remains a separate teardown issue, not a
claim it is fixed. All older ignored local data remain preserved.

## Exact workstation commands (from the repository root)

```bash
# One-time experimental off-policy Q model from REAL previous V9 player replay:
.venv/bin/python experiments/train_replay_q.py --source-run v9longpilot1104 --run-tag qcrack_new01 --goal effect:visible_crack_progress --epochs 40 --updates-per-epoch 25 --batch-size 48 --device cuda

# Evaluate a frozen Q model on never-trained natural world indices / RNGs:
tools/run-replay-q-evaluation.sh qcrack_v1a1104 example_q_eval effect:visible_crack_progress 600 600 4 731117 q_sampled

# Begin a NEW Q self-play + GPU off-policy replay session on TRAIN world indices >=2:
GMAINCRAFT_Q_UPDATES_PER_WORLD=64 tools/run-replay-q-collection.sh start example_q_session 240000 1200 qcrack_v1a1104 effect:visible_crack_progress 2

# These operations act on the same run, cumulative target and EXACT options:
tools/run-replay-q-collection.sh status example_q_session
tools/run-replay-q-collection.sh stop example_q_session
GMAINCRAFT_Q_UPDATES_PER_WORLD=64 tools/run-replay-q-collection.sh resume example_q_session 240000 1200 qcrack_v1a1104 effect:visible_crack_progress 2
.venv/bin/python tools/inspect-experience.py example_q_session

# Export a separate immutable frozen model only from checkpoint-committed worlds:
.venv/bin/python tools/export-replay-q-snapshot.py example_q_session qsnapshot01
tools/run-replay-q-evaluation.sh qsnapshot01 qsnapshot_eval01 effect:visible_crack_progress 600 600 6 813777 q_sampled
```

**A larger live run was actually launched** under tag
`qselfplay_v1_1104` with **240,000 cumulative units / 1,200 per natural
train world**, up to 200 distinct new noise seeds starting at index2. It
uses 64 online Q updates after each world if genuine crack outcomes have
been archived, 32 prior worlds of replay, and 20 GiB rolling retention.
An independent connector invocation confirmed the detached process,
held advisory lock, live real Minecraft JVM and initial checkpoint. For
its CURRENT actual status or safe recovery, follow [NEXT_SESSION](NEXT_SESSION.md).
**First long-job world already verified:** new TRAIN seed index2
`2449750809322325767`, 1,200 six-tick units / 7,200 game ticks,
451 neural choices, 2 actual `minecraft:spruce_log`, 1 spruce sapling,
1 stick and 5 dirt inventory gains. Player-known replay for this episode
has 18 actual observed crack-rise transitions and prompted 64 new CUDA
Q training updates (1,000 -> 1,064). Exact checkpoint cursor3 and
player-only archived `.npz` were reopened. The next unique seed3 began
running with the active advisory lock. This is self-play TRAINING, **not**
heldout validation or evidence of reliable log gathering; monitor later
independent frozen runs before drawing performance conclusions.

Do not rerun the `start` examples with existing tags; the wrappers refuse
replacing old logs/weights. The longer session's exact real tag and status
are reported in `docs/NEXT_SESSION.md`. Use a single collector at a time to
avoid MCio/GameScope locks. Training worlds are consumed once from the train
namespace, not heldout; never train on eval journals. Further item-specific
practice can use self-collected inventory goals (e.g. a previously observed
`minecraft:dark_oak_log`) with `experiments/train_replay_q.py` on a committed
collector's player-only archive. **That remains a goal-instructed local skill,
not a learned task selector.**

## Next objective

Accumulate many more diverse self-earned inventory outcomes and bona fide
same-target crack progress; score a frozen Q snapshot at fixed heldout
indices and several independent action RNG seeds against original Q/V9 and
matched random controls. Improve broad resource approach and true
on-screen dropped-item observations; item- and action-interface expansion,
observed causal effects, then independently learned skill composition follow
those evidence gates. Do not simply treat 1-2 successful seeds as a solved
first-log policy or automatically add preauthored wood->stone->End phases.
