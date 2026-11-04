# Durable self-play experience and first learned-goal toolchain

**New research infrastructure, not a trained high-level planner.**
Read [the long-term architecture](BOOTSTRAP_TO_ANY_PERCENT.md) first.

## What is actually implemented

- Opt-in **player-knowledge-only replay recording** in the existing real
  fresh-survival V2–V8 treadmill. Each actual selected neural action records
  before/after bounded observations, ordinary motor inputs, game ticks,
  own inventory before/after, reward and episode terminal state. **No seed,
  hidden world/portal map, command, target oracle or training labels.**
- Compressed, independently addressable per-Minecraft-world replay files
  under Git-ignored 'data-local/experience/RUN/episode_NUMBER.npz'.
  The archive publishes only **after Minecraft closes normally** but
  **before its PPO checkpoint commits**. Interrupted ahead-of-checkpoint
  episodes are quarantined on resume. Readers compare source PPO cursor.
- Optional per-run rolling disk budget, default 20 GiB, or 0 to retain
  all replay. Oldest new replay episodes may be pruned with journal audit.
  Original V2–V8 models/worlds/logs are untouched.
- A small, separate from-scratch **goal-conditioned neural motor**.
  It trains via hindsight behavior cloning of the agent's **own actually
  achieved inventory increases**, with a goal item ID from observed items.
  This is a first baseline, NOT general RL, an Any% planner or a recipe list.
- A distinct frozen **goal practice evaluator** for fresh held-out
  Minecraft seeds. It sends normal motor actions and scores real owned
  inventory gains. The practice goal is provided explicitly for this
  *diagnostic*; the agent still has no autonomous high-level goal selector.

## Long-term training with replay recording

On the host, in your local repository:

~~~bash
cd "$HOME/Documents/minecraft-discovery"
GMAINCRAFT_RECORD_EXPERIENCE=1 tools/run-fresh-treadmill.sh start archivev8_01 forever 600 wood_sustain_v8
~~~

This starts a fresh V8 policy and produces never-repeated actual vanilla
survival worlds. It uses CUDA, durable PPO checkpoint/optimizer/RNG and
the new episode archive. The **foreground shell remains occupied** until
stopped; no daemon or host service is installed.

~~~bash
tools/run-fresh-treadmill.sh status archivev8_01
tools/run-fresh-treadmill.sh stop archivev8_01
GMAINCRAFT_RECORD_EXPERIENCE=1 tools/run-fresh-treadmill.sh resume archivev8_01 forever 600 wood_sustain_v8
~~~

Always supply the record env var again on resumption. The recorder
does not alter original model checkpoint metadata or training rewards;
a run that previously started without recording can resume with it.
Each new run tag must be unique. The source JSONL may record seeds for
research but they never appear as skill input or archive fields.

For a rolling 50 GiB disk budget use
'GMAINCRAFT_EXPERIENCE_LIMIT_GB=50'; for NO replay pruning, set
'GMAINCRAFT_EXPERIENCE_LIMIT_GB=0' (monitor your filesystem).
Single episode safety cap:
'GMAINCRAFT_EXPERIENCE_MAX_EPISODE_MB=256' by default.
A failed/oversized archive halts rather than silently discarding
uncommitted experience. Never delete a held advisory '.lock' file.

**Important:** V8 currently fails the held-out conditional wood-mining
reliability test. You *can* run V8 with replay for months, but more V8
cycles alone do not guarantee mastering wood. A new better learning
method for player-visible target/crack history is a separate next gate.

## Inspect real successful inventory outcomes

~~~bash
.venv/bin/python tools/inspect-experience.py archivev8_01
~~~

The output reports only PPO-committed episodes, neural action count,
game ticks, compressed size and ACTUAL newly owned item types/counts.
A training run that sees 10,000 wood surfaces but collects zero logs
must report **zero acquired wood goals**.

Goal relabeling traces back a bounded number of normal actions from
real inventory increases within the same world. It excludes prior
states that already held more of the item. Its positive samples are
not expert demonstrations; some actions may be incidental rather
than causal. Next methods should compare hindsight RL and learned
dynamics, not assume this imitation baseline will succeed.

## Train a new goal-conditioned skill from self-play

After inventory gains have genuinely appeared in the replay:

~~~bash
.venv/bin/python experiments/train_hindsight_goals.py --source-run archivev8_01 --run-tag ownskill01 --epochs 20 --device cuda
~~~

Resume after accumulating more completed training worlds:

~~~bash
.venv/bin/python experiments/train_hindsight_goals.py --source-run archivev8_01 --run-tag ownskill01 --resume --epochs 40 --device cuda
~~~

The epoch target is **cumulative**. Model+optimizer live in
'checkpoints/hindsight_ownskill01.pt'; its ignored journal is
'logs/hindsight_ownskill01.jsonl'. If real inventory gains have not
occurred, the script refuses to create a fake successful policy.
A withheld subset of own training episodes measures in-dataset
imitation generalization; it is NOT final held-out Minecraft scoring.

## Evaluate a frozen skill in real Minecraft

~~~bash
tools/run-hindsight-evaluation.sh ownskill01 ownskill_eval01 minecraft:oak_log 2400 600
~~~

The practice goal is specifically oak logs. A manually chosen
practice item is a *task instruction*, NOT a scripted harvesting
sequence or a hidden world coordinate; full autonomous goal selection
remains to be learned. Use new evaluation tags and action RNG seeds
for repetitions. The evaluator checks the source seed root, motor
profile, observation schema and actual inventory outcomes. It does
not update the neural network, teleport the player or cheat via
administrative commands.

**Not implemented yet:** real item collection mastery, full use/place/
hotbar/crafting UI, autonomous option selection, learned skill
initiation/termination, causal planning, or dragon/exit scoring.
Offline imitation accuracy is not a Minecraft skill success rate.

## Verified evidence and limitations

The independent real Minecraft/CUDA smoke 'replayproof1104a'
generated one fresh normal survival seed, completed **36 six-tick units,
216 Minecraft ticks, 10 neural selections, 2 PPO CUDA updates**, broke
2 ordinary blocks, saw 9 new wood surfaces and gained **zero wood**.
Then the SAME checkpoint was successfully resumed for a DIFFERENT natural
noise seed, another 36 game-time units / 216 ticks / 9 neural actions
and 2 PPO CUDA updates, with 0 wood inventory. The trainer committed
both episodes: **432 total real game ticks, 19 motor choices and 4 CUDA
PPO updates**, one valid model checkpoint plus TWO distinct compressed
replay files, both reopened and verified against the journal. Outer Gamescope exited with its previously known post-child
139, but the actual game and model run completed, as shown by its
independently verified journal/checkpoint.

Pure tests separately demonstrate synthetic *test-only* inventory
gains, actual archived feature shapes, replay safety, crash quarantine,
disk retention, hindsight labeling, scratch supervised neural learning,
checkpoint resume, and strict goal-eval source/motor validation.
**No fake goal examples were inserted into actual Minecraft replay.**

### Next research gates

1. A new versioned natural actor with real target/crack action history
   and/or unbiased-eval forest-rich training curriculum to create
   significantly more genuine self-achieved resource outcomes.
2. Replay-based goal-conditioned **RL** or off-policy actor learning
   using observed action consequences, compared with this supervised
   baseline on real fresh seeds.
3. Autonomous option applicability/termination and a two-skill manager
   with actual independent phase switching, not a prescribed recipe.
4. Complete vanilla controls and bounded player-visible item/entity
   perception, then full-game outcome-model learning.
