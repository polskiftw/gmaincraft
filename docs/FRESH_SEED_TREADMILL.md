# Fresh-seed Minecraft survival training treadmill (current primary runner)

**Latest checked November 3:** The original `speedrun01` completed 24,000
six-tick decisions / ten seeds and exhibited action collapse: see
[`SPEEDRUN01_POLICY_COLLAPSE_2026_11_03.md`](SPEEDRUN01_POLICY_COLLAPSE_2026_11_03.md).
The optional `wood_progress_v3` uses one-time visible wood crack advancement,
plus existing v2 reward, learning rate 0.00015 and entropy coefficient 0.04.
It needs a **new** training run tag; original v2 remains unchanged. Example:
`tools/run-fresh-treadmill.sh start woodv3_01 24000 2400 wood_progress_v3`.
The training wrapper's fifth argument selects the reward version, and the
frozen evaluation wrapper's sixth argument does likewise. An actual
96-decision Minecraft v3 smoke and 130 pure tests passed; **no natural
wood pickup yet**.


**Nov 3, 2026. Tested in actual Minecraft Java 1.21.3 / MCio / Gentoo NVIDIA.**
This supersedes the previous *eight fixed worlds rotated forever* long-run
launcher. The user wants a FROM-SCRATCH neural speedrunning agent that
learns in an open-ended stream of new vanilla survival levels and eventually
works on ANY previously unseen random seed. Repeated pre-aimed oak or
pre-placed item lessons remain only deterministic diagnostics.

## Start training — you DO NOT need to prepare a set of maps

In the isolated Gentoo Shell `/workspace` (not a host-level Minecraft
installation), launch a normal foreground terminal job:

```bash
cd /workspace
tools/run-fresh-treadmill.sh start mytrain01 forever
```

This automatically generates a new random-looking **64-bit Minecraft world
seed** whenever an episode ends, creates real `minecraft:noise` survival
terrain with the upstream Java server, plays ordinary vanilla game actions,
updates the SAME from-scratch CUDA policy, commits its training checkpoint,
and moves to the **next NEW seed**. It does NOT cycle through a small fixed
training cohort. The only limit to the seed stream is an astronomically large
64-bit space; `'forever'` means **until you stop the process**, not a daemon.
No game-world fixture, tree placement, target teleport, artificial item
spawn, phase handoff, predefined route, world-map lookup or prerecorded
human demonstration is involved in scored gameplay.

The recommended per-seed duration is currently **2,400 policy decisions**
(14,400 simulated Minecraft game ticks). It is configurable; changing it
on an existing run is intentionally rejected to preserve checkpoint
comparability. To start a bounded training experiment instead:

```bash
tools/run-fresh-treadmill.sh start mytrial01 100000
```

Or explicitly request a different **per-seed** horizon when STARTING:

```bash
tools/run-fresh-treadmill.sh start mytrial02 100000 1200
```

Monitor, request a graceful stop, or resume EXACTLY the same policy with
all optimizer/RNG state and the next **never-before-consumed seed**:

```bash
tools/run-fresh-treadmill.sh status mytrain01
tail -f logs/fresh_treadmill_mytrain01_latest.log
tools/run-fresh-treadmill.sh stop mytrain01
tools/run-fresh-treadmill.sh resume mytrain01 forever
```

The 2,400-decision horizon is a compromise between new-seed variety and
Minecraft restart overhead. If your original run used a 1,200-decision
horizon, **resume must also supply 1,200** as the last argument. When using
numeric targets, `resume mytrial01 150000` means **150,000 total decisions**,
not an extra 150,000. You can repeatedly raise the target. `stop` safely
finishes an in-progress normal action and closes Minecraft; it consumes a
partial seed, so that a subsequent `resume` receives a fresh one.

**No background service was installed or scheduled.** The user must run the
command in a terminal. It will remain a foreground process until stopped
or the specified decision target is reached. To display the nested game
window instead of NVIDIA headless Gamescope, choose the existing permitted
SDL backend with `GMAINCRAFT_GAMESCOPE_BACKEND=sdl` for the run; this is
only a scoped environment variable, not a change to the host GPU/settings.

## What is actually shared between fresh worlds?

**Shared and learned:** one randomly initialized `NaturalPPO` model and
its optimizer, PyTorch CPU/CUDA RNG state and a global total of real
observed gameplay and inventory outcomes. Every decision chooses a normal
movement, camera movement and attack together from direct player-visible
Minecraft memory, identical in training and frozen evaluation. World seeds
exist in the runner's audit log but **are NEVER input to the policy**.

**Reset on each new episode:** the 2x2-block visitation/first-hit surface
novelty memory and current world's cumulative mined/picked-up stats. The
new vanilla game starts without artificially inherited inventory or
knowledge of the previous world's terrain. This is intentional: the
network must learn transferrable Minecraft behaviors, not memorize a
single world's map. Current model observation is typed
`natural_memory_online_v1_first_hit_objects_ppo6`: own health, hunger,
position/heading, crosshair, block-breaking crack animation, own inventory,
and up to 24 13x9 first-hit camera block surfaces. These remain imperfectly
calibrated for true pixel-level human visibility (especially glass/leaves
and tiny distant targets), and **dropped item entities are not yet supplied**.
The first wood acquisition in an unscripted natural world is STILL unsolved.

**Nature of the random seeds:** `SeedSchedule` uses a versioned,
full-width, invertible 64-bit SplitMix-style permutation on an increasing
seed cursor. The result is a pseudorandom-looking and **nonrepeating**
stream of both positive/negative and even/odd signed Java long seeds.
The trainee uses even *internal permutation counters* and the evaluator
uses odd internal counters; because the mapping is a bijection, the two
sets are mathematically disjoint without restricting output seed parity.
Each stream has up to **2^63 unique seeds** before wraparound is forbidden.
This is deterministic for checkpoint reproducibility, not repeated maps.
Its root can be changed **only when starting a brand-new run**, and frozen
evaluation must use the SAME root as its training checkpoint.

For an apples-to-apples comparison, the generated world has Minecraft's
vanilla `spawnRadius` gamerule set to **0**. This does NOT change the world
seed, biome, trees, structures or vanilla noise terrain; it only removes
per-player random scattering around the world's real spawn point. Both
frozen policy and uniform-random controls then start at the same spawn
coordinates on the same seed. Original MCio's old controlled fixture
commands remain unused in this primary trainer; the source generator
only enables the usual isolated MCio world configuration for observation.

## Evaluation (genuinely unseen Minecraft seeds)

Use a separate bounded evaluation command with a frozen model, and then
the equal-seed uniform-random control. E.g. after training `mytrain01`:

```bash
tools/run-fresh-evaluation.sh mytrain01 eval01 9600 policy_sampled
tools/run-fresh-evaluation.sh - random01 9600 uniform_random
```

Each command generates the **exact same series of held-out seeds**, because
the test sequence's root and unique odd internal counters are fixed.
Neither the model nor the random baseline is allowed a scripted reset
inside gameplay or an admin command. The frozen model receives NO optimizer
updates. Each separate eval tag gets fresh independent world copies and
an immutable JSONL report. Two other supported controls are
`policy_greedy` (argmax, which previously showed poor stationary collapse)
and `untrained` (an independent scratch neural model, using `TRAIN_RUN=-`).
A true speedrunning score requires actual inventory, progression and
ultimately the game completion event, not just discovering a wood-looking
surface or accumulating synthetic novelty rewards.

**Scientific caveat from the pilot:** The earlier pre-fix quick experiments
used default Minecraft `spawnRadius=10`, so although the policy and random
controller had identical seeds, they sometimes started a few blocks apart.
These old tests are **NOT strict matched-start policy comparisons**.
They remain valid world-generation/checkpoint smoke tests. New evaluations
pin spawn scatter to zero and verify the world metadata. Changing this
training environment deliberately invalidates old pilot run metadata for
safe resume; their ignored worlds and checkpoints are preserved untouched.

## Minecraft lifecycle and measured speed

**Requested ideal:** replace a world without restarting the full Minecraft
client. **Current verified fallback:** each seed requires a fresh Minecraft
client and integrated server. MCio 1.6.1 `sync` has a global
`MCioSyncUtil` client/server semaphore handoff and a one-way
`gameRunning` transition; it does NOT currently implement a tested
hot-world reload/reset or new integrated-server re-synchronization API.
Calling QuickPlay/world switching mid-sync could deadlock the two threads
or report incorrectly aligned observations. Rather than trust an unverified
fast path for a long unattended training run, we preserve the proven
restart path, record its overhead and amortize it with 2,400 decisions per
seed. A future **separate** hot-switch R&D branch should implement and
falsify proper sync reset, loading screens, server thread disposal, frame
sequence reset and zero ghost inputs before being made the default.

Observed actual smoke costs on this Gentoo workstation (multiple new
worlds, no Torch during generation):

| Measured phase | Approximate wall time per new seed |
|---|---:|
| Java vanilla world generation | **6.6–7.4 s** |
| Minecraft client startup, connect, first fresh observation | **9.4–14.3 s** |
| Each warmed real-game action | variable; typically faster than cold starts |

The shortest 12-decision smoke episodes intentionally spent most of their
wall time starting Minecraft. They do NOT reflect optimized amortized
2,400-decision performance, and no single-run wall-time promise is made.
Every episode logs `worldgen_s`, `client_startup_s`, `game_io_s`,
`game_io_ms_per_tick`, actual mined blocks/wood/inventory and neural
updates. A high-latency warning is emitted rather than silently accepting
a performance regression. World generation and hot-switch benchmark data
are kept out of model inputs and reward calculations.

## Checkpoint, interruption and disk-retention safety

Each training run has its own exclusive file lock:
`checkpoints/fresh_stream_<tag>.lock`. The runner refuses concurrent
writers, silently changed policy schema, changed world-generation rules,
changed reward profile, changed seed stream/root, changed model seed or
changed episode horizon. It also refuses to overwrite an existing tag;
use `resume` rather than starting a second run with the same name.

Its main persistent artifacts (all under `/workspace` and Git-ignored):

- `checkpoints/fresh_stream_<tag>.pt`: atomic network/optimizer/RNG state,
  total **committed** decisions, next unique seed cursor and real-event
  totals. No pretraining or external model weights are imported.
- `logs/fresh_stream_<tag>.jsonl`: append-only episode startup, memory
  discovery, game event, PPO update, performance and checkpoint records.
  Rows labelled `uncommitted_*` MUST NOT be counted after a sudden crash.
- `logs/fresh_stream_<tag>.inflight.json`: fsynced atomic transaction marker
  recording the seed/index being played. Removed ONLY after Minecraft
  shutdown, final world verification and atomic model checkpoint.
- `logs/fresh_stream_<tag>.stop`: one-shot graceful-stop request. It is
  automatically removed after successful clean pause.
- `mcio-data/instances/DemoInstance/saves/MDL_FreshStream_<tag>_...W`:
  uniquely named disposable game-world saves. By default **only the two
  most recently completed worlds** are retained for inspection; prior
  completed owned worlds are removed *after* their checkpoint commits.
  The just-generated source save is also removed only after its verified
  live copy is made. This avoids disk growth across millions of worlds.

If a process dies **mid-episode**, its model has NOT committed the new
unverified rollout. The next explicit `resume` uses the previous good
checkpoint. It moves the owned incomplete episode into
`mcio-data/aborted_fresh_streams/` for inspection, deregisters that
unfinished Minecraft save, and regenerates the SAME never-successfully-
consumed seed from scratch. No incomplete world is silently resumed,
no truncated episode receives reward, and no unrelated personal save or
checkpoint is touched. A crash after the model checkpoint but before
clearing its marker is recognized as already committed (no double-count).
If a crash truncates the **last JSONL journal record**, the resume path
preserves the partial bytes in a uniquely named local file and repairs
only that final line before appending; complete valid lines stay intact.
Incomplete checkpoint/marker staging files are quarantined rather than
overwritten. A corrupted complete record or conflicting artifact still
fails closed for inspection rather than guessing or inventing progress.
A `stop` command checks the run's exclusive file lock and refuses to
leave a stale stop marker if there is no active training process.

`--retain-worlds N` is available in the underlying Python runner (0–10),
with the wrapper default N=2. Quarantined failure snapshots are
**intentionally not automatically erased**, since they are debugging
evidence; repeated hard failures can grow that directory and need manual
review. Clean stop/resume and the explicit recoverable incomplete-world
path have been tested. Sudden machine power failure at arbitrary storage
write boundaries remains less tested than ordinary graceful exit and
an intentionally incomplete local episode; this is not a promise of
perfect transactional filesystem semantics.

## Actual verified experiments

These are **real Minecraft** observations and outcomes, not faked
`torch` or scripted-tree reward samples. Very short horizons were used
to verify the kit without launching an hours-long training job in the
user's environment. None demonstrates learned random-seed log pickup.

| Test and world policy | Actually generated seeds | Decisions | Verified result |
|---|---:|---:|---|
| Initial proof `smoke_1103a` (old v1 seed partition, spawn scatter) | 4 | 48 | Resumed same model from 24 to 48; 8 PPO updates; actual world copies checked |
| Mixed-parity seed proof `v2wrapper_1103a` (spawn scatter still default) | 3 | 36 | Public launcher start/resume, one config row, 3 unique real seeds, optimizer continued |
| `v2stop_1103a` (old spawn setting) | 2 | 8 then 16 | Real in-game `.stop` request paused after 8; resumed NEXT seed, total 24 and 2 updates |
| Recovery `v2recovery_1103a` | 1 retried | 12 | Deliberately incomplete owned world quarantined, then same seed regenerated; only completed 12 decisions committed |
| **Current fixed-spawn** `v2fixedtrain_1103a` | 1 | 12 | Real NBT confirmed Minecraft noise terrain and `spawnRadius=0`; one random-weight PPO update |
| **Current fixed-spawn** frozen `v2fixedpol_1103a` | 1 held out | 12 | Same start XYZ as random, 1 non-wood block mined, 0 inventory wood, 0 training updates |
| **Current fixed-spawn** uniform random `v2fixedrnd_1103a` | same held-out seed | 12 | Exact same start XYZ `(48.5, 64.0, 32.5)`, 0 blocks mined, 0 inventory wood |

The current single-seed held-out comparison is only a **correctness
control**, not a statistically meaningful advantage over random. A prior
720-decision fixed-cohort baseline also collected no natural wood. Real
random-seed Minecraft speedrunning still requires substantial work on
navigation, movement, targeting, attack credit, item perception,
inventory and crafting. The goal of this iteration is to make unbounded,
non-memorizing **training itself** dependable and measurable.

All new code has pure unit tests for billions-scale deterministic unique
seed scheduling, disjoint train/eval domains, signed long bounds, typed
config validation, exact checkpoint schema, atomic marker creation,
crash quarantine, safe ownership/name validation and stale-marker
recovery. The full Python suite passed **118 tests** at the last
integration check, plus independent actual Minecraft world creation,
training, restart/resume, clean-stop, held-out comparisons and recovery.
The original Fabric sensor JAR and old training checkpoints were neither
replaced nor deleted by this work. Nothing runs indefinitely unless the
user explicitly executes the `forever` foreground command.

## Next development effort (not another fixed scripted-arena loop)

1. Run longer fresh-seed **training** and independently sampled/greedy
   **held-out evaluation**, always paired with a uniform-random baseline
   on the exact same seeds and spawn position. The first milestone is
   **actual unscripted wood in inventory**, not simply novelty reward.
2. Improve policy learning/exploration when greedy actions collapse into
   stationary loops, and instrument death/respawn/action effectiveness.
3. Replace the remaining blind spot in perception with a bounded,
   player-visible **dropped item entity** memory channel; add item use,
   crafting and more continuous action duration in versioned model inputs.
4. Investigate a truly persistent Minecraft client hot-switch as a
   **separate measured experiment** with safe MCio client/server sync
   reset, not as a hidden assumption in this reliable treadmill.

Do not switch the new primary project back to repeated pre-aimed oak or
fixed-world benchmarks. Those earlier scenes remain useful as occasional
regression diagnostics, not the way to train a general speedrunner.
