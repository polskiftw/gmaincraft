## Nov 5: evidence-gated numeric item learning

- [x] Finish prior staged numeric 200-episode study; frozen model 54-55/100
  against random 56/100 on 100 independent cases: **no learned advantage**.
- [x] Implement separately versioned egocentric dropped-item v2 observation
  + scratch real-transition double Q, persistent success replay and strict
  resumed checkpoint; no hidden fixture position/action teacher.
- [x] Improve fixture distribution: world yaw across 360 degrees and item
  bearings ahead/lateral/behind; keep vanilla inputs and camera-only eyes.
- [x] Run real 60-case training (25/60 true inventory pickups, 612 Q
  updates), freeze, evaluate twice (2*100 independent case seeds) against
  random and fixed forward: pooled greedy 139/200 vs random 96/200 and
  forward 97/200. Scored command count zero; frozen updates zero.
- [x] Build a reproducible two-seed, inventory-based real-game promotion
  gate requiring +10 points above BOTH controls; provisional PASS (+21).
- [x] Regression suite 203 tests, bounded fixture cleanup retry and docs.
- [ ] Reproduce advantage with other initial weights / independent run RNGs
  and independently heldout scenario groups. Track hard/off-screen cases.
- [ ] Transition from summoned lab item to naturally mined ItemEntity in
  vanilla survival, with learned motor/skill switching (no test-only phase
  handoff). Do NOT call this a natural-world speedrunning policy yet.
- [ ] Investigate Gamescope exit139 after completed valid Python journals;
  separate from evidence of network/game success.

## Nov 4: finished Q and visible dropped-item research

- [x] Finished qv2memtrain1104: 20 train worlds, cursor326,
  24000 six-tick units, 38 wood inventory gains, 53 mined blocks,
  13/20 positive worlds, cumulative Q updates 4364.
- [x] Export immutable qv2memf1104 frozen checkpoint.
- [ ] Complete disjoint frozen heldout eval on indices16..23 with two
  action RNGs vs qv2longf1104, no model updates.
- [x] Implement opt-in, bounded, on-demand camera-frustum and ray-tested
  dropped-item observation as research-only separate typed schema.
- [x] Compile pinned Fabric mod; unit-test request and parser contracts
  without mutating old learned observation.
- [ ] After frozen eval exits, install backed-up research mod in only
  DemoInstance; run actual exposed/stone/glass/leaves/away item test.
- [ ] Research staged visual recognizability and natural mined-drop pickup.
- [ ] Design a new versioned entity-aware pickup/navigation learning skill.
- [ ] Avoid more identical Q wood-crack reward training without better
  independent frozen success.

## Memory-as-eyes audit, November 4

- [x] Inspect active model inputs, screenshot flow and known visibility failures.
- [x] Skip unnecessary Python RGB conversion in memory-only training while preserving legacy RGB path.
- [x] Benchmark six matched RGB/zero-RGB frozen worlds: 13.77% lower game I/O, 5.87% lower wall time; player sensor unchanged.
- [x] Build experimental opt-in zero-RGB MCio protocol preserving original capture callback and sync.
- [x] Install/test zero-RGB protocol in disposable DemoInstance, old JAR backed up, real default RGB with 10 screenshots also verified.
- [ ] Research a genuinely capture-independent tick protocol separately; current mode still renders but skips readback.
- [ ] Calibrate visibility across transparent/partial occluders, lighting, FOV and object size.
- [ ] Add bounded genuinely visible dropped item entity representation.
- [x] Freeze completed 100-world Q training as immutable qv2longf1104 checkpoint.
- [x] Complete matched heldout two-RNG comparison on seeds10-15: new V2 wood13 vs prior wood9, both 4/12 positive runs; new had more deaths.
- [x] Launch 20-world qv2memtrain1104 at TRAIN index306, immutable previous model with optional zero-RGB only; lock/first world verified.
- [ ] Check/finish new qv2memtrain1104; freeze committed Q and compare on NEW heldout seeds and independent RNG.
- [ ] Investigate post-completion launcher segfault separately.

## CURRENT NOV 4: finished 200-world V1, V2 rare own-success learner / durable lineage

- [x] V1 `qselfplay_v1_1104` finished 200 natural train worlds,
  240,000 six-tick units, 139,165 motor choices, 9,512 cumulative Q
  updates but only 3 true inventory log pickups in earliest 20 worlds.
  No reliable wood skill; MCio I/O stable ~2.0-2.3ms/tick.
- [x] Frozen SAME-real-state audit: Q stationary attack probability with
  wood under crosshair collapsed from 0.1130 to 0.00042, despite late
  worlds containing visible wood; fix sample/reward target before scaling.
- [x] V2 explicit own-visible wood-crack outcome, rare-success stratified
  replay with 52 actual positives from archived 200 train worlds,
  conservative double-Q, trained-action imitation and frozen older own
  policy anchor outside wood. No invented attacks, target coords or routes.
- [x] V2 full CUDA offline 500-update train; frozen unseen world0 got
  2 real logs, world6 4/3 with independent RNGs vs old Q0, world7 1 vs
  old Q0; world8/9 none (no wood under crosshair). Seed7 spawn Y drift
  means imperfect matched control. Not statistically reliable mastery.
- [x] Real corrected V2 start+resume new world indices203-204: 481
  ordinary actions, 3 oak + 1 birch real pickups, 16 CUDA Q updates,
  strict V2 outer/inner checkpoint schema and two archived worlds.
- [x] Real V2 chained smoke index205: one more true log pickup and nine
  new own successful motor effects, eight CUDA updates; original 52
  successes + newer 23 retained. Preserve corrupted earlier outer-V1
  smoke tag202 unchanged, do not silently migrate.
- [x] Versioned immutable V2 snapshot lineage: `qv2triplef1104` tracks
  52+23+9=84 actual player-known successes from THREE previous completed
  train sources; follow source cursors and seed split, NO heldout replay.
- [x] LAUNCH `qwood_v2long1104` longer V2: max 100 fresh natural
  1,200-unit worlds starting at previously unused TRAIN index206,
  84 protected previous real successes, 32 CUDA Q updates per world,
  20 GiB replay cap and strict versioned resume. Detached real Minecraft
  Java process and held run lock independently verified.
- [x] First new V2 long-train seed index206 committed: 1,200 units /
  7,200 ticks, **5 real log pickups**, 34 true new wood-crack motor
  outcome labels, 32 CUDA Q updates (524->556), exact checkpoint
  cursor207, new seed index207 already launched.
- [ ] Recheck the ongoing V2 long run without interrupting it, report
  success/targeting trend by independent groups of new train worlds,
  saved optimizer cursor, own inventory goals, performance and failures.
  If complete: frozen multi-seed and multi-action-RNG controls versus
  starting V2 and original Q before any further scale-up.
- [ ] Frozen progress gate on MANY independent unseen natural seeds and
  sampled RNGs vs original Q/V1 final, matched random, and V2 pretraining.
- [ ] Improve wood access, dropped item visibility, whole item/vanilla
  action/crafting UI, general achieved item-goal and learned skill manager.

## NEWEST: replay Q online self-play and independent frozen gate

- [x] Implement versioned conservative double-Q from only original V9
  player-known before/after archived states, real inventory and observed
  crack-stage outcomes; no human action teacher or hidden/seed information.
- [x] From scratch, 1000 CUDA replay-Q updates on 449 actual V9 real-world
  transitions with only FOUR real crack-progress outcomes. Frozen heldout
  seed0 gained 2/1 wood items at two RNG seeds vs original V9 0/random0;
  heldout indices1/2 gained zero (wood crosshair never observed at action
  boundaries); UPDATED Q snapshot heldout index3 zero. NO reliable skill.
- [x] Separate crash-aligned Q-controlled natural self-play with atomic
  real own-memory transition archive, 32-world prioritized online replay,
  exact-optimizer resume, strict phase-free motor/action/sensor contract,
  and immutable frozen checkpoint export. Initial cursor starts at exact
  train-seed index to make crash recovery safe.
- [x] REAL two-world online Q start/resume across new noise seeds4/5:
  470 ordinary actions, 7200 game ticks, 16 new CUDA replay updates,
  one actual dark-oak log acquired, 2 verified replay files and checkpoint.
- [x] Launch extended Q self-play `qselfplay_v1_1104` from train index2
  (avoid original V9 training seeds0/1); 240,000 cumulative units, up to
  200 fresh 1,200-unit worlds, bounded 20 GiB replay, 64 GPU Q updates
  per world with observed successes; verify launcher/JVM and held lock.
- [x] Verify the FIRST committed `qselfplay_v1_1104` real new train world:
  451 real neural decisions, actual 2 spruce logs + 1 sapling + 1 stick
  + 5 dirt, 18 true crack-rise replay outcomes and 64 CUDA Q updates.
  Saved checkpoint cursor3, 1,200 units, 1,064 Q optimizer steps;
  next world index3 started under the held lock.
- [ ] Recheck the detached job and its REAL committed episodes/optimizer
  updates after the longer session; resume safely only if the lock is free.
- [ ] Freeze later Q weights and compare on many NEW unbiased heldout seeds,
  independent action RNGs; report actual logs owned vs original Q/V9/random.
- [ ] Improve resource approach and own-visible dropped item perception;
  learn generalized actual item-goal skills and eventually learned independent
  two-skill switching, not a hardcoded route.

## Highest priority after V7/V8 measured real PPO sample-efficiency failure

- [x] Quantify the real training budget: V5 1,118 neural choices, 21 crosshair-wood, nine attack-on-wood, **zero** stationary attack-on-wood; original random four-head stationary long attack before aim has probability 1/252.
- [x] V7 versioned 20-way learned normal-input motor (no auto-target), one selectable held attack, 10 natural worlds/6,000 units/82 updates: 1 true training wood, heldout 0/4.
- [x] V8 versioned reward (same motor, player-visible 0.75 per six-tick bounded attack-on-wood, less passive sighting) 10 worlds/6,000 units/82 updates: 2 real training wood; heldout default sampled 1/4, alternate sampled RNG 0/2; random V8 motor mined wood but did not acquire it on seeds6-7.
- [x] Explicit offline counterfactual probes confirm V7/V8 policy stationary-attack probability changes by less than ~0.3 percentage points when the crosshair switches to wood in one controlled typed player observation. Neither model learned reliable wood-conditional action.
- [x] Verify 149 pure tests and saved strict-schema ignored checkpoints for original V2-V6, V7 and V8; preserve a full NVIDIA SIGFPE failed eval and successful fresh-name rerun.
- [ ] NEXT: investigate train-only curriculum from REAL vanilla forest-rich worlds and/or bounded player-own previous visible target + mining stage, with explicit honest training bias and unbiased random heldout noise seeds.
- [ ] Establish early model-selection gate from conditional response on actual wood-targeted states, not reward-count hacks; evaluate across independent action RNGs and enough unseen seed worlds before long training.
- [ ] Train item perception/visible dropped-log pickup separately after wood mining is reproducible without any evaluator phase handoff.

## Latest priorities: V5/V6 complete and CUDA healthy (Nov 3)

- [x] Verified after Gentoo Shell bridge restart: `nvidia_uvm` node available and PyTorch CUDA tensor compute on RTX 4070 SUPER passed.
- [x] V5 `wood_target_v5`: 10 true fresh survival worlds, 6,000 units, 41 updates, training one actual collected wood; frozen default RNG 2/4 wood on heldout seeds6-9.
- [x] V6 `wood_engagement_v6`: new schema explicitly labels only visible wood in current first-hit camera and crosshair; 10 natural worlds, 6,000 units, 41 updates, zero training wood and 0/4 heldout wood.
- [x] Cross-check V5 different action RNG and greedy: 0/4 alternative sampled RNG, 0/2 greedy on heldout seeds6-7. Reliable learned wood harvesting NOT established.
- [x] All 143 pure tests, including CUDA test, passed. Historical models/checkpoints preserved.
- [ ] NEXT major work: diagnose and teach sustained correct target + attack + stable camera and actual growing visible crack stage (no scripted commands/aim).
- [ ] On a newly versioned agent, evaluate many unseen Minecraft seeds with several independent *policy* RNG seeds, comparing actual wood inventory to current V5/V6/V4 and random; watch for greedy collapse.
- [ ] Only after independent real inventory gains are robust should the next long 24k+ training round start; no repeating previous objectives blindly.

## Current November 3 priorities after completed V4 24k run

- [x] Inspect `persistpilot1103` complete journal: 40 worlds, 3 true wood inventory pickups, no broad generalization claim.
- [x] Audit two fresh unseen seeds 4/5 against frozen pilot and random controls (grown 0 wood, pilot 0, random 1).
- [x] Provide optional CPU frozen-evaluation device for isolated CUDA unavailability without touching training defaults.
- [ ] Fix/diagnose isolated CUDA access before another full GPU train; `/dev/nvidia-uvm` absent; do not alter host modules from sandbox.
- [ ] Investigate and improve visible-wood attack alignment + uninterrupted no-look crack accumulation; version any reward/motor change as a NEW experiment.
- [ ] Compare next model against matched unseen random and frozen V4 on enough seeds; inventory wood is success, visibility and ordinary mined are diagnostics.

# TODO — gmaincraft

## V9 measured continuation: new target memory works; learning not proven

- [x] New V9 own-visible target/crack/action history with 10 bounded numeric
  features, separate schema and checkpoint; preserve V8 motor and reward.
- [x] Real two-world 240-unit matched V9/V8 scratch comparison; neither
  gained real wood inventory. Strict model/replay compatibility tests.
- [x] Independent V9 real start + resumed new-world 2 x 600 units,
  449 actions and 15 CUDA updates; 24 regular blocks, one genuine
  gravel inventory gain, zero wood. Both player-only archives saved.
- [x] Offline frozen history ablation on 449 ACTUAL own observation states;
  mean total variation 0.00312, NOT proof of learned memory use.
- [x] Train separate CUDA item-goal neural network from genuine gravel
  outcome, then frozen natural seed evaluations; failed gravel objective.
  Found/fixed goal-controller alternate-RNG reproducibility bug.
- [ ] NEXT: increase genuine training exposure to naturally occurring
  resources without target coordinates, preaimed trees or injected routes.
  Then compare a sample-efficient off-policy learner against current PPO.
- [ ] Multiple unbiased heldout seeds, multiple action RNGs, actual
  inventory gains and truly autonomous skill switching before scaling.


## IMPLEMENTED toolchain: years-scale replay and goal-conditioned practice

- [x] Optional, versioned player-knowledge-only before/after action archive
  in existing CUDA fresh-seed PPO. No seed/world geometry, no raw MCio
  packet, no evaluator-only stat, and V2-V8 checkpoint format unchanged.
- [x] Atomic compressed per-episode files published after normal JVM exit
  and before PPO checkpoint; replay readers require original committed
  checkpoint cursor; incomplete files quarantined on resume.
- [x] Bounded per-run replay cap (configurable, 0 unlimited) and live
  action/tick/inventory-inspection CLI. Preserve old weights/worlds.
- [x] Learn independent from-scratch goal-conditioned motor by imitation
  of the policy's SELF-achieved actual item inventory gains. Checkpoint
  optimizer/epochs; safe resume; never invent success labels.
- [x] Connect goal model to frozen live Minecraft evaluation with known
  practice item ID, normal game inputs, exact own-inventory scoring and
  source observation/motor/seed-root contract checks.
- [x] Two REAL CUDA+Minecraft fresh-seed sessions, including resuming
  original optimizer/RNG: 2 worlds, 432 ticks, 19 neural actions, 4
  CUDA PPO updates, 2 preserved replay files, 0 inventory goals.
  Synthetic test-only achievement/goal network checks isolated.
- [ ] Build a new real V9 sample-efficient learner with player-owned
  temporal target/crack history and/or natural forest curriculum. Capture
  enough real item gains to train and test replay-conditioned item skills.
- [ ] Train/evaluate learned goals on real acquisitions vs flat and random
  under independent unseen Minecraft seeds/action RNGs. Offline imitation
  accuracy is NOT proof of a competent goal-following policy.
- [ ] Autonomously select and terminate at least two options without a
  teacher-driven phase handoff, then start longer-horizon planning.


## New project-level goal: discover and compose skills toward Any% (DESIGN ONLY)

**Architecture contract:** [docs/BOOTSTRAP_TO_ANY_PERCENT.md](docs/BOOTSTRAP_TO_ANY_PERCENT.md).
This explicitly resolves how independently learned capabilities can eventually
compose without a scripted wood->stone->End route, privileged world knowledge,
or globally hand-tuned item rewards. Nothing in this section is implemented.

- [x] Document proposed shared goal-conditioned skills, learned initiation/
  termination, observed-effect model, high-level chooser, primitive fallback,
  reward separation, curriculum and complete Any% evaluation gates.
- [ ] Before many specialized resource trainings, design and version
  goal/option/outcome schema and player-known transition replay, with strict
  no-leak train/frozen parity and compatibility protection for V2-V8.
- [ ] Add full normal player interaction API (right-click/place, hotbar and
  inventory/crafting UI), full item/entity observations and sampled actions
  without privileged "craft" or "mine block" shortcuts.
- [ ] Reproduce FIRST robust natural resource->real inventory acquisition,
  including observed dropped-item pickup, across multiple held-out worlds
  and independently sampled action RNGs. CURRENT V7/V8 FAIL this gate.
- [ ] Learn two reusable skills and an independent high-level selector that
  picks/interrupts/recovers based on its own knowledge. Controlled practice
  scenarios allowed, but NO scripted handoff in scored tests; compare to
  flat policy, fixed sequence and random.
- [ ] Prototype replay/achieved-goal relabeling vs PPO, and only then an
  uncertain learned consequence model/planner when it wins a held-out
  data-efficiency and wall-time comparison.
- [ ] Instrument verified dragon defeat AND vanilla exit portal entry on
  ordinary survival starts; report win rate before speed, independent
  sealed seed population and game ticks vs wall clock.
- [ ] Develop upstream discovery/bootstrap of *rare late-game states* without
  sneaking known human game routes, scripts, recipes or oracle coordinates
  into training; make explicit how near-goal training starts are sourced.
  Do not expect terminal win reward alone to teach full Any% from scratch.
- [ ] Target: success on fresh natural seeds and eventually <=36,000 in-game
  ticks for successful runs; not guaranteed on literally every possible seed.


## LATEST: verified first genuine autonomous natural wood mined + picked

- [x] Inspect full `woodv3_01`: 24,000 six-tick decisions/10 noise worlds,
  338 ordinary blocks mined, 2,895 newly visible wood surfaces, but **zero
  wood mined/inventory**. Action persistence remains inadequate.
- [x] Versioned new four-head `wood_persistence_v4` neural motor: choose
  6/24/72 ticks to hold learned ordinary movement/look/attack (no scripted
  target, commands, handoff, waypoint or pretrained model). Same bounded
  player-visible memory train/frozen; SMDP game-tick discounting; game
  horizon measured in constant six-tick units across all policy variants.
- [x] Real from-scratch vanilla-noise seed `persistpilot1103`, 600 units,
  115 sampled policy decisions and 4 PPO updates: Minecraft `mined wood`
  at unit355 and `wood inventory increased 1` at unit361. True mined+
  picked event; **first autonomous natural inventory log in this project**.
- [x] Matched frozen V4 versus uniform-random V4 with identical action
  durations and 3,600 game ticks/seed on four truly unseen Minecraft
  seeds: frozen wood inventory **1/4** vs random **0/4**; ordinary mined
  blocks90 vs106. Too small to prove learning advantage. Old V3 frozen
  on the first two matched seeds mined 15 nonwood and collected 0 wood.
- [x] Safe start/resume on independent live V4 120->240 units across
  two different new seeds, saved CUDA/optimizer/RNG. Added optional
  `EVAL_SEED_START` to evaluate disjoint additional unseen seeds without
  replaying earlier ones. Original v2/v3 checkpoints remain compatible.
- [x] Preserve original successful 600-unit V4 pilot model as local ignored
  immutable-comparison copy `checkpoints/fresh_stream_holdv4_pilot600.pt`,
  SHA256 in detailed report. **134 pure tests pass.**
- [ ] NEXT user run: `tools/run-fresh-treadmill.sh resume persistpilot1103
  24000 600 wood_persistence_v4` in terminal. Keep exact original 600
  per-world horizon, trained optimizer and full train-seed cursor.
- [ ] After next long round, compare grown V4 to frozen historical pilot
  plus uniform-random *same-duration action* control on many disjoint
  held-out seeds. Track *actual inventory wood*, rate per seed, camera/
  action persistence, survival, time and learning collapse; avoid overclaim.
- [ ] Then add calibrated bounded dropped-item entity observations, neural
  item-use/inventory/crafting and eventually independent normal Any%.


## LATEST: speedrun01 policy-collapse audit and v3 learning experiment

- [x] Validate resumed `speedrun01`: 10 vanilla seeds, 24,000 committed
  decisions / 250 CUDA updates, 106 ordinary blocks broken, 759 unique
  wood surfaces seen, **zero** wood actually mined or collected; last two
  training seeds had zero block breaks.
- [x] Complete same-seed, identical-spawn, held-out Minecraft frozen
  controls, sampled v2 vs uniform random, 480 decisions each. V2 chose
  jump-forward452/480 and attack12/480, mined0; random mined2. Neither
  picked up wood. Treat as diagnostic, not proof of generalization.
- [x] Add observational action diagnostics for attack rates, crosshair wood,
  camera stability, crack feedback, attack streak and per-head actions;
  include in journal and pulse without changing the actor's input.
- [x] Add optional versioned `wood_progress_v3`: bounded, deduplicated
  own visible crack-stage gain for consecutively targeted wood, no hidden
  targets/scripted handoff; lower learning rate and higher entropy only for
  new v3. Existing v2 checkpoint/resume settings remain unchanged.
- [x] Run a real scratch-v3 96-decision/one-CUDA-update vanilla Minecraft
  smoke plus the full 130-test suite; checkpoint verified.
- [ ] NEXT: run `tools/run-fresh-treadmill.sh start woodv3_01 24000 2400
  wood_progress_v3` from a foreground terminal. Compare frozen v3 to
  original v2 and random on fresh same-spawn held-out seeds. Measure actual
  wood mining, inventory, target accuracy and sustained camera-still attacks.
- [ ] Investigate action persistence and visible dropped-item entities,
  without making phase switching scripted.


New optional scoped NVIDIA mitigation: fresh treadmill now launches
Minecraft JVM with `__GL_THREADED_OPTIMIZATIONS=0` by default, reversible
via `--nvidia-gl-threading default` for direct Python evaluation. Verified
12-decision real Java Minecraft smoke `nvidia_gl_safe_1103` completed
without a crash; not proof of fixing the user's prior NVIDIA SIGFPE.

## CRASH FIX complete; natural seed trainer resilience

- [x] Inspect `speedrun01`: committed checkpoint 12,000 decisions / 125
  neural updates / five full Minecraft seeds, zero natural inventory wood.
  Original sixth-world startup crashed `SIGFPE/FPE_INTDIV` in NVIDIA
  `libnvidia-glcore.so.595.104.02`. Firefox cause NOT established.
- [x] Find synchronous MCio bug: `ControllerSync.recv_observation()`
  ignores timeout and waits forever when Java dies, preserving a flock
  inaccessible to new bubblewrap PID namespace. NEVER delete held lock.
- [x] Add process-aware 30s native observation and 50s connection
  watchdog, bounded same-seed retries, automatic uncommitted rollback,
  GPU/JVM error logging, safe original-options restoration on resume.
- [x] Preflight actual file lock BEFORE starting Gamescope; give clear
  lock-owner message instead of creating a second stuck compositor.
- [x] Live `watchdog_smoke_1103`: first Java unexpectedly exited rc=0;
  detected, quarantined and retried SAME seed, then real12 decisions/1
  CUDA update and verified checkpoint. Original speedrun01 saved weights
  remain unchanged. 127 pure tests pass.
- [ ] Monitor actual resumed `speedrun01` through more world launches;
  capture driver error reports if NVIDIA SIGFPE reproduces. Differentiate
  `rc=0` early JVM exits from native driver faults. Consider an isolated
  graphics backend comparison before touching the user's host system.
- [ ] Primary skill goal remains genuine natural first wood in inventory;
  compare fresh unseen seeds against matched random baselines.

## ACTIVE: never-repeated random Minecraft levels and robust training

- [x] Replace fixed eight-world long-training rotation as PRIMARY roadmap:
  `experiments/fresh_seed_treadmill.py`, `tools/run-fresh-treadmill.sh`,
  `tools/run-fresh-evaluation.sh`. No manual seed cohort preparation.
  Every completed episode gets a new real vanilla `minecraft:noise`
  survival world; one shared from-scratch neural PPO policy is trained.
- [x] Full signed 64-bit seed bijection `fresh_seed_schedule.py`: up to
  2^63 unique train and 2^63 disjoint held-out eval seeds; deterministic
  checkpoint cursor; both output parities appear in both populations,
  and seed is never a policy feature. Unit tests verify disjointness.
- [x] Protect clean train/frozen comparison: newly generated sources set
  ONLY vanilla `spawnRadius=0` so the SAME seed produces SAME initial XYZ
  for separate policy and random-control runs. Old early spawnRadius10
  pilot comparisons were NOT strictly matched. Real held-out spawn
  `(48.5,64.0,32.5)` confirmed identical. No seed/layout data enters actor.
- [x] Implement model+optimizer+RNG atomic checkpoint only AFTER genuine
  Minecraft world exit; journal clearly marks precommit observations;
  fsynced in-flight marker, owned-world quarantine + deterministic retry
  after simulated incomplete-world crash; no orphaned experience credited.
- [x] Graceful one-shot stop DURING actual Minecraft rollout verified:
  8 decisions consumed, clean checkpoint and stop marker removed; resume
  continued model on NEXT unique seed. Multiple actual start/resume tests
  confirmed no training counter reset, source/seed NBT verified.
- [x] Bound disk use: only last two owned finished worlds retained by
  default; source generated by vanilla server removed after verified copy;
  incomplete/crashed worlds quarantined rather than overwritten or erased.
- [x] Matched held-out evaluation on real Minecraft with frozen weights:
  separate disjoint seed schedule and EXACT same initial spawn as random;
  sampled policy 1 nonwood break, random0 in 12 decisions; both0 wood.
  Only correctness smoke, NOT statistical advantage or full speedrun.
- [x] Measure startup overhead: vanilla source creation ~6.6-7.4s and
  full client startup ~9.4-14.3s/seed. Safe restart retained, default
  2,400 decisions/seed amortizes cost; no proof hot switching works yet.
- [ ] **NEXT:** train on many truly unique natural seeds, compare frozen
  sampled/greedy/untrained/random baselines on large heldout seeds,
  measure actual *inventory wood*, success rate, survival and effective
  throughput. Improve non-stationary exploration and RL sample efficiency.
- [ ] Prototype a persistent MCio client across world changes as a
  SEPARATE isolated test: reset global client/server sync barrier and
  semaphores, unload/dispose old world, reload new integrated server,
  reset packet/observation epochs, prove no deadlocks, stale data or
  wrong seed. DO NOT put an unverified hot switch in default trainer.
- [ ] Add player-perceivable dropped item entity memory, crafting and
  item-use controls; train to obtain FIRST actual natural inventory log,
  then later any-random-seed Minecraft speedrunning progression.

## ACTIVE: natural survival AI training, scalable continuous learning

- [x] User direction change: **normal unseen survival worlds first**, fixed oak training and manually scripted pickup only for brief diagnostics. Record honest natural inventory success instead of optimizing scripted-fixture accuracy.
- [x] Implement a new FROM-SCRATCH memory-first (no pretrained weights/RGB dependency) `NaturalPPO`, 50,913 params, three heads simultaneously selecting movement, look and attack every six Minecraft ticks; typed player-own+crosshair+current crack+top24 bounded 13x9 camera first-hit block surfaces. Schema `natural_memory_online_v1_first_hit_objects_ppo6`; old checkpoint families unaffected.
- [x] Natural survival world seed generator `tools/prepare-natural-training-worlds.py` + verified noise terrain, and per-world continuous evolving MCio saves. `--source-cohort` or explicit `--sources` training; no tree fixture, privileged world location, scripted phase switch, spawn item or teleport during play.
- [x] Long-run launcher `tools/run-long-natural.sh start|resume TAG COHORT TOTAL_DECISIONS`: actual start/resume run passed, per-world PPO updates, reproducible model+optimizer+RNG+visited surfaces+event counters checkpoint after clean world save, appended JSONL, rolling speed telemetry, process lock, `.stop` safe pause verified, no service/daemon installed. Nested Gamescope may exit139 **after** valid Python journal; launcher validates results.
- [x] Real natural `explore_v1` run/resume: 480 total decisions, 12 updates, 114 new cells, 5727 observed surfaces, 40 actually mined blocks, **no wood**. Existing checkpoint kept local.
- [x] New versioned `wood_discovery_v2` reward for first-time, physically camera-visible wood ID (no hidden map, no goal coordinates), plus cells, surfaces and real mined/picked rewards. From scratch across three natural worlds: **720 decisions/12 updates**, 136 cells, 7499 surfaces, **366 newly seen wood surfaces**, 21 nonwood blocks broken, **0 actual wood**. Resumed 360 -> 720 on same three worlds.
- [x] Frozen evaluation on completely held-out natural seeds: first `explore_v1` 120-decision greedy vs random gave cells11 vs20, mined blocks2 vs1, wood0 each. `wood_discovery_v2` 120-decision greedy vs sampled vs uniform random: **cells 2/24/28**, actual blocks **0/5/15**, wood0 in all. NOT proven better than random; greedy policy action collapse is actionable next weakness.
- [x] Live Minecraft showed valid transient `actively_destroying=true, crack_stage=-1`; revised Python adapter to represent **unknown** crack stage, not fabricated 0 or a hard crash; added pure test; subsequent actual 360+360 natural wood run no invalid readings.
- [ ] **NEXT:** improve the natural-world training learning curve and held-out generalization, especially degenerate greedy/stationary action selection, without returning to staged tree curriculum. Compare truly frozen sampled/greedy/untrained/random across more seed/time budgets, track actual wood pickup and performance/episode survival.
- [ ] Harden recovery from abrupt SIGKILL, world auto-save vs model checkpoint drift (clean `.stop`/world-save path already verified). Never misrepresent orphaned post-checkpoint journal events as committed experience.
- [ ] Add bounded player-perceivable *dropped-item entity* memory with actual camera/FOV/occlusion/size checks, and train one autonomous attack/collect policy; add missing interaction, crafting, tools and dimension progression in new versioned policy schemas.
- [ ] Drive real multiple-seed, sustained natural Minecraft first-log inventory success, then resource/crafting progression toward independent Any% completion; all scripted stages remain diagnostics only.

## Prior: real mined-drop inventory acquisition + memory observation work

- [x] Add opt-in `MiningFeedbackRequest(fresh=True)` on same synchronous
  action packet, gated by `GMAINCRAFT_MINING_FEEDBACK_ON_DEMAND_V1=1`;
  verify 50 fresh packets / 300 normal game ticks, no data on other 250,
  no lost frames and original frozen mine motor **12/12** preserved.
- [x] Same-build no-Torch 300 warm tick controls: status-only 0.7527s,
  per-tick crack stage0.7915s and on-demand0.8055s. Request frequency
  reduced sixfold but **no proven wall-time benefit**. Do not claim free
  sensing or another speed regression root cause without replication.
- [x] Compose frozen learned own-crack mining + independently trained RGB
  item pickup; real log must be mined before pickup begins, no spawned
  items or commands during scored play. Main held-out seed784232 12 cases:
  learned mine+learned p02 pickup **11/12 true inventory logs**,
  always-attack+learned11/12, learned+forward9/12, random7/12,
  no-movement0/12, no-attack0/12. Train/eval use the exact same memory
  motor encoder; scripted handoff is NOT learned. Read
  `docs/COMPOSED_MINE_PICKUP_AUDIT.md` for raw evidence and caveats.
- [x] Fix false per-episode mined/pickup stat deltas from cumulative MCio
  StatsUpdate after per-episode cache reset. Persist absolute counter
  tracker across cases; corrected 12-case logs verify one real event per
  successful inventory acquisition. Pilot event deltas INVALID, though
  its inventory observations were real.
- [x] Additional eval seed784233 comparison: frozen pickup p01 yielded
  7/12 learned-mine inventory success; p02 yielded9/12; forward9-10/12.
  No statistically established learned superiority over trivial controls.
- [x] Independently scratch-train the attack motor using on-demand feedback
  throughout 28 real Minecraft episodes:14/28 mining successes, first half
  2/14 vs second 12/14; frozen 12/12 on fresh cases, untrained0/12 and
  always-attack12/12. Older checkpoint and weights not overwritten.
- [x] Verify combined on-demand first-hit surface blocks AND own crack
  stage can share the SAME normal 6-tick action: 13/13 real Minecraft
  controls passed, including both absent on the next non-request packet;
  `surface_mining_joint_nov03d`. Original V5 status-only crosshair 4/4
  still passed with new research env flags OFF (`mining_req_compat_nov03d`).
- [ ] **NEXT:** add bounded **player-visible dropped-item entity state**
  with FOV/occlusion/render-distance/size checks and real validation;
  no hidden items or world/chunk/entity list sent to policy. Train and
  frozen game must use exactly same memory schema and action history.
- [ ] Train a new from-scratch agent to choose attack/pickup phase on its
  own, rather than scripted evaluator handoff, and test real held-out
  actual inventory logs with no controller commands during play.
- [ ] Investigate intermittent early on-demand training error (game active
  while crack stage violates accepted range); preserve failed JSONL,
  report raw values before relaxing validation. Do not silently invent
  crack progress or accept aborted run as training success.
- [ ] Eventually discover/approach/aim/mine/collect wood in truly new
  natural survival noise seeds with continuous ordinary vanilla actions.
  No Any% win or autonomous first log has been demonstrated yet.

## Earlier: actual vanilla mining progress + learned attack skill

- [x] Inspect Minecraft 1.21.3 client and add opt-in read-only
  `gmaincraft_mining_feedback_v1`: client's own `isDestroying` and visible
  `getDestroyStage` crack stage, plus vanilla crosshair-hit boolean. No
  hidden block data or world seed; fail-closed Python CBOR adapter.
- [x] Real disposable-world mining lab: stage 1->4, release reset, resumed
  1->9, actual mined-oak stat+1, later mouse-away inactive; 9/9 live checks.
  The item was NOT picked up. Full evidence:
  `docs/MINING_FEEDBACK_AND_ATTACK_LEARNING.md`.
- [x] Train a from-scratch **narrow attack-only motor** (no pretrained
  weights, no hidden target features) using real crack progress and real
  mined events; 3/14 training episodes broke a log in first half, 10/14
  in second. Save versioned checkpoint locally, never in Git.
- [x] Separate independent frozen 12-case actual Minecraft evaluation:
  learned 12/12, untrained 0/12, random 0/12, always-hold 12/12,
  no-attack 0/12. Learned motor matches rather than beats always-attack;
  fixture aims at reachable wood, pickup remains 0. Real game only.
- [x] Preserve earlier V5 status-only compatibility on the new patched
  mod: real Minecraft `miningcompat_nov03a` passed 4/4 with mining OFF.
- [x] Same-JAR no-Torch performance control with real 320x180 NVIDIA RGB:
  status only 0.7440s vs status+per-tick mining feedback 0.8109s for 300
  warmed game ticks. Zero frame gaps; one-run cost signal, not proof.
- [ ] Optimize mining sensor emission to an explicit fresh 6-tick decision
  request and rerun flat/natural performance and train/frozen parity.
- [ ] **NEXT:** composed learned attack + frozen learned item pickup on
  naturally mined drops, versus always-attack and always-forward controls;
  never summon an item to fake this integration result.
- [ ] Add **bounded** player-perceivable item-entity observation (camera
  FOV, render range, occlusion and screen-size audits); feed the SAME
  schema to any new from-scratch train and frozen test, no hidden loot/map.
- [ ] Eventually learn to find/approach/aim/mine/collect wood on unseen
  survival noise seeds with normal actions; then crafting and full Any%.

## Earlier: camera visibility experiment and natural-scene performance

- [x] Correct invalid lab synchronization: controller `/setblock`
  completion lags fast MCio action observations. Retire invalid first
  geometry logs; wait for Minecraft server updates, verify camera pose,
  unchanged-scene RGB and post-placement stability. No fixture coordinates
  feed the agent.
- [x] Independent 320x180 rendered-RGB vs MCio first-hit-ray test at 23
  settled block offsets/distances, with sensitivity slices and explicit
  crosshair/HUD masks. Coarse13x9 found16/23 target placements and13/14
  strong rendered controls; fine25x17 found18/23 and14/14 strong rendered
  controls. A 182-changed-pixel off-center block was missed by coarse and
  recovered by fine. This is NOT a universal human-visible threshold.
- [x] Check stone/glass/leaves/slit/darkness with real game images; opaque
  stone blocked target, glass/leaves were visible-but-missed cases.
- [x] Compare no-sensor/coarse/fine grouped sampling on three existing
  natural noise world seeds (31415927,271828183,161803398), all normal
  survival input with no world edits during measured samples; 0 action or
  frame gaps, 50 requested observations/300 ticks, real NVIDIA RGB.
  Scene-dependent warm times and exact repeat counts in
  `docs/VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`.
- [x] Fix camera-ray degeneracy at exact +/-90 pitch; a separate actual
  Minecraft test at both extremes passed 12/12 request/visibility checks,
  including off-center overhead oak.
- [x] Verify earlier V5 crosshair/status interface remains compatible
  with the final patched mod: a fresh real Minecraft positive/negative
  4/4 smoke `statuscompat_1103a` passed with surface flags OFF. This is
  NOT a rerun of learned V5 game success.
- [ ] NEXT: validate rendered projection and player-knowable thresholds at
  multiple FOVs, biome/weather/light, partial occlusion, transparent and
  noncollidable blocks, entities and moving natural scenes. Preserve
  first-hit/no-X-ray limits; do not give tiny or dark block IDs as oracles.
- [ ] After calibrating, use a versioned *bounded* current observation in
  new V6 learning BOTH train/frozen, and test held-out sustained mining,
  inventory pickup and action history. Keep 13x9 as current default, 25x17
  explicit research alternative. No Any% or natural first-log win yet.

## Earlier: tested grouped surface-memory research (not neural policy)

- [x] Build opt-in read-only MCio first-hit 13x9 camera ray sampling,
  bounded by the actual 4-chunk render setting in tests; record per-ray
  block ID, XYZ, face, distance and light without scanning chunks.
- [x] Group duplicate sampled rays by block, preserving ray indices,
  sample-screen occupancy and all visible first-hit faces. Verify **exact
  grouped/raw parity** in real MCio copied survival scenes. Six live
  scene/parity checks and 86 pure tests passed; off-center oak is detected,
  stone blocks hidden oak, glass conservatively blocks oak as a known
  visual false negative. Docs: `docs/SURFACE_SENSOR_V1.md`.
- [x] Preserve pristine, earlier V5 status and grid-only Fabric mod JARs
  through hash-checked backup; install only in disposable DemoInstance.
- [x] Quantify sensor overhead in same-location, matched idle 300-tick
  NVIDIA+RGB no-Torch experiments: baseline current JAR **0.6746s** warmed,
  raw grid previous JAR **0.9497s**, grouped current JAR **0.8898s**.
  Single-run comparisons, not statistical performance claims.
- [ ] **NEXT:** empirically calibrate sampling vs rendered/player-perceivable
  visibility (FOV/dynamic zoom, small distant blocks, partial occlusion,
  glass/leaves/fluid, light/night, entities, held-out natural worlds).
  Distinguish sample occupancy from pixel area, and detect false X-rays.
- [x] FIXED: avoid the measured ~32% per-tick grouped overhead: investigate an
  explicit request for a fresh surface observation on each 6-tick learner
  decision, with schema/age validity; re-benchmark matched real frames.
- [ ] Only once the detector is evidence-backed, wire a versioned **V6**
  learner to the SAME bounded surface state in both training/frozen normal
  gameplay, and test sustained normal mining/pickup against held-out seeds
  and memory-ablated/RGB-only controls. V5 log collection remains zero.

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

## ACTIVE NEXT: perceptible world objects + sustained learned action control

Latest live evidence: `docs/MCIO_PLAYER_SENSOR.md` (80 passing pure tests,
real MCio/Fabric source build and actual Java survival scene tests). User
approved all isolated disposable game setting experiments. More work than the
historical V4/RGB tasks below is complete:

- [x] Pin/build the exact upstream MCio 1.6.1 Fabric source and apply the
  tracked reversible **opt-in status/crosshair patch**, preserving original
  mod backup and only editing isolated DemoInstance. Real game positive
  oak hit, mouse-turn negative, stone occlusion, glass occlusion **all
  passed**. Controller fixture commands only outside scored gameplay.
- [x] Empirically measure one valid vanilla **crosshair actionability**
  boundary: oak block front faces at1.5/2.5/3.5 blocks detected, at
  4.5/5.5/6.5/7.5 not detected in this particular controlled raycast.
  Transparent glass targeted before oak. Do NOT declare these thresholds
  a general player-visible FOV/range detector.
- [x] Implement opt-in **V5 memory+status+RGB from-scratch learner** with
  stable full-string block-ID byte encoding, 40 numeric features, identical
  V5 observation in Minecraft train/frozen gameplay and schema-checked
  checkpoint; 4 train CUDA episodes +one update, 8 frozen cases, plus
  paired natural seed31415927 real run completed. All 0 log pickups;
  **this verifies integration, NOT learning competence**. Legacy V4 frozen
  evaluation remains compatible and was rerun with patched mod installed.
- [x] Isolate **separate conditional 60-ticks/s SDL Gamescope limit** by
  restoring the original mod for matched no-Torch tests: SDL ~5s/300 ticks,
  changing virtual Hz300 still ~5s; **headless Gamescope ~0.77s/300
  warmed**, same vanilla RGB/input correctness. Patched mod status ON vs
  OFF headless warmed 0.723/0.714s per 300 ticks. Keep visible SDL option
  for watching while using headless for high-throughput training. Earlier
  independently measured 4-chunk render-distance benefit still matters.
- [x] Instrument V5 playing-policy **crosshair hit distributions**. Frozen
  stationary case looked directly at oak 30/30 decisions but attacked 0
  ticks; near case saw oak 1/30 vs bedrock29; broad case missed 30/30.
  These are actual sensor readings, NOT hidden trainer target labels.
- [ ] Derive and test bounded **camera-perceivable** block/entity candidates
  (distinct from single actionable crosshair), calibrating FOV, projected
  size, line of sight, light, transparency, partial occlusion and distance
  on diverse real scenes/held-out natural seeds; measure false X-ray leaks,
  missing visible blocks and throughput. Do not bulk dump loaded chunks.
- [ ] Diagnose/experiment on **sustained vanilla ATTACK and camera control**
  from learned actions. Test different action durations/temporal memory and
  player-visible block-breaking feedback with actual held-out same-case
  negative controls before increasing training budgets. 0/8 frozen V5
  pickups and 0/4 V5 training successes are not good enough to scale.
- [ ] Demonstrate genuine **unrestricted find, mine AND collect** on unseen
  natural survival worlds (verified inventory), compare against fair random,
  prior RGB-only and matched V4/V5 ablations. No Any% achievement yet.

## Current design decision — structured Minecraft state, not RGB-only

- [x] Record the user's requested **direct game-engine/memory reads during
  both gameplay AND training** for information a skilled Minecraft player
  could know: own XYZ/yaw/pitch, inventory, status, stats/counters and
  eligibility-filtered observable blocks/entities. RGB can supplement this
  state, not replace it. Canonical spec: `docs/OBSERVATION_POLICY.md`.
- [x] Make visible/actionable block-data eligibility a **future empirical
  research task**: calibrate FOV, projected size, range, occlusion,
  transparency, lighting and vanilla interaction reach. Do NOT choose
  arbitrary cutoffs, feed hidden ore maps or claim a detector exists yet.
- [x] Implement the **first opt-in, versioned `player_knowledge_v1`
  packet adapter** for player XYZ/yaw/pitch, health, owned inventory slots,
  and available own progress stats. Pure and actual Minecraft tests passed.
- [x] Integrate identical memory observations in new from-scratch TRAINING
  and FROZEN normal survival evaluation: opt-in V4 `memory_factorized.py`,
  `joint_log_memory.py`, `memory_frozen_eval.py`, `natural_world_benchmark.py`.
  Ten tuned real CUDA train episodes/two PPO updates and 12 frozen cases
  completed; **0/10** and **0/12** wood acquisitions. Old RGB V2/V3 intact.
  See `docs/MEMORY_POLICY_V4.md`.
- [x] Implement **opt-in direct read-only targeted crosshair block** and
  own missing status in MCio/Fabric; V5 explicitly has only an actionable
  crosshair hit, not a general visible-block/entity candidate list.
- [ ] Add empirically filtered wider **player-visible** block/entity list
  with calibrated FOV/LOS/occlusion/lighting/projected area/reach and a
  bounded, versioned knowledge schema for both training and gameplay.

## Historical Priority 0 — MCio slowdown narrowed, render-distance mitigation measured

**Read `docs/NEXT_SESSION.md`, `docs/MINECRAFT_RENDER_PERF.md`, and
`docs/MEMORY_POLICY_V4.md` first.** The older no-Torch vs CUDA regression is
now substantially narrowed and the next research gate is world-object
memory sensors plus real unrestricted first-log acquisition; preserve
performance monitoring and matched display settings.

- [x] Create fresh-tagged 300-tick, no-Torch, sequence-checked real-Minecraft
  benchmark (`experiments/mcio_throughput_probe.py`). Four separate launches
  and 10 timed repeats demonstrated 0.62–0.72 s per 300 ticks after first
  300-tick warm-up; all acquired real oak-log inventory. Send ~0.008–0.01s,
  receive ~0.6–0.7s. See `docs/MCIO_THROUGHPUT_2026_11_02.md`.
- [x] Measure synchronous send vs receive in an instrumented vanilla game:
  >98% of game I/O wait is observation receive. First repeat 1.17–1.24s,
  later repeats 0.62–0.72s, exact accepted action sequences, no frame gaps.
  Temporary no-VSync/260-FPS control did not justify permanent changes.
- [x] Reproduce a comparable **natural-world no-neural** I/O slowdown
  (2.33 s for 180 random ticks on natural seed246813579) versus warmed flat
  world 0.6–0.7 s/300; show model inference is not cause. Isolate rendering
  impact with paired 600-tick natural/idle test: defaults 4.5349 s, uncapped
  4.2651 s, renderDistance8 3.3872 s, 6 2.59 s, **4 1.6945 s**.
  Details `docs/MINECRAFT_RENDER_PERF.md`. Historical 4s episodes may have
  additional transient conditions; exact old workload not reproduced A/B.
- [x] Introduce scoped, exact-backup-restored game display profiles and
  use fast 4-chunk vanilla option for V4 training/frozen evaluation; no
  permanent changes to the user's normal Minecraft or sandbox.
- [ ] Reproduce original slow controlled PPO workload under exact old host/
  game conditions if it returns and profile any residual bottleneck. No
  evidence yet to claim every historical slowdown was caused by render
  distance alone.
- [x] Trace first-100-tick warm-up and cumulative I/O by 100-tick segment.
  At defaults natural seed246813579, first100 ~15.99ms/tick vs last100
  ~5.51; with 4-chunk display, first100 ~6.16 vs last100 ~1.79.
  Actual internal Java render-vs-simulation split remains a future profiler
  task; don't guess a system driver or GPU fall-back.
- [x] Demonstrate an evidence-backed reversible repo-scoped game option
  lever rather than changing drivers/sandbox: 4-chunk render distance,
  VSync off, FPS260 improves 600 idle ticks **2.68x** on one natural seed;
  new fast V4 curriculum runs ~1.3–1.5ms/tick after warmup. Caveat: this
  limits distant visibility; test at 6/8 chunks for future long-horizon tasks.
- [x] Document, test, commit and push V4/performance work through isolated
  Gentoo Shell (this task is complete when the accompanying commit is
  confirmed on `origin/main`).
- [ ] **NEXT beyond V5 raycast + status:** investigate calibrated lists of
  genuinely perceptible block/entity objects and first-log learning with
  action history. Compare memory+RGB vs legacy RGB and memory-ablated new
  models under EXACT held-out cases, with true game inventory. Contaminated
  `v3event01` remains excluded. No manual speedrun/macros as the solution.

## Historical November 2 research snapshot — RGB perception and mining events

- [x] Inspect actual policy RGB screenshots; confirm command error overlays
  obscure much of 84x84 scene. Add dedicated MCio `chatScale:0.0` option fix;
  verify in real game it preserves camera/HUD. Hidden chatVisibility2 BREAKS
  yaw/pitch reset, do not use. No changes to user's normal Minecraft options.
- [x] Self-collect random action/RGB sequences using normal Minecraft inputs;
  13 train/4 validation same-arena episodes (507/136 action pairs), distinct
  RNG seeds; longer first collector run timed out after 13 valid files.
- [x] Train own-action inverse-dynamics CNN on CUDA from random weights:
  held-out camera **42.6%** vs majority23.5%, attack **84.6%** vs52.2%,
  movement **15.4%** vs14.7%. Camera-locked movement **38.5%** vs14.7%.
  Explicit temporal difference and GRU diagnosed but did NOT solve simultaneous
  movement/camera confusion. Models are diagnostics, not Minecraft skill.
- [x] Add optional `--vision-init-tag` loading ONLY CNN representation learned
  from user's own Minecraft RGB/actions, with random policy heads. No verified
  downstream game success from pretraining. No outside model weights.
- [x] Separate schema-tagged **12-game-tick V3** factorized decisions from 6
  ticks; preserve old checkpoint compatibility. `v3clean01` interrupted after
  15 recorded train episodes, saved valid checkpoint after 14; never claim 28.
- [x] Add Minecraft real mined/picked-up statistic reader on EACH MCio tick,
  private evaluator only (not AgentView).
- [x] **Invalidate `v3event01`** reward experiment: cumulative stat jumps
  were incorrectly treated as per-episode new mined blocks. Implement
  `CumulativeMinedTracker` scoped to live arena; real Minecraft negative
  control `statfix01` mine/no-attack/mine yields **1/0/1** new oak blocks.
- [x] Run corrected event reward experiment `v3eventfixed01` from clean
  `v3clean01`: 10 new CUDA train episodes, **3 truly mined blocks, zero
  collected inventory logs**. Paired frozen restricted audit: 4/4 mined both,
  0/4 collected clean checkpoint versus 2/4 for continued checkpoint. All
  unrestricted near **0/4**; no direct evidence event reward caused the gain.
- [x] Load V3 in genuine fixed-spawn noise Minecraft survival world seed
  31415927: 0 natural wood within 240 ticks for random and frozen V3,
  exactly paired spawn, no observed transfer.
- [x] Narrow historical MCio throughput concern: natural-world chunk/render
  workload plus cold-start penalties explain measured current slowdown; a
  reversible 4-chunk profile improves active step I/O 2.68x in controlled
  natural-world test. Exact original old host/state regression not causally
  proven. See `docs/MINECRAFT_RENDER_PERF.md`.
- [ ] **NEXT research gate:** make full-movement and camera action effects
  separable with history/object-centric perception; evaluate online rather
  than trusting offline classifier accuracy. Try replay-aware/sample-efficient
  algorithms only with valid held-out tests, not stale PPO optimization.
- [ ] Reach reliable **unrestricted** real mined-log + inventory pickup on
  diverse controlled scenes and genuinely unseen natural worlds, then one
  uninterrupted wood-to-crafting episode. Any% still unachieved.

See `docs/VISUAL_SELF_SUPERVISION.md` and `docs/FACTOR_POLICY_REVIEW.md`.
All local world saves/checkpoints/datasets/frames are Git-ignored.

## Historical November 2 research snapshot — V2 learning-efficiency gate

- [x] Replace V1's dominating 80-tick attack-only macro with **3 simultaneous
  6-tick movement/look/attack channels** in a separate 888,750-parameter CNN
  and factorized PPO implementation; preserve old checkpoints.
- [x] Verify actual vanilla input continuity: `continuous_mine_probe`
  **mined and collected one real log** after 20 successive six-tick attack
  selections. This is an instrumentation control, NOT learned success.
- [x] Fix stale-inventory reset race after successful log acquisition:
  wait for correct pose AND empty wood inventory plus two settled steps;
  abort rather than contaminate positive results.
- [x] Matched baseline audit `controlv2b`, near-seed3737 N=12:
  **attack-then-forward 8/12**, forward-attack 3/12, random 0/12,
  geometry-based oracle 11/12 (privileged, NOT fair learned policy).
- [x] Frozen 12-case V2 comparison under the SAME scenarios:
  `factorv2pilot01` 0/12, `attackrl01` 0/12, `aimnear01` 0/12.
  Common observed failures: backtracking, continuous turning, or attacking
  without maintaining crosshair. These models are NOT proven competent.
- [x] Narrow real-game attack-only curriculum `attackrl01`: **3/32** train
  and **2/4** held-out restricted stationary success. All unrestricted
  near/broad tests 0. Labels must distinguish restricted from real play.
- [x] Test aim+attack staged warm-start and mixed/interleaved curriculum:
  `aimnear01` aim train 1/36, unrestricted train 0/20, frozen full 0/4;
  `mixstable01` 0/48 train combined, frozen full 0/4 near and broad.
  Interleaving and a sustained-attack teacher reward did not solve it.
- [x] Plug frozen V2 into truly seeded Minecraft evaluation; seed 246813579,
  fixed identical starting pose in three modes (random/forward/V2), all 0
  natural logs in 360 game ticks. This is a transfer **test**, not success.
- [x] Opt-in local agent RGB/action/GAE trajectory recording before PPO
  update. Real CUDA pilot `recordpilot01` produced/read 40 transitions,
  240 game ticks, compressed 135670 bytes, ignored in Git.
- [ ] **Research pivot:** diagnose whether the two-frame CNN actually
  extracts reusable target/action cues and learns camera stability; try
  memory-capable and sample-efficient representation/off-policy methods
  against identical held-out baselines. Do not continue blindly training
  current V2 PPO for thousands of episodes without measured gains.
- [ ] Establish REAL, unrestricted first-log learning on controlled diverse
  target positions, then **unseen natural world seeds**, then a continuous
  wood-to-crafting trajectory. **Speedrunning Any% remains distant.**

See `docs/FACTOR_POLICY_REVIEW.md` for the exact score ledger and runnable
commands. Local runs/checkpoints/worlds stay protected and ignored.

## CUDA confirmed and real Minecraft GPU pilot (after bridge restart)

- [x] Host UVM character devices now available in the sandbox after bridge
  restart. `cuInit(0)=0`, PyTorch GPU tensor allocation **PASS** on RTX 4070
  SUPER; no host filesystem sandbox expansion.
- [x] Compare CUDA and CPU CNN update microbench: batch 64, 25 steps,
  **879.69 CUDA vs 128.99 CPU updates/s** (~6.82x GPU) on synthetic data.
  This is not Minecraft environment speed.
- [x] Run `cuda01`: **12 real Minecraft training episodes and 3 actual CUDA
  PPO optimizer updates**, followed by 4 frozen held-out episodes, each
  220 simulation ticks; **0/12 train and 0/4 eval logs**, checkpoint saved.
- [ ] Investigate fixed 80-tick mining-action overuse (frozen cuda01 policy
  exhausted 220 ticks in only 3 decisions for all 4 cases). Improve the
  action-duration / temporal exploration design BEFORE large GPU training.
- [ ] Evaluate a corrected model against always-forward and randomized
  natural-world baselines using genuinely unseen seeded survival worlds.

## Continuity / checkpoint reliability

- [x] Make joint PPO checkpoint writes atomic and include optimizer state,
  metadata and accumulated updates.
- [x] Add `--init-model` warm start under new run tag; require fresh
  `--train-seed` and `--policy-seed`. Validated one 100-tick real-game CPU
  warm-start `warmcheck01`, 23 transitions, updated and saved checkpoint;
  **0 logs**, repeated old scenario seed in this smoke only. No evidence
  of generalization.
- [ ] Implement fully resumable multi-day online training including trainer
  RNG, active episodes/world reset state, optimizer and trajectory batch;
  current warm-start starts a *new* research run, not exact episode resume.

## Historical November 2 research snapshot — joint visual learner / GPU

- [x] Prove nested GLX hardware renderer: `NVIDIA GeForce RTX 4070 SUPER`,
  NVIDIA OpenGL 4.6 driver 595.104.02 (NOT llvmpipe). Do not mistake
  Minecraft's CPU world simulation for CPU rasterization.
- [x] Install optional PyTorch 2.15.0+cu132 **in project `.venv` only**;
  add `requirements-gpu.txt`, CUDA preflight and CPU-vs-CUDA microbenchmark.
- [x] **CUDA DEVICE BLOCKER RESOLVED:** host UVM nodes present after bridge
  restart, cuInit(0)=0 and actual GPU model training successful in `cuda01`.
  Device nodes might require recreation after host reboot.
- [x] Implement learned pixel-CNN actor-critic with joint movement/look/mine
  action selection, timed (semi-Markov) trajectory GAE, clipped PPO, losses,
  checkpointing, configurable CPU/CUDA and 320x180 default game rendering.
- [x] Implement real variable-player/variable-log vanilla training arena,
  privileged scalar shaping **only for training**, and independent frozen
  evaluation with no hidden world state passed to model.
- [x] Real Minecraft `jointsmoke01` CPU train 2 episodes + frozen eval 2,
  each 140 ticks; PPO update, local checkpoint, results **0/2 train and
  0/2 eval**, as expected for a smoke test. No proof of skill acquired.
- [x] Frozen `jointsmoke01` loaded in natural seed `161803398`, 120 game
  ticks, **0 logs**; actual vanilla inputs and no-cheat pipeline verified.
- [ ] **Next:** make CUDA available, compare CPU/GPU measured training
  throughput, then train sufficient independent-seed joint models and evaluate
  whether first-log retrieval improves over controls. Extend the same run to
  wood-to-crafting *only after* validated natural-world transfer.
- [ ] Replace fixed 80-tick mining primitive with chosen action durations;
  add memory beyond two frames and realistic exploration/inventory/crafting.

See [docs/JOINT_LEARNING.md](docs/JOINT_LEARNING.md) and
[docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md). Existing checkpoints,
flat-arena source, real-world source saves and prior results remain preserved.

## Historical November 2 research snapshot — old course correction

- [x] Audit existing learning milestones: useful instrumented prototypes,
  **not** proven general speedrunning or a scalable final RL architecture.
- [x] Add fair `AgentView` RGB-only and bounded Minecraft normal-input `AgentAction`
  adapter; protect evaluators' privileged state from policy inputs.
- [x] Generate a REAL Minecraft 1.21.3 noise world and smoke-test four frozen
  diagnostics; pilot seed 314159265, 0/100-tick inventory logs in all modes.
  **Not a paired benchmark:** default spawn-radius caused offset starts.
- [x] Implement fresh-seed natural-world benchmark prep with worldgen seed and
  noise verification, fixed spawn radius 0 and refusal to overwrite source,
  copy or logs. Tests pass; **new fixed-spawn in-game comparison still needed**.
- [x] Verify in **actual Minecraft** that freshly generated fixed-spawn
  copies match: seed 271828183, idle/forward-attack both (0.5,77.0,0.5),
  yaw=0 pitch=0, 25 ticks each, 0 logs. Local JSONL includes observed start.
- [x] Add automatic paired-mode initial-pose mismatch failure to natural
  benchmark (seed group). Smoke-tested matching source for two modes.
- [ ] Run longer natural first-log comparisons across genuinely untouched
  world seeds and all four diagnostic modes.
- [ ] Short variable-trunk + variable-player-start study on real vanilla
  blocks, compare trained frozen policies against fixed forward/random.
- [ ] Collect full trajectories and comparable valid/held-out-world data,
  then implement a small joint temporal RGB policy and a proper multi-step
  reinforcement-learning algorithm. No claim that PPO or Dreamer is solved.
- [ ] Make autonomous FIRST LOG acquisition reliable across distinct unseen
  natural Minecraft seeds, then expand the *same uninterrupted episode* to
  crafting, rather than endlessly stacking toy-only skills.

See [docs/COURSE_CORRECTION.md](docs/COURSE_CORRECTION.md). All previous
verified experiments/checkpoints stay intact. Benchmarks are diagnostic, not
an agent that can already beat Minecraft.

## Historical subskills handoff (superseded by the joint learner and GPU priorities above)

- [x] Minecraft camera-relative numeric navigation: 3 independently initialized/trained policies each achieved 50/50 held-out + 50/50 extra yaw/target scenarios (2026-11-02). Privileged goal direction; not visual.
- [x] **Camera is functional, not cosmetic**: no human-likeness, gaze-stability or head-level rewards. Must still obey Minecraft view/raycast/reach for actual interaction.
- [x] RGB-based 5-action camera+ATTACK policy in real Minecraft, 2 independent 500-episode runs. Fresh drop-cleaned 20-case audit: 0/20 untrained, 15/20 `rgb01`, 18/20 `rgb02`, real mined log in inventory. Hardcoded oak-brown mask, scalar privileged angular shaping, initially scripted W pickup. Invalid initial uncleaned 12/14 results superseded by audit.
- [x] Add **4-key W/A/S/D collection policy** from 14 RGB-only handcrafted pixel features (no player/item coords in policy input); 2 independent 150-episode real Minecraft RL runs `p01`, `p02`, trained with controller-summoned **dropped** log and scalar item-distance shaping. Actual inventory success, no scripted movement in policy modes.
- [x] Fresh drop-fixture audit in `MDL_DropAudit_dclean02` with zero item initial motion and clean resets (seed 8804; **32 cases/model**): no-action 0; untrained 2/32; random 16/32; always-W 21/32; trained `p01` **30/32**, `p02` **24/32**. Earlier `dclean01` audit was aborted by a conservative distance guard (item drift); preserve as invalid partial evidence.
- [x] **Integrate both saved neural policies** in `log_mine_and_pickup_audit.py`: camera-look+ATTACK selected by `rgb02`; afterward W/A/S/D selected by frozen pickup policy, **no scripted forward walk**. Two fresh copied-world camera-angle audits (`f01` seed 8816, 24 cases; `f02` seed 8817, 32 cases): fixed-W 23/24 + 26/32 = **49/56**; untrained pickup 8/24 + 13/32 = **21/56**; `p01` 21/24 + 22/32 = **43/56**; `p02` **22/24 + 29/32 = 51/56**. No-attack control 0 in f02; all positive results require actual mined-log inventory acquisition. `p02` versus W is a *small*, unproven advantage; both mostly use W in centered setup.
- [x] Reset hygiene: controller `kill @e[type=minecraft:item]`, clear inventory, fresh template copies for audits, no-attack/no-movement controls. The fixture now uses `Motion:[0.0d,0.0d,0.0d]` to avoid suspect distance-from-spawn audit failures. Original `DemoWorld` and all previously trained models preserved, local ignored CSVs/weights saved.
- [x] Implement variable actual trunk/player starting positions in the new joint learn/look/move/mine curriculum, with RGB policy observations and held-out scenario RNG. CPU real-Minecraft **2-train/2-eval smoke only, 0 successes**. [ ] Prove improvement against appropriate frozen baselines and genuinely natural-world transfer on unseen seeds.
- [ ] Replace the two hand-engineered color filters with robust **learned RGB representations** and evaluate lighting, wood types/biomes, obstacles and genuine random Minecraft world seeds. Reduce/remove privileged angular/distance shaping as tasks mature.
- [ ] Go from two **sequential separately trained** policies to a combined learned controller that decides when to look, move, attack, and collect, with nontrivial survival/block interaction objectives.
- [ ] Investigate Gamescope shutdown SIGSEGV 139 only if it prevents future experiments; the Python game tests and checkpoints completed before compositor crashed.
- [ ] User physical-hover check of host pointer suppression is still pending; SDL2 synthetic test and MCio controls passed.

## Verified foundations

- [x] Install and launch isolated Minecraft Java 1.21.3 Fabric + MCio controller.
- [x] Work around GLFW viewer visibility using XWayland; resolve CBOR decoder incompatibility by pinning `cbor2<6`.
- [x] Random-action player control, observations, and fully restarted disposable-world episodes.
- [x] Windowed and headless throughput measurements; inspect total runtime, not just active step time.
- [x] One-process soft-reset benchmark: 5 × 1,000 steps, <0.12s player resets; document that the world is **not restored**.
- [x] Item novelty reward detector: first oak log +1, second +0, stone pickaxe +1, reset clears score.
- [x] Random-weights policy-gradient navigation in real Minecraft (120 episodes): baseline 7/20, trained 20/20 on held-out targets, fixed yaw.
- [x] Establish GitHub source-of-truth layout, `AGENTS.md`, `TODO.md`, handoff/docs, unit-test and CI scaffolding.

## Next research/engineering work (prioritized after successful orientation test)

- [x] Validate camera-relative training across three independent **numerical policy/training/evaluation RNG configurations** and disjoint target/yaw draws (see `docs/EXPERIMENTS.md`). Independent Minecraft **world** seed generalization and broader variance remain untested.
- [x] Learn camera yaw/pitch aiming and choose ATTACK from RGB-only engineered visual features in a fixed log arena (15/20,18/20 audited pickups; two independent trainings). Keep learning camera as a functional input, not a human-likeness objective. [ ] Extend to moving targets, agent-controlled pickup and less-engineered visual perception.
- [ ] Replace privileged numerical target-direction observations with visually detectable objectives; log exactly what information the policy receives. Keep structured *human-equivalent* HUD/inventory signals as an optional experiment, not hidden map data.
- [x] Learn a standalone RGB-feature W/A/S/D pickup policy with real inventory rewards and evaluate it in a fresh world (30/32 p01, 24/32 p02 vs fixed W 21/32). [ ] Extend to randomized natural tree locations, joint perception/actions, block placement, hotbar, inventory/crafting UI. Controller `/setblock` is permitted only during curriculum reset, never as an agent action.
- [ ] Separate environment adapter, observation encoder, action mapper, reward tracker, trainer, evaluator and experiment recorder into tested reusable modules under `src/gmaincraft/`. Keep the existing experiments runnable while refactoring.
- [ ] Implement reliable **full-world snapshot/restore** or equivalent independent episode creation without repeated full JVM startup. Audit mobs, inventory, time, weather, terrain, RNG, entities, player, and achievements.
- [ ] Capture experiment configuration, version metadata, train/validation/test seeds, event counts, rewards, failure modes, throughput, checkpoints, and optional video snapshots. Never accidentally commit private/local world artifacts.
- [ ] Add an on-demand visual monitor/timelapse without burdening headless throughput.

## Long-term curriculum (not implemented)

- [ ] Learn movement, looking, jumping, breaking, picking up items and basic survival under fair observations.
- [ ] Learn resource acquisition and crafting (wood -> tools -> iron).
- [ ] Learn exploration, caves, food, hazards, combat and inventory decisions.
- [ ] Learn Nether entry, fortress/blaze resources, Ender pearls/eyes, stronghold search and End portal access.
- [ ] Learn End fight, dragon kill, exit portal and complete Any% termination detection.
- [ ] Investigate scalable visual/world-model RL (e.g., DreamerV3 or alternatives), hierarchical skill reuse, GPU budget and sample efficiency; no promise that this will solve Minecraft within days or weeks.
- [ ] Evaluate honestly on **unseen random seeds**, with success rate and in-game time; compare to scripted/random baselines while labeling them clearly.

## Maintenance rules

1. Check off only verified accomplishments; mark pending or blocked tests explicitly.
2. On every meaningful code/docs commit, revisit this file and update the handoff status.
3. Move major completed milestones to a concise history; do not let a stale checklist masquerade as live status.
4. Record *actual* experimental results in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), not merely in chat.
5. Future chats should begin by reading this and [docs/HANDOFF.md](docs/HANDOFF.md).

## Follow-up: on-demand sample scheduling verified

- [x] Synchronized last-action-tick requests produced 50 fresh block views
  across 300 real game ticks, without stale samples on other packets.
- [x] Eight real bridge protocol controls passed; 87 Python tests passed.
- [ ] Next: natural-scene/visual calibration and learned pickup control.
