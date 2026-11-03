# TODO — gmaincraft

## NEWEST: camera visibility experiment and natural-scene performance

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
