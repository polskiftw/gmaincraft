# TODO — gmaincraft

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

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

## Latest work — joint visual learner / GPU (2026-11-02)

- [x] Prove nested GLX hardware renderer: `NVIDIA GeForce RTX 4070 SUPER`,
  NVIDIA OpenGL 4.6 driver 595.104.02 (NOT llvmpipe). Do not mistake
  Minecraft's CPU world simulation for CPU rasterization.
- [x] Install optional PyTorch 2.15.0+cu132 **in project `.venv` only**;
  add `requirements-gpu.txt`, CUDA preflight and CPU-vs-CUDA microbenchmark.
- [ ] **CUDA DEVICE BLOCKER**: host `nvidia_uvm` loaded, but sandbox
  `/dev/nvidia-uvm` missing; `cuInit(0)=999`. User must create host UVM nodes
  and reboot bridge, then verify GPU tensor allocations. Launcher already
  binds UVM devices when present. Do not report CUDA success yet.
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

## Highest priority — speedrunner course correction (2026-11-02)

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
