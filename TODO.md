# TODO — gmaincraft

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

## Current handoff: learned wood aiming achieved; next learn movement/pickup and robust perception

- [x] Three independent camera-relative **numerical-goal navigation** models achieved 50/50 held-out + 50/50 extra targets each in a controlled arena (2026-11-02).
- [x] User camera policy: **do not reward human-like head placement or movement**. Camera direction is an input/sensor; legal Minecraft line-of-sight/raycast/reach still applies for interaction.
- [x] Scripted vanilla log-input probe: +120/+50 cursor pixels produced +18.00° yaw/+7.50° pitch; 100 ticks holding attack + 11 W ticks placed a genuinely mined oak log in observed inventory (no `/give`). Controller placed trunk during setup; NOT a learned experiment.
- [x] Build RGB-only feature encoder (12 engineered oak-brown spatial features), random-initialized five-action linear REINFORCE policy (look left/right/up/down/ATTACK), a 500-episode real Minecraft trainer, and a separate frozen evaluator. Trainer receives controller-side scalar angular-progress rewards; **no player pose or target coordinates in policy features**.
- [x] **Audited real-game evaluation** in fresh `MDL_LogAimAudit_clean01` copied world, novel camera seed 8815, 20 cases per model: no-attack control **0**, untrained policy **0/20**, trained `rgb01` **15/20**, independently initialized/trained `rgb02` **18/20**. These are real observed inventory pickups after agent-chosen camera/attack; moving toward dropped items still scripted. Each trained 500 episodes.
- [x] Detect and correct **stale item-drop contamination** in initial soft-reset tests: kill all item entities during controller reset, validate no-attack negative control, reject implausibly distant successful attack angles. Earlier 12/14 and untrained 11/14 numbers are **not valid evidence**. Audit file: ignored `logs/log_aim_audit_clean01.csv` (60 rows).
- [ ] **NEXT: choose movement and collection actions with the policy**, instead of fixed post-attack W steps; report successful inventory acquisitions and failure cases. Keep world reset contamination guards.
- [ ] Replace oak-color handcrafted mask with a learned visual encoder, vary trunk position/biome/light and camera ranges; remove the privileged angular-progress reward and advance to real trees without hidden target coordinates.
- [ ] Validate whole policy across independent **Minecraft world seeds** later; current model learns one 3-block target in the same flat arena and cannot speedrun.
- [ ] Investigate Gamescope shutdown SIGSEGV 139 only as needed; Python experiments and checkpoints completed first.
- [ ] User physical-hover check of host mouse suppression remains pending; SDL2 synthetic and real MCio control tests passed.

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
- [ ] **Learn movement and pickup decisions** to complement already-verified learned visual aiming/ATTACK. Then generalize to naturally generated trees, block placement, hotbar, inventory/crafting UI, with actual agent-caused inventory event rewards. Controller `/setblock` is permitted ONLY for training arena setup, never policy action.
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
