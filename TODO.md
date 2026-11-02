# TODO — gmaincraft

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

## Current handoff: two learned camera/attack + pickup subskills verified; next variable tree positions

- [x] Minecraft camera-relative numeric navigation: 3 independently initialized/trained policies each achieved 50/50 held-out + 50/50 extra yaw/target scenarios (2026-11-02). Privileged goal direction; not visual.
- [x] **Camera is functional, not cosmetic**: no human-likeness, gaze-stability or head-level rewards. Must still obey Minecraft view/raycast/reach for actual interaction.
- [x] RGB-based 5-action camera+ATTACK policy in real Minecraft, 2 independent 500-episode runs. Fresh drop-cleaned 20-case audit: 0/20 untrained, 15/20 `rgb01`, 18/20 `rgb02`, real mined log in inventory. Hardcoded oak-brown mask, scalar privileged angular shaping, initially scripted W pickup. Invalid initial uncleaned 12/14 results superseded by audit.
- [x] Add **4-key W/A/S/D collection policy** from 14 RGB-only handcrafted pixel features (no player/item coords in policy input); 2 independent 150-episode real Minecraft RL runs `p01`, `p02`, trained with controller-summoned **dropped** log and scalar item-distance shaping. Actual inventory success, no scripted movement in policy modes.
- [x] Fresh drop-fixture audit in `MDL_DropAudit_dclean02` with zero item initial motion and clean resets (seed 8804; **32 cases/model**): no-action 0; untrained 2/32; random 16/32; always-W 21/32; trained `p01` **30/32**, `p02` **24/32**. Earlier `dclean01` audit was aborted by a conservative distance guard (item drift); preserve as invalid partial evidence.
- [x] **Integrate both saved neural policies** in `log_mine_and_pickup_audit.py`: camera-look+ATTACK selected by `rgb02`; afterward W/A/S/D selected by frozen pickup policy, **no scripted forward walk**. Two fresh copied-world camera-angle audits (`f01` seed 8816, 24 cases; `f02` seed 8817, 32 cases): fixed-W 23/24 + 26/32 = **49/56**; untrained pickup 8/24 + 13/32 = **21/56**; `p01` 21/24 + 22/32 = **43/56**; `p02` **22/24 + 29/32 = 51/56**. No-attack control 0 in f02; all positive results require actual mined-log inventory acquisition. `p02` versus W is a *small*, unproven advantage; both mostly use W in centered setup.
- [x] Reset hygiene: controller `kill @e[type=minecraft:item]`, clear inventory, fresh template copies for audits, no-attack/no-movement controls. The fixture now uses `Motion:[0.0d,0.0d,0.0d]` to avoid suspect distance-from-spawn audit failures. Original `DemoWorld` and all previously trained models preserved, local ignored CSVs/weights saved.
- [ ] **NEXT: randomize actual trunk location and starting player position** in the combined learn/look/move/mine/collect curriculum, so simply holding W isn't near-optimal. Use actual pixel observations rather than hidden coordinates as policy input; compare held-out seeded scenes to always-W and scripted baselines.
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
