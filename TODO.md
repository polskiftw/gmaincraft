# TODO — gmaincraft

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

## Current handoff: multi-seed reproducibility, then camera control

- [x] Author and syntax-test `experiments/orientation_learner.py` using camera-relative numeric goal vectors and randomized starting yaw.
- [x] **Run in actual Minecraft (2026-11-02):** original fixed 50/50 vs random yaw 24/50; new random-initialized policy 9/50 before, **50/50 after** 300 training episodes, **50/50 on extra unseen cases**. Training CSV 299/300 successes; learner time 30.16s.
- [x] Verify saved local `orientation_policy.npz` and 300-row `orientation_training.csv`; capture caveats in `docs/EXPERIMENTS.md` and handoff.
- [ ] **NEXT: reproducibility:** parameterize policy/training/evaluation RNG seeds and use distinct output filenames to avoid overwriting the successful checkpoint. Run at least 3 independently initialized trainings on disjoint target/yaw scenarios; report variance, seeds, confidence bounds, and wall-clock costs.
- [ ] Next after independent replication: learn camera-turning input and movement jointly, then remove privileged numerical goal-direction observations in favor of perceivable targets.
- [ ] Investigate Gamescope exit-139 segfault only if it causes lost measurements, leftover processes, or prevents reliable experiments. Do not conflate the wrapper crash with Minecraft training failure.

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

- [ ] Validate orientation/navigation gains across multiple **independent** training RNG seeds and disjoint evaluation scenarios; report success rate, confidence intervals/variance, steps, and compute cost (see NEXT above).
- [ ] Add a camera-turning action and teach the agent to navigate using its own control of yaw/pitch, without the controller freezing orientation. Test randomized camera yaw after spawn.
- [ ] Replace privileged numerical target-direction observations with visually detectable objectives; log exactly what information the policy receives. Keep structured *human-equivalent* HUD/inventory signals as an optional experiment, not hidden map data.
- [ ] Develop real object interaction: look at and break a block, acquire a dropped item naturally, switch hotbar slots, place blocks, and open/use inventory/crafting UI. Connect the existing first-seen-item reward detector to agent-caused events.
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
