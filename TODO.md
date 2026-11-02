# TODO — gmaincraft

**Last curated:** 2026-11-02. This is the live source of truth for project status. **All agents must read and update this file alongside substantive changes** (see [AGENTS.md](AGENTS.md)). Never mark a task done from source code alone; require actual run evidence where appropriate.

## Current handoff: visual log acquisition (camera movement only when useful)

- [x] Real Minecraft reference (2026-11-02): fixed old yaw 50/50, old random yaw 24/50; new policy 9/50 -> 50/50 plus 50/50 extra unseen target/yaw cases after 300 training episodes (30.16s learning, 299/300 training wins).
- [x] Add independently configurable seeds and write-once `--run-tag` to `experiments/orientation_learner.py`, keeping default behavior/filenames; 5 new unit tests (10 total).
- [x] Independent numeric RNG replication: `rep_b` and `rep_c` started 7/50 each, ended **50/50** on held-out and **50/50** extra unseen target/yaw sets after 300 real Minecraft training episodes each (26.67s and 25.17s learning, 299/300 training wins each). All three use the same type of saved-world flat arena.
- [x] Validate each independent run's 300-record CSV and finite (4,3) checkpoint; preserve local files under git-ignored `logs/` and `checkpoints/`.
- [x] **Verify normal Minecraft camera/mouse, mining and item pickup (2026-11-02):** in a fresh copied disposable world, MCio mouse deltas (+120,+50 pixels) rotated yaw +18.00 and pitch +7.50 degrees, reverse deltas restored both; vanilla left-click mining followed by W movement yielded 1 observed oak log (20.15s elapsed for launch/probe); no /give during the run. The actions were scripted *diagnostic* inputs; the log was controller-placed before gameplay, so NOT learned visual acquisition.
- [x] Record camera policy: **no reward for human-looking head position or motion**. Turning/looking is just an available input for observation and necessary raycast interactions. Efficiency and legal Minecraft constraints determine success.
- [ ] **NEXT: a learned visual wood-log curriculum**, starting from camera-targeting using rendered RGB frames, player-controlled yaw/pitch and ordinary attack/pickup. Begin with an isolated wood block; score true breaks/inventory acquisitions without exposing target coordinates to the policy. Report any engineered image preprocessing, privileged scalar training reward and failure modes.
- [ ] Extend from the log curriculum to traversal, real trees, general visual perception, resource collection and independent Minecraft world seeds. Keep previous successful movement models safe.
- [ ] Afterwards replace privileged numeric target direction with visually detectable objectives and a documented human-equivalent observation contract.
- [x] Add local `tools/run-mcio-viewer.sh` and SDL2 mouse-input filter: the user confirmed the original pointer lock was removed but physical hover still moved Minecraft's camera; the revised shim filters Gamescope's `SDL_WaitEvent` mouse motion, clicks and scrolling. Synthetic SDL2 regression passed and Minecraft MCio completed 10/10 exchanges with the new shim.
- [ ] **User verification pending:** physically hover/move/click over the game preview and confirm the camera no longer responds to the desktop mouse. The shell smoke tests did not inject physical mouse motion.
- [ ] Gamescope still returns SIGSEGV 139 after the child exits; investigate if it affects experiment stability or loses results. All three Python experiments completed and were validated.

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
- [ ] Learn camera yaw/pitch, break/collect via normal inputs and visual target selection; no human-like camera-motion objective. The real MCio camera and vanilla mining controls are independently verified, but the learned visual policy is not yet. Test randomized yaw after spawn.
- [ ] Replace privileged numerical target-direction observations with visually detectable objectives; log exactly what information the policy receives. Keep structured *human-equivalent* HUD/inventory signals as an optional experiment, not hidden map data.
- [ ] Progress from **verified scripted vanilla log break/pickup** to **learned visual targeting and log acquisition**, then place blocks, change hotbar, use inventory/crafting UI. Connect the existing first-seen-item reward detector to agent-caused events. Do not count controller `/give` or `/setblock` as a learned acquisition.
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
