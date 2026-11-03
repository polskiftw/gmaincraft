# START HERE — next ChatGPT/Gentoo Shell session

**Priority order is intentional.** Updated November 2, 2026. This document is the explicit continuation contract for the next conversation, not a claim that any unrun experiment has succeeded.

## FIRST: diagnose and restore MCio simulation throughput

**Do this before further reinforcement-learning architecture work, extended GPU training, or crafting/speedrun subtasks.** The present bottleneck is a ~5x-6x regression in real Minecraft game-step throughput, making research expensive. It is **not yet explained or fixed**.

Measured, not inferred:

| Measured workload | Earlier | Current |
|---|---:|---:|
| ~300 simulated game ticks in an ordinary joint-policy episode | ~0.6-0.8 seconds | ~3.9-4.0 seconds |
| `profile01`, 300 ticks, CUDA PPO, 320x180 RGB | — | MCio send/receive 3.876 s; CUDA inference 0.09 s; RGB processing 0.015 s |
| `no_torch_speed01`, 300 ticks, normal controls without Torch | — | 5.008 s |
| `smallframeperf01`, same control at 160x96 | — | 4.892 s |

Historical and current runs are **not necessarily a strictly matched A/B comparison** (different run times, host conditions, process startup/world state). Establish a reproducible matched baseline before claiming a root cause or speedup. See `docs/VISUAL_SELF_SUPERVISION.md` section **5** and local ignored `logs/` for original measurement evidence.

### Immediate operating procedure for the assistant

1. Confirm the user enabled **Gentoo Shell** for this conversation. Work only in its isolated `/workspace` filesystem. Run `pwd`, `git status --short --branch`, `git log -1 --oneline`, and optionally `git pull --ff-only` if clean and remotes are accessible. Git works **directly in the bridge** (including push); the GitHub connector is unnecessary for normal repo tasks. If the bridge is not connected, explain it plainly instead of pretending to access the computer.
2. Read `AGENTS.md`, `TODO.md`, `docs/HANDOFF.md`, **this file**, `docs/VISUAL_SELF_SUPERVISION.md`, `docs/COURSE_CORRECTION.md`, and `docs/FACTOR_POLICY_REVIEW.md`. Respect the newer instructions even if older headings below describe historical priorities.
3. Verify the supported graphics compute stack with `.venv/bin/python tools/gpu-preflight.py`. NVIDIA hardware OpenGL and CUDA were already working on an RTX 4070 SUPER; **don't reinstall Gentoo drivers or widen the Bubblewrap sandbox** based on an untested hypothesis.
4. **First new technical work:** isolate where per-game-tick time is spent, using small fresh-tagged no-neural Minecraft tests. The existing `MCioAgentBridge.apply` loops one send/receive per tick. Compare time in `send_action`, `recv_observation`, server synchronization, rendering, Python frame decode, and between-run startup/warmup as instrumentation permits. Sample the same world/viewport/action sequence across conditions. Review `tools/run-mcio-viewer.sh`, `src/gmaincraft/agent_interface.py`, `experiments/joint_log_factorized.py`, `experiments/factorized_controls.py` and MCio runtime logs. Do **not** assume Minecraft is CPU-rendering: NVIDIA GLX hardware rendering has already been verified.
5. Reproduce the slowdown **without PyTorch**. Confirm effect of launching through the nested Gamescope viewer vs a safe comparable configuration, and isolate Minecraft-versus-client/server pacing. Do not confuse a 90-second outer shell timeout or Gamescope's occasional post-child **139** with an unsuccessful Python experiment; inspect the child logs and durable outputs. Cleanup **only child processes owned by the experiment**.
6. Implement the smallest justified repo-scoped performance fix and remeasure **equal game ticks, same rendering settings, same seed/arena/action sequence**. Preserve correctness: truly synchronized vanilla input, RGB frames, player camera, real inventory events and no action/observation cheating. Update benchmarks/docs with the actual before/after comparison. If not resolved, clearly document the narrowed bottleneck and the next falsifiable experiment; **do not mark this TODO completed**.
7. Run the pure tests (`.venv/bin/python -m unittest discover -s tests -q`) and an appropriate short real-Minecraft smoke. Commit and `git push origin main` through the bridge, updating this file, `TODO.md`, `AGENTS.md`, `docs/HANDOFF.md` and experiment results. Use unique run tags and leave ignored worlds/datasets/checkpoints intact.

**Performance exit criterion:** reproducibly demonstrate a material throughput improvement on matched real-game measurements (and preserve correctness), ideally approaching the previous ~300 ticks / 0.6-0.8 s range. That historical number is a goal, **not a guaranteed entitlement**. If it is not technically attainable, provide an evidence-based explanation and explicitly ask the user whether to proceed with slower training; do not quietly switch priorities after a single inconclusive run.

## SECOND: resume the updated course correction, not the obsolete V1 path

After the throughput issue has been diagnosed and fixed or clearly quantified with the user deciding to proceed:

- Preserve the goal: a **from-scratch learning agent using legitimate player-observable information** to complete Minecraft Java Any% on **unseen genuinely random survival seeds**. Its **current** deployed policy consumes RGB Minecraft screenshots; **RGB-only is NOT a permanent restriction on the project**. A future model may also use ordinary player-accessible game audio, rendered HUD and inventory UI, its own input/action history, elapsed action durations and learned/recurrent memory when implemented and verified. Do not assume these extra modalities have already been wired in.
- **Training is not RGB-only.** Valid learning feedback includes the chosen actions, rewards, game tick durations, episode outcomes, self-collected trajectories, and privately measured Minecraft mined/picked-up stats or controller-only distances for clearly labeled training rewards. Distinguish this privileged *trainer/evaluator feedback* from the observations supplied to the playing policy. No human demonstration data, external pretrained model, hidden world coordinates/target or inventory internals supplied directly as policy observations, or scripted speedrun substitutions unless the user explicitly revises this rule.
- The **first functional milestone** is actual unrestricted finding, mining **and collecting** a log, judged by normal Minecraft inventory. Do not rush to separate crafting/Nether/dragon subsystems while unrestricted first-log collection still fails.
- The current factorized policy has separate **6-tick V2** and **12-tick V3** schemas; real vanilla-mine event stats are evaluator-only, and `CumulativeMinedTracker` persists across resets. **Never** use contaminated `v3event01` as a valid checkpoint or evidence: cumulative stat reset falsely awarded multiple blocks. `v3eventfixed01` is the clean follow-up (10 training episodes; 3 truly mined; 0 inventory pickups in training). In frozen restricted stationary audit, old/new mined 4/4 with 0/4 vs 2/4 inventory pickups; both failed 0/4 fully unrestricted. New survival seed 31415927: 0/240-tick wood for random and V3.
- `docs/VISUAL_SELF_SUPERVISION.md`: own random-action screenshots + CUDA inverse dynamics learned to predict camera and attack, but not movement reliably when simultaneously turning. Camera-locked movement recovered ~38.5% vs a ~14.7% baseline. Causal GRU and frame-delta encodings exist as **offline diagnostics**, not working online memory. Test action disentanglement/history and sample-efficient learning against **same held-out real Minecraft cases**, not merely offline accuracy. Self-collected datasets, local `.pt` weights and world files are **Git-ignored**; do not regenerate/overwrite without need.
- Negative controls and proper validity are mandatory: initial inventory clear, paired scenarios/fixed spawn, held-out different scenario/world seeds, frozen inference, real inventory, checkpoint hashes, no manual command interventions during scored play. The simple normal-input control scored 8/12 versus V2 learned 0/12 on matched nearby log scenes, and the privileged geometry oracle 11/12 is **not** a valid pixel-only policy comparison.
- Optimize Minecraft **game time to win** eventually; faster computation is a tool, not the Any% objective.

## Fresh chat starter (copy-paste into a new conversation)

> Please connect to my **Gentoo Shell** and continue `polskiftw/gmaincraft` in `/workspace` directly. Read `AGENTS.md`, `docs/NEXT_SESSION.md`, `TODO.md`, `docs/HANDOFF.md`, and `docs/VISUAL_SELF_SUPERVISION.md`. **First priority: investigate and fix/quantify the MCio game-step throughput regression (~300 ticks / 0.6-0.8 s before versus ~3.9-4.0 s now), using a controlled no-Torch benchmark. Do not start more agent training until this is addressed.** Once fixed or explicitly triaged, continue the current from-scratch, player-observable-input speedrunning course correction toward real unrestricted first-log acquisition. Preserve all local worlds/weights, use direct Git in the shell, update docs/TODO/AGENTS, run tests, and commit/push validated changes. Please execute rather than merely explain a plan.

**Bridge availability is per conversation:** a new chat does not automatically inherit tool authorization or the old chat's detailed working context. The repository records the project decisions; give the new chat the starter text and enable Gentoo Shell when needed.
