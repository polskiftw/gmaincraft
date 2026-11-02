# Experiment record

This is the **verified evidence ledger**, not a roadmap. Add measured results here only after reading actual runtime output and identify evaluation conditions. Current records were reported from the user's Gentoo machine on 2026-11-01. Retain caveats and avoid converting synthetic tests into Minecraft evidence.

| Test / script | Verified output | Interpretation / limitation |
|---|---|---|
| MCio demo | Minecraft 1.21.3 runs, player state and RGB frames received; viewer visible with XWayland and `cbor2<6` | Control connection works, not an RL result |
| `random_agent.py` | Two random-action episodes ran and restarted a clean saved-world copy | Baseline only, no training |
| `benchmark_agent.py` windowed | Two 1,000-step runs, 192.6 and 197.3 active steps/s; 195.0 average | Active simulation only, single machine |
| `benchmark_agent.py --headless` | 163.7 and 207.2 active steps/s; 185.5 average. Whole 2,000-step run 55.505s | Small-sample headless variance; full startup overhead large |
| `persistent_benchmark.py` | Five 1,000-step runs: 208.5, 300.4, 332.6, 314.6, 306.1 steps/s. Combined active 5,000 steps / 17.58s = 284.5 steps/s; whole run 40.50s = 123.5 steps/s | Soft reset ~0.08–0.11s; **not a full world reset** |
| `reward_probe.py` | First oak log +1, second oak log +0, stone pickaxe +1, reset, oak log +1; overall PASS | Controller `/give` tested event detection; agent did not acquire items |
| `navigation_learner.py` | Before: 7/20 targets, avg final distance 5.49 blocks, avg successful time 98 ticks. After 120 training episodes: 20/20, avg final distance 0.70, avg successful time 24 ticks. 12.8s learning time | Real Minecraft movement + online scalar reward updates. Fixed yaw, known numeric goal direction. One held-out set and one run; no statistical confidence estimate |
| `orientation_learner.py` (2026-11-02, MCP sandbox) | 300 training episodes, 299/300 wins; former fixed yaw 50/50; former random yaw 24/50; new before 9/50; **new after 50/50**; **new extra unseen 50/50**; learner 30.16s | Genuine Minecraft rollouts, frozen evaluations in the same flat arena, camera-relative *numeric* goal direction; one training RNG seed, no visual recognition or learned yaw control; Gamescope segfaulted only on exit |

## 2026-11-02 — Real Minecraft camera-orientation curriculum

- Git revision of experiment code: `f3de439ff228b73a422adfec459ed4d259fec1e3` (experiment script unchanged during test).
- Runtime: Python 3.14.10; `mcio_ctrl==1.5.2`, `cbor2==5.9.0`, Minecraft 1.21.3 Fabric with MCio 1.6.1; NVIDIA RTX 4070 SUPER driver 595.104.02, accelerated Gamescope nested Xwayland in Bubblewrap. Host home remained hidden.
- World/reset: copied `DemoWorld` once into **only** `MDL_OrientationLab_v1`; flattened arena. Per-scenario resets teleport player, reset health/effects/inventory and set floor marker. **Not** independent random-world seeds or full world restoration each episode.
- Policy/actions: original saved absolute-goal policy tested with camera yaw 0 or randomized; fresh `EgocentricPolicy(seed=2718, learning_rate=0.12)` initialized from random weights (not from previous checkpoint). Both receive exact numeric target offset; actions restricted to W/A/S/D, **no camera-turn action**. Reward is distance improvement minus 0.025/decision with success bonus +4 and failure penalties.
- Scenario seeds: original held-out target/yaw generation `60211`; additional unseen target/yaw generation `99501`; training scenarios `81003`; frozen evaluation action RNG `771`. Test sets are different target/yaw draws, **not different Minecraft world seeds**.
- Configuration: `--episodes 300 --eval 50 --decisions 35 --ticks 6 --visible`. Success means within 1.1 blocks of controller-selected numeric target in an empty arena.
- Measured summaries (win count; mean final distance; mean ticks **among successful trials**):

| Evaluation | Wins | Mean final distance (blocks) | Mean successful ticks |
|---|---:|---:|---:|
| Old policy, fixed camera | 50/50 | 0.72 | 25.6 |
| Old policy, randomized camera | 24/50 | 7.03 | 41.2 |
| New camera-relative policy BEFORE training | 9/50 | 7.22 | 71.3 |
| New camera-relative policy AFTER training | **50/50** | 0.66 | 25.6 |
| New policy, additional unseen target/yaw scenarios | **50/50** | 0.65 | 26.0 |

- Training output: **299/300 successful training episodes**, five runs of 50/50 from episodes 51–300; measured `Learning time: 30.16s` (excludes initial game startup and all evaluations). CSV row IDs 1–300 validated; checkpoint fields `weights` shape (4,3), `baseline`, and `updates`=1391 were loaded and checked as finite. Original navigation checkpoint remained in place.
- Local-only artifacts: `checkpoints/orientation_policy.npz` (SHA-256 `9882cae98a038b84c403ba0e644f91710f6af2fc48d0d6aa1628cfc3daf9d399`) and `logs/orientation_training.csv` (SHA-256 `41daac9949d5b5785e2df682573f7902a34a6b5d7a131457b968a67375545fd0`). Neither is committed to public GitHub.
- The Gamescope wrapper **segfaulted at shutdown** (exit 139) after Python printed final evaluations and checkpoints/logs had been written and validated; this is a real compositor issue but does not erase the measured experiment results. A tiny 8-episode/5-target pilot had also completed before the full run.
- **Interpretation:** convincing progress on an egocentric **toy motor-control** objective in real Minecraft; not evidence of perception, learned camera turning, robust training across independent seeds, independent-world generalization, or Any% capability. Next: independently seeded replication, then learned turn actions and nonprivileged perception.

## Measurement conventions

- `steps/s` is MCio synchronous environment step throughput, not general Java FPS and not neural-network updates/s.
- `active` excludes Minecraft start/stop and world reset; `end-to-end` includes those costs. State which one is used.
- Navigation success requires reaching within the controller's configured radius of a chosen gold block on a flattened test arena. It does not require seeing/identifying gold in the screenshot; the target vector is provided.
- Training and evaluation data should be disjoint by explicit RNG seed, with seed sets and action/reward protocol reported. The 20-target improvement alone does not establish robustness.
- Fast soft reset retains changes to Minecraft world; distinguish from clean save-copy full restarts.

## Known nonfatal logs

- Minecraft's Linux narrator could not load `libflite.so`; game and MCio experiments still worked.
- Some launches timed out looking up Mojang profile properties; the game still loaded.
- These are not currently reasons for system-wide dependency changes.

## Future entry template

### YYYY-MM-DD — experiment name

- Git revision:
- Runtime: Minecraft/MCio/Python; visible or headless; machine:
- Training/world seeds; test seeds; policy starting state:
- Observation contract; available actions; reward function; reset mechanism:
- Episodes/steps/time; baseline and frozen eval:
- Actual raw summary, not inferred: 
- Limitations and failures:
- Checkpoints/logs location (local, ignored by Git):
- Follow-up TODO and corresponding code commit:
