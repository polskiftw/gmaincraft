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
| `orientation_learner.py` (2026-11-02, MCP sandbox) | **3 independent** 300-episode policy/train/eval RNG runs: before 9/50, 7/50, 7/50; after **50/50 each** and extra unseen **50/50 each**. Training 299/300 successes each; learning 30.16s, 26.67s, 25.17s. | Real Minecraft rollouts but one flat-arena world template, numeric goal direction, no vision or learned yaw action; Gamescope exit 139 after completed child |

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
- **Interpretation:** convincing progress on an egocentric **toy motor-control** objective in real Minecraft, now replicated across three independent numeric RNG setups (below); not evidence of perception, learned camera turning, independent Minecraft world-seed generalization, or Any% capability. Next: learned turn actions and nonprivileged perception.

## 2026-11-02 — Independent RNG replication (same saved-world arena template)

The script now has `--run-tag` (write-once separate trial world/checkpoint/CSV) and `--policy-seed`, `--train-seed`, `--eval-seed`, `--extra-seed`, `--action-seed`. Defaults preserve the reference. All runs used Minecraft Java 1.21.3 Fabric/MCio 1.6.1, RTX 4070 SUPER, 300 training episodes, 50 held-out and 50 additional target/yaw scenarios, 35 decisions maximum and 6 ticks/decision. Each disposable world is copied from the **same** `DemoWorld`; *not* independent Minecraft world seeds.

| Run | Policy/train/eval/extra/action seeds | Old fixed/random yaw | New before | New after | Extra unseen | Training wins | Learning time |
|---|---|---|---:|---:|---:|---:|---:|
| Reference | 2718/81003/60211/99501/771 | 50/50, 24/50 | 9/50 | **50/50** | **50/50** | 299/300 | 30.16s |
| `rep_b` | 2719/81004/60212/99502/772 | 50/50, 16/50 | 7/50 | **50/50** | **50/50** | 299/300 | 26.67s |
| `rep_c` | 2720/81005/60213/99503/773 | 50/50, 19/50 | 7/50 | **50/50** | **50/50** | 299/300 | 25.17s |

All three 300-row CSVs were checked for sequential episodes, and all three checkpoints loaded with finite (4,3) matrices. Frozen evaluations total 150/150 held-out and 150/150 additional target/yaw cases, but the shared arena makes broad confidence claims inappropriate. None of this tests visually discovered goals, Minecraft random-world seeds, crafting, combat or Any%. Each Gamescope wrapper exited with SIGSEGV 139 **after** its Python child had printed final results and saved the model/CSV.

Local-only ignored checkpoint SHA-256:
- `orientation_policy.npz`: `9882cae98a038b84c403ba0e644f91710f6af2fc48d0d6aa1628cfc3daf9d399`
- `orientation_policy_rep_b.npz`: `1793a3f59a2fc92346292e1bbc639095fb5dba80c28fce683bf95fb9e488ba21`
- `orientation_policy_rep_c.npz`: `5ac534ce727ab6191dae830a09451588c9c88e9a3ec219adc13b69e52f263f`

**Next milestone:** policy-controlled mouse/yaw (then pitch), followed by nonprivileged visual goals and independent-world seed evaluation.

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
