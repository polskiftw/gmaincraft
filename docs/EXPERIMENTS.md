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
| `log_visual_aim_learner.py` and `log_visual_aim_evaluator.py` (2026-11-02) | In **fresh audited 20-case world**, no-attack negative 0; untrained RGB-feature policy 0/20; independent trained policies rgb01 **15/20** and rgb02 **18/20** real oak-log inventory pickups. Both trained 500 episodes (18.01s, 28.71s learner time). | Agent learned 4 camera movement actions and ATTACK from engineered RGB-only oak-color features and scalar controller angle reward; fixed W finisher, one controlled trunk; **initial 12/14 tests invalidated by stale drops**, audited evaluator clears items before each case. |
| `log_interaction_probe.py` (2026-11-02, MCP sandbox) | Test with controller-placed oak-log column in new copied disposable world, mouse (+120,+50) input changed yaw +18.00°/pitch +7.50°, reverse mouse restored 0/0; 100 ticks left-click mining then 11 ticks forward motion yielded **one oak log in observed inventory**. Launcher-to-finish 20.15s; Gamescope exit 139 after completion. | **Scripted INPUT PROBE, not learned behavior.** Controller /setblock and /tp only for setup; during mine/collect phase no admin commands and no `/give`; real break/pickup via vanilla game mechanics. |
| `orientation_learner.py` (2026-11-02, MCP sandbox) | **3 independent** 300-episode policy/train/eval RNG runs: before 9/50, 7/50, 7/50; after **50/50 each** and extra unseen **50/50 each**. Training 299/300 successes each; learning 30.16s, 26.67s, 25.17s. | Real Minecraft rollouts but one flat-arena world template, numeric goal direction, no vision or learned yaw action; Gamescope exit 139 after completed child |

## 2026-11-02 — Learned RGB-feature camera aiming and real oak-log interaction

- **Git sources:** `experiments/log_visual_aim_learner.py`, `experiments/log_visual_aim_evaluator.py`, `src/gmaincraft/log_vision.py`, `tests/test_log_vision.py`. Python 3.14, `mcio_ctrl==1.5.2`, Minecraft Java 1.21.3 Fabric/MCio 1.6.1; **NVIDIA RTX 4070 SUPER** under nested Gamescope/Xwayland with host mouse forwarding filtered. Two distinct disposable copies of `DemoWorld`: `MDL_LogAim_rgb01` and `MDL_LogAim_rgb02`. The audited independent clean world is `MDL_LogAimAudit_clean01`.
- **Training observation contract:** model sees **only** a 12-dimensional vector derived from the current actual Minecraft RGB screenshot: engineered red/green/blue oak-like color threshold, five horizontal and three vertical mask densities, centroid X/Y, mask fraction and bias. It **never receives** target/block coordinates, Minecraft player position, yaw, pitch or a privileged raycast. This engineered color heuristic is **not learned image recognition**, and can easily fail across different backgrounds/lighting.
- **Actions:** a randomly initialized (5×12) linear softmax REINFORCE policy selects look left/right/up/down (45 cursor pixels ≈6.75°) or ATTACK. Actual mouse movements go through vanilla MCio cursor events, without teleporting camera during scored action. When ATTACK is selected, the *test harness* holds Minecraft's left button for 100 ticks and then walks forward with a fixed W sequence to collect a drop; **pickup movement is NOT learned**.
- **Curriculum:** a 3-high oak trunk 2 blocks ahead of the stationary player on a flat bedrock arena, survival mode. Controller may /setblock, /tp, heal, clear inventory **only at episode reset**, not during scored action. Initial yaw/pitch sampled from `[-28°, +28°]` and `[-8°, +13°]`; one world template and fixed trunk, **not** independent Minecraft world seeds.
- **Reward:** controller computes scalar angular-progress shaping (`0.32×(previous error − new error) −0.04`) and a proxy success bonus/penalty for attack near the known log direction. Only this scalar reward reaches the learned policy; *no correct action labels or geometric goal coordinates*. Frozen evaluator does **not** use the proxy for success: real Minecraft inventory must contain a mined `minecraft:oak_log`.
- **Training runs:** two fresh random-weight models trained separately for 500 real Minecraft episodes. `rgb01`: policy seed 5501/train seed 7723/initial eval seed 8813, 282/500 geometric-proxy successes, learner 18.01s, model updates 1773. `rgb02`: seeds 5502/7724/8814, 220/500 geometric-proxy successes, learner 28.71s, updates 3558. Timing is training only, excludes game startup/evaluation/shutdown. Both training CSVs contain 500 rows; models have finite (5×12) weights.
- **Initial evaluation bug and invalidation:** the first evaluator cleared inventory and rebuilt the target but did **not** remove old dropped item entities from the persistent world. It reported `rgb01` 12/14 and `rgb02` 12/14, but also a nonsensical `rgb02` untrained 11/14 with yaw far from the log. Those **are INVALID as evidence of aiming or acquisition** and should not be quoted as success. Reset now runs controller-only `kill @e[type=minecraft:item]` before every scenario. Real-game item pickup itself remains purely normal Minecraft mechanics.
- **Audited frozen evaluation (the trustworthy result):** a **new clean copied** world (`MDL_LogAimAudit_clean01`), **new held-out camera seed 8815**, 20 identical scenarios per model, maximum 18 look/attack decisions per case. Explicit no-attack/no-mining negative control walked into the log: zero items in inventory. Every trial reset removes all previous dropped item entities and clears inventory. The models were frozen, made decisions from RGB-feature vectors, and only actual observed oak logs counted:

| Policy | Successful inventory pickups | Mean final angular error | Mean camera/attack decisions |
|---|---:|---:|---:|
| No-attack control | 0 | n/a | n/a |
| Untrained policy (seed 5501) | **0/20** | 17.79° | 18.0 |
| Trained `rgb01` | **15/20** | 6.77° | 3.0 |
| Trained `rgb02` | **18/20** | 7.37° | 2.6 |

- **Audit checks:** `logs/log_aim_audit_clean01.csv` has 60 per-model-case records, SHA-256 `15a1db28351db6bbd6bf7df242ea8082ae19947aa3c4b3d1b4d3a0f547f63900`. Of the 33 real successful pickups, attack yaw was within approximately ±9° of log center; a guard rejects grossly implausible success angles. Some valid pickups fall outside the conservative training proxy, showing why actual inventory scoring is essential. Frozen evaluation causes no weight updates. Gamescope still segfaults with exit code 139 only after completed Python child shutdown.
- **Local-only ignored checkpoints:** `checkpoints/log_aim_policy_rgb01.npz` SHA-256 `3c7d94407a107b43db3b59ff31aa3e2026610611667e53fec9fa2049f1f1816a` and `checkpoints/log_aim_policy_rgb02.npz` SHA-256 `78fb07a90fc5cd385e9281a105776386c77269417772e2775afcd6fb6364372b`; training CSVs and world saves are also local/ignored, **not uploaded to GitHub**.
- **Interpretation:** reproducible learned **camera and ATTACK decisions** from visual color features in a toy Minecraft trunk interaction task, with audited actual survival-mode mining/inventory outcomes. Does **not** prove end-to-end visual object recognition, navigation, autonomous pickup, multi-block resource gathering, different biomes/lighting, world-seed generalization or Minecraft speedrunning. No aesthetic or human-looking camera movement objective. Next: learn approach/pickup actions and robust visual detection in varied scenes.

## 2026-11-02 — Wooden-log interaction instrumentation

- Git source: `experiments/log_interaction_probe.py` + pure helpers in `src/gmaincraft/log_curriculum.py`, with unit tests. The test initially ran before commit in the user-owned isolated workspace.
- A **new** copy of `DemoWorld` named `MDL_LogInputProbe_pilot01` was used. Privileged controller *setup only* created a flat bedrock floor, placed a three-block oak-log column 2 blocks ahead at eye height and teleported the survival-mode player to its starting position.
- Once setup finished, the controller sent **no Minecraft administrator commands**. MCio input actions calibrated mouse movement, held left mouse button for 100 ticks and stepped forward. Observed `cursor_pos` began at `(0,0)`; +120 x/+50 y pixels caused +18.00° yaw / +7.50° pitch, and inverse motion restored yaw/pitch 0. One real oak log appeared in Minecraft inventory after 11 W steps. No `/give` was used anywhere.
- Minecraft 1.21.3 Fabric MCio 1.6.1, Python controller 1.5.2, NVIDIA Gamescope viewer with physical host mouse filter. Elapsed since launch 20.15s. Gamescope SIGSEGV exit 139 **after** the Python process printed its PASS and sent stop.
- **Interpretation:** validated actual vanilla input/action/observation and item pickup path. The camera/movement script was predetermined; it used a controller-placed target and no RL training or RGB-based target discovery. Do **not** report this as autonomous wood gathering.
- **User objective:** camera look should be optimized only to support perception and required game interactions, not look smooth, level or human-like. Skyward travel is fine if useful, but mining requires a legitimate crosshair raycast, survival breaking time, reach and inventory observation.
- Next: learn where/when to look, attack and move from rendered pixels; keep privileged arena reset separate from policy inputs and actions.

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
