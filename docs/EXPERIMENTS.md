# Experiment record

This is the **verified evidence ledger**, not a roadmap. Add measured results here only after reading actual runtime output and identify evaluation conditions. Initial records were obtained on 2026-11-01, and subsequent experiments were executed in the user's isolated Gentoo Shell MCP workspace on 2026-11-02. Retain caveats and avoid converting synthetic tests into Minecraft evidence.

| Test / script | Verified output | Interpretation / limitation |
|---|---|---|
| MCio demo | Minecraft 1.21.3 runs, player state and RGB frames received; viewer visible with XWayland and `cbor2<6` | Control connection works, not an RL result |
| `random_agent.py` | Two random-action episodes ran and restarted a clean saved-world copy | Baseline only, no training |
| `benchmark_agent.py` windowed | Two 1,000-step runs, 192.6 and 197.3 active steps/s; 195.0 average | Active simulation only, single machine |
| `benchmark_agent.py --headless` | 163.7 and 207.2 active steps/s; 185.5 average. Whole 2,000-step run 55.505s | Small-sample headless variance; full startup overhead large |
| `persistent_benchmark.py` | Five 1,000-step runs: 208.5, 300.4, 332.6, 314.6, 306.1 steps/s. Combined active 5,000 steps / 17.58s = 284.5 steps/s; whole run 40.50s = 123.5 steps/s | Soft reset ~0.08–0.11s; **not a full world reset** |
| `reward_probe.py` | First oak log +1, second oak log +0, stone pickaxe +1, reset, oak log +1; overall PASS | Controller `/give` tested event detection; agent did not acquire items |
| `navigation_learner.py` | Before: 7/20 targets, avg final distance 5.49 blocks, avg successful time 98 ticks. After 120 training episodes: 20/20, avg final distance 0.70, avg successful time 24 ticks. 12.8s learning time | Real Minecraft movement + online scalar reward updates. Fixed yaw, known numeric goal direction. One held-out set and one run; no statistical confidence estimate |
| `log_pickup_learner.py` + `log_pickup_evaluator.py` (2026-11-02) | 2 random-weight RGB-feature W/A/S/D policies trained 150 episodes each; fresh 32-case actual inventory audit: untrained 2/32, random 16/32, always-W 21/32, trained p01 **30/32**, trained p02 **24/32** | Controller-summoned log fixture and privileged scalar distance shaping for training, **no item position in policy features**; one flat test world template, no mining in pickup-only training |
| `log_mine_and_pickup_audit.py` (2026-11-02) | 2 fresh integrated real mined-log camera-angle audits, total 56 cases/movement: untrained 21/56, fixed W 49/56, learned p01 43/56, learned p02 **51/56** | **Two frozen RGB-feature neural stages:** mouse/ATTACK then learned movement. Centered controller-placed trunk; fixed hold-to-mine timing; no general real-tree perception or statistically established advantage over always-forward |
| `log_visual_aim_learner.py` and `log_visual_aim_evaluator.py` (2026-11-02) | In **fresh audited 20-case world**, no-attack negative 0; untrained RGB-feature policy 0/20; independent trained policies rgb01 **15/20** and rgb02 **18/20** real oak-log inventory pickups. Both trained 500 episodes (18.01s, 28.71s learner time). | Agent learned 4 camera movement actions and ATTACK from engineered RGB-only oak-color features and scalar controller angle reward; fixed W finisher, one controlled trunk; **initial 12/14 tests invalidated by stale drops**, audited evaluator clears items before each case. |
| `log_interaction_probe.py` (2026-11-02, MCP sandbox) | Test with controller-placed oak-log column in new copied disposable world, mouse (+120,+50) input changed yaw +18.00°/pitch +7.50°, reverse mouse restored 0/0; 100 ticks left-click mining then 11 ticks forward motion yielded **one oak log in observed inventory**. Launcher-to-finish 20.15s; Gamescope exit 139 after completion. | **Scripted INPUT PROBE, not learned behavior.** Controller /setblock and /tp only for setup; during mine/collect phase no admin commands and no `/give`; real break/pickup via vanilla game mechanics. |
| `orientation_learner.py` (2026-11-02, MCP sandbox) | **3 independent** 300-episode policy/train/eval RNG runs: before 9/50, 7/50, 7/50; after **50/50 each** and extra unseen **50/50 each**. Training 299/300 successes each; learning 30.16s, 26.67s, 25.17s. | Real Minecraft rollouts but one flat-arena world template, numeric goal direction, no vision or learned yaw action; Gamescope exit 139 after completed child |

## 2026-11-02 — Self-collected vision, V3 mining events and MCio I/O diagnosis

See **[`VISUAL_SELF_SUPERVISION.md`](VISUAL_SELF_SUPERVISION.md)** for all
measured data, sample counts, tagged run results, invalidated reward data and
limitations. Summary of actual experiments (not future forecasts):

- **Policy screenshot issue fixed.** Real 320x180 MCio RGB contained red
  admin-command chatter obscuring game. Dedicated instance-only
  `chatScale:0.0`, `chatVisibility:0` retained normal HUD/aim and removed it.
  `chatVisibility:2` was proven to BREAK observed reset yaw/pitch; never use.
- **Clean self-collected visual experience.** `dyntrain01`: 13 episodes/507
  frame-action pairs (14th unfinished on timeout, not counted); `dynval01`:
  4 distinct episodes/136 held-out pairs. Trained from random weights CUDA:
  raw inverse action CNN validation movement15.4%, camera42.6%, attack84.6%
  versus majority14.7%,23.5%,52.2%. Movement-only camera-locked data
  38.5% prediction versus baseline14.7%. Alternative delta encoding and
  strictly causal GRU diagnostic had modest/uneven gains and did NOT solve
  mixed action control; these are NOT complete Minecraft policies.
- **V3 interval 12.** 12-tick selectable action timing is schema-separated
  from V2 6-tick policy; `v3clean01` had 15 completed training episodes by
  bridge timeout but only 14 included in latest valid atomic checkpoint,
  some restricted stationary mining/collection wins; unrestricted none.
- **Minecraft actual event signal.** `eventstats01` mined and picked 1 actual
  log via vanilla input; both `minecraft:mined` and `minecraft:picked_up`
  increased by one. The original `v3event01` reward data are **INVALID**
  because fresh per-episode MCio caches misread Minecraft cumulative mined
  stat counts (fake 1+2+3+4+5+6). Its checkpoint should not be reused.
- **Corrected reward and no-attack negative control.** Shared
  `CumulativeMinedTracker` prevents resets from manufacturing mining; real
  three-episode `statfix01` reports **1 mined, 0 mined (no attack), 1 mined**.
  Clean `v3eventfixed01` from prebug checkpoint trained10 CUDA episodes,
  **3 truly mined oak blocks**, 0 inventory pickups in training.
- **Frozen same-case audit `v3fixedaudit01`.** Both pre-event V3 and continued
  V3 mined4/4 logs in restricted pre-aimed station; inventory 0/4 vs2/4.
  FULL unmasked movement/look/attack near scenario stayed 0/4 for both.
  Different total training budgets: no proven effect of new event bonus.
- **New actual natural world `v3worldtest01`.** Minecraft seed31415927,
  same spawn (-15.5,76.0,16.5), random and frozen V3 each **0 logs** in240
  game ticks. This verifies V3 natural-eval plumbing only, NOT world transfer.
- **MCio throughput regression unexplained.** `profile01` 300ticks: ~3.876s
  inside normal game action send/receive, versus 0.09s CNN inference and
  0.015s RGB resize. No-Torch control5.008s, half viewport4.892s,
  chat-scale and 6/12 decision timing did not recover earlier 0.6-0.8s.
  Do not guess host causes or attribute to CUDA. Tests/CLI can run under
  the shell timeout by using small fresh-tagged cohorts.

**Conclusion:** genuine learning of some restricted mining/input effects,
no demonstrated unrestricted first-log acquisition, no natural world
survival skill, no completed Any% run. New experiments should target
better pixel-only action/state representations, temporal credit, robust
reward/dropped-item acquisition and sample efficiency, not raw token/episode
count. Local video/screenshots/training data/checkpoints remain ignored.

## 2026-11-02 — V2 factorized action learning and matched controls

**Read `FACTOR_POLICY_REVIEW.md` for complete methods and run commands.**
This is a dated factual report. New CNN controller has **888,750 weights**
and independent movement(6)/camera(5)/attack(2) heads every 6 game ticks.
It is NOT checkpoint compatible with V1's 80-tick single attack macro.
Both versions remain intact; trained from scratch unless explicit warm-start
is noted. All scored phases use vanilla MCio input and check real inventory.

- **Motor-mechanics positive control:** `continuous_mine_probe.py` held the
  real mouse attack button for 20 separate 6-tick actions on a newly placed
  survival oak log. Observed 1 real log in inventory, correct target geometry
  only for fixture setup. NOT learned intelligence.
- **Reset race discovered and fixed:** initial `controlv2a` was intentionally
  aborted by a stale-inventory guard after a genuine positive. MCio hadn't
  delivered `clear @s` before the pose-only reset succeeded. `reset()` now
  requires **both** pose convergence AND empty wood inventory, settles two
  more frames, then begins scoring; next complete run `controlv2b` is valid.
- **Twelve-case paired CONTROL results**: same actual nearby-log cases,
  held-out scenario seed `3737`, fresh reset for every mode. Privileged
  geometry aiming oracle: **11/12** (instrumentation only, NOT fair agent);
  normal scripted 120-tick attack then W: **8/12**; walking-attack then
  stationary hold: **3/12**; random factorized controller: **0/12**.
- **First V2 CNN pilot `factorv2pilot01`:** real CUDA 40 nearby train cases,
  **0/40 inventory successes**. Frozen eval 0/4 near, 0/4 broad. Greedy
  policy often moved backward and away from target, attacked ~5/360 ticks
  in a later matched audit.
- **Stationary restricted `attackrl01`:** 32 CUDA train episodes with
  movement and look explicitly masked out; **3/32 logs**. Held-out frozen
  restricted station **2/4** logs, unrestricted near **0/4**, broad **0/4**.
  This verifies a narrow learned attack tendency, NOT usable general play.
- **Sequential staged `aimnear01`:** warm-start from `attackrl01`, 36 camera
  + attack with movement disabled (1/36 training logs) then 20 unrestricted
  near cases (0/20); all four 4-case frozen slices **0/4**. With all heads
  active, the model often selected stationary movement + endless rightward
  looking + continuous attack. Long-term mining was not possible.
- **Interleaved staged `mixstable01`:** 48 CUDA train episodes: 12 stationary,
  14 camera+attack, 22 unrestricted near, **0 successes in all three**.
  Trainer added a modest privileged sustained-alignment scalar reward
  (NOT actual block damage). Frozen stationary-only restricted 1/4, aim-
  restricted 0/4, unrestricted near 0/4 and broad 0/4. More mixing and
  shaping **did not** fix this model family. Its Python run completed and
  logged all results even though the outer Gamescope wrapper reached
  timeout exit 124 during shutdown.
- **Frozen three-model matched near audit `nearcompare01`:** identical
  `3737` 12-case test and new copied world, all **0/12** for
  `factorv2pilot01`, `attackrl01` and `aimnear01` versus scripted **8/12**.
  Mean ending tree distances ~12.97, ~15.77, ~2.61 blocks respectively.
  The latter stayed nearby but rotated the camera every tick window.
  Model SHA256 fingerprints recorded in local JSONL.
- **Natural seeded-world transfer `factortransfer01`:** new vanilla
  Minecraft 1.21.3 noise-survival world seed **246813579** with fixed
  spawn radius0. Three *different copied worlds* for `forward_attack`,
  `random`, and frozen `joint_factorized` (`mixstable01`). All observed
  identical initial position **(16.5,124.0,0.5)** and yaw/pitch 0/0,
  **0 wood in 360 game ticks** each. Learned policy made 60 decisions.
  This proves world loading/policy interface only; it does NOT show
  positive natural-world transfer.
- **Opt-in full RGB trajectory data `recordpilot01`:** two 120-tick REAL
  CUDA training episodes, 40 actions, 1 valid PPO update, **0 logs**;
  135,670-byte compressed `.npz` locally in ignored
  `runs/joint_factorized_recordpilot01/rollout_00002.npz` reloaded without
  pickle. Arrays include `(40,6,84,84)` uint8 RGB histories, triple
  ordinary actions, scalar rewards/values/logprobs, game-tick durations,
  terminal flags and masked-head settings. Exactly 2 terminal events.
  **No privileged target coordinates or player poses** stored in rollout.

**Conclusion:** V2 eliminates the pathological V1 80-tick action bottleneck
but does not learn general look/move/mine from the observed sample budgets.
A simple scripted normal-control approach remains far stronger on the same
12 held-out near scenes (8/12 vs learned 0/12). More identical PPO training
is not currently justified; next test more sample-efficient visual encoders,
recurrent state and off-policy/replay-aware algorithms using the recorded
agent-only experience while keeping fully independent Minecraft outcomes
as the only success metric. All local saves/checkpoints/results are ignored
by Git; system CUDA and hardware OpenGL function as previously verified.

## 2026-11-02 — Natural-world transfer pilot, NOT paired evidence

- Real Minecraft 1.21.3 server created **seed 314159265**, normal
  `minecraft:noise` overworld. MCio then launched four independent fresh
  source copies; real RGB frames and normal input/pixel-only policy bridge
  worked. No admin Minecraft commands during scored action phases.
- `naturalpilot01`: 100 synchronized game ticks each. Frozen diagnostic
  results: idle **0 logs**, random **0 logs**, forward+attack **0 logs**,
  frozen old `rgb02` camera + `p02` pickup (100-tick scripted attack and
  handoff) **0 logs**. All four ended by tick budget, not a Python error;
  the output lives in ignored `logs/natural_world_naturalpilot01.jsonl`.
- **This was a short functional pilot, not a fair paired outcome comparison.**
  Original source world had `spawnRadius=10`: in logs, modes spawned at
  slightly different coordinates. The runner now writes `spawnRadius=0`
  for *newly prepared* natural source worlds, checks worldgen seed and
  biome/terrain generator, and rejects unverified source templates before
  scored comparisons. It still needs a fixed-spawn real-game verification.
- No learned natural-world acquisition, meaningful effect size, unseen-world
  generalization or Any% success has been demonstrated. No model training
  was performed. The old learned checkpoints and worlds were preserved.
- Existing known Gamescope post-child shutdown exit 139 recurred despite
  completed JSONL results. Do not label wrapper status 139 as a clean exit.

## 2026-11-02 — First joint PPO, natural-world frozen transfer and GPU probes

- **Project-local dependencies:** PyTorch `2.15.0+cu132` successfully
  installed in `.venv` (Python 3.14.10). CPU forward/backpropagation and
  optimizer updates work. **CUDA still fails** inside Gentoo Shell:
  `nvidia_uvm` module loaded after host intervention but current sandbox
  lacks `/dev/nvidia-uvm`; `libcuda cuInit(0)=999`, Torch CUDA false.
  Do not report any GPU-trained model yet.
- **OpenGL renderer:** exact nested Gamescope/XWayland GLX query reports
  vendor `NVIDIA Corporation`, renderer `NVIDIA GeForce RTX 4070
  SUPER/PCIe/SSE2`, OpenGL `4.6.0 NVIDIA 595.104.02`, **not** software
  rasterization. Gamescope still exits SIGSEGV 139 after child termination.
- **CPU synthetic throughput:** `tools/benchmark-training.py --device cpu
  --batch-size 64 --steps 15 --threads 8` measured `0.159 s`, **94.08
  batched gradient updates/s, 6021 images/s**. This tests *only* a fixed
  synthetic CNN batch, NOT Minecraft simulation, PPO training or CUDA.
- **Real Minecraft teacher arena:** `jointsmoke01`, actual Minecraft Java
  1.21.3 with the vanilla survival input path and controller-variable tree
  + player positions. `2` independent 140-game-tick training episodes,
  outcomes **0/2** real log acquisitions. Model: 888,236 random-initialized
  CNN actor-critic weights, two RGB84 frames, 11 joint movement/look/attack
  macros; one PPO update on `34` real collected transitions. Update
  diagnostics: finite `loss=0.04366`, `value_loss=0.14338`,
  `entropy=2.39714`. One valid `jointsmoke01.pt` local model and JSONL
  were saved. **Held-out fresh arena**: 2x140 ticks, **0/2** actual logs,
  models frozen. Complete child test, known Gamescope post-close 139.
- **Natural seeded-world transfer:** `jointtransfer01` on normal noise
  Minecraft seed `161803398`, `spawnRadius=0`, frozen saved checkpoint
  `jointsmoke01`, pure pixel input and normal vanilla actions, no
  controller cheats, **0 logs in 120 game ticks**, no Python/game errors.
  This only validates the interface; *two training episodes* cannot prove
  any useful natural-world learning. Local JSONL:
  `logs/natural_world_jointtransfer01.jsonl`.
- **Not tested:** CUDA acceleration, large-scale learning, statistically
  meaningful held-out worlds, full training speed, consistent first-log
  competence, crafting, Nether/End or Any%.

## 2026-11-02 — First actual CUDA real Minecraft training (post-restart)

- Gentoo Shell bridge restarted after host UVM device creation; inside
  bubblewrap `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` now bound;
  `cuInit(0)=0` and actual PyTorch CUDA tensor compute succeeded on NVIDIA
  GeForce RTX 4070 SUPER, driver 595.104.02.
- Synthetic 25 CNN-update benchmark with batch 64 and 8 CPU threads:
  CUDA 879.69 updates/s vs CPU 128.99 updates/s = ~6.82x speedup for
  **only network training**, not Minecraft/ZMQ throughput.
- `cuda01`: **12 x 220-tick actual Minecraft training episodes** on real
  variable-position log fixture, PPO on CUDA with 3 successful optimizer
  updates (95/63/66 transitions), **0/12 actual logs collected**. Frozen
  held-out arena evaluation **0/4 logs** (220 ticks each).
- Frozen episodes used just 3 actions per 220-tick rollout, a possible
  learned degeneration into repeated **fixed-duration 80-tick mining**.
  Next investigate action durations/exploration/reward before scaling up.
- Local saved checkpoint: `checkpoints/joint_log_ppo_cuda01.pt` (ignored);
  training ledger: `logs/joint_log_ppo_cuda01.jsonl` (ignored). Gamescope
  still exits 139 after successful Python program termination. No claim of
  first-log competence, transfer or Any% learning is warranted yet.

## 2026-11-02 — Atomic checkpoint / warm-start smoke

- CPU `warmcheck01`: loaded pre-existing `jointsmoke01` CNN weights,
  completed one real Minecraft 100-game-tick training episode (**0 logs**),
  updated PPO on 23 real trajectory decisions (second update), then
  atomically saved a valid ~11 MiB checkpoint including optimizer state and
  metadata; local JSONL recorded all details. This *pilot* used the same
  source train-seed and therefore repeated an earlier scenario: it verifies
  continuity plumbing but **not** independent learning. Later code requires
  new policy + training RNG seeds when using `--init-model`.
- Frozen held-out evaluation was explicitly skipped in this check; no
  generalization claim. The earlier independent `jointtransfer01` pilot
  remains the only real natural-world frozen test of this network family.
- 320x180 game rendering with 2-frame 84x84 RGB network input worked.
  Prior `jointsmoke01` and `jointtransfer01` ran at 640x360 before the
  current new-run default changed; do not conflate cross-resolution runs.

## 2026-11-02 — Fixed-spawn actual Minecraft confirmation

- A second distinct *normal noise-generated* survival seed `271828183` was
  prepared using `WorldManager.create`, read-back worldgen seed + generator,
  and **source-only** NBT `spawnRadius=0` (no in-game player teleport).
- Two new copied-world MCio runs, each 25 simulation ticks:
  `idle` initial observed `(0.5, 77.0, 0.5), yaw 0, pitch 0`, **0 logs**;
  `forward_attack` EXACT same observed starting pose, **0 logs**.
- Both completed by tick budget; local result
  `logs/natural_world_fixedspawn01.jsonl` recorded source seed, mode, pose,
  observations, outcome, elapsed time and world names. Not a meaningful
  learning result or statistical test. Gamescope still segfaulted only after
  completed Python benchmark and JSONL writing. First pilot's 0/100 results
  remain unpaired and should not be used for policy comparison.

## 2026-11-02 — Learned W/A/S/D pickup and two-policy mined-log integration

**Environment:** same Gentoo/OpenRC, RTX 4070 SUPER, Minecraft Java 1.21.3 Fabric MCio 1.6.1, MCio Python controller 1.5.2, nested GPU Gamescope viewer with desktop-mouse filter. Repeated synchronous Minecraft rollouts in *copied disposable worlds* based on the same `DemoWorld` template; no independent random world seeds.

**New learner:** `experiments/log_pickup_learner.py`, `src/gmaincraft/log_pickup.py`, `tests/test_log_pickup.py`. A 4×14 linear softmax REINFORCE policy initialized with fresh random weights chooses **real W/A/S/D key inputs** every 5 game ticks. It sees **only** 14 hand-engineered features from the **actual screenshot** (cropped oak-brown color-mask 7 horizontal, 3 vertical bins, centroid X/Y, detection strength, constant bias). No item/block coordinates or player yaw/position are policy inputs. Training controller privately knows the *fixture spawn position* to provide scalar reward: `3 × (distance before − distance after) − 0.03` per decision, plus +12 only for **real Minecraft oak-log inventory acquisition**; out-of-bounds penalty −4. **Training fixture:** one controller-summoned dropped `minecraft:oak_log`, position dx around ±0…2.45 blocks, dz 2.45…3.75, initial yaw ±15°, pitch 30°. This controller spawn is a *teaching setup*, not natural mined-item acquisition.

**Independent training:** 150 real Minecraft episodes each. These two original trainings occurred **before** adding zero-initial-Motion to the fixture code; the later accepted fresh-world audit used the corrected fixture. `p01` weights seed 4401, train cases seed 6601, in-process eval seed 8801; 123/150 inventory pickups, 14.59s learner-only. `p02` seeds 4402/6602/8802; 109/150 inventory pickups, 15.56s learner-only. Original in-process frozen-eval success was `p01` 19/20 (before 1/20) and `p02` 14/20 (before 0/20), but independent fresh-world audit takes precedence. Their ignored checkpoint files are `checkpoints/drop_pickup_policy_p01.npz` (4×14, 989 updates; SHA-256 `93abd3a119654de53c4502942128c019d1fd3e26941a04086252318a5fed8162`) and `checkpoints/drop_pickup_policy_p02.npz` (4×14, 998 updates; SHA-256 `163c2c427f24d776ca1f759ebb80d6363668107803a193f3fcf87ef9c966cf44`). Training CSV files each contain exactly 150 episodes; both are ignored/local.

**Audit correction:** the first independent audit `MDL_DropAudit_dclean01` was aborted after a random policy picked an item 2.5077 blocks from its *original spawn location*; that single distance is not sufficient to establish stale item contamination because Minecraft dropped entities drift. This **incomplete audit is NOT evidence**. To make the fixture reproducible, new summons include `Motion:[0.0d,0.0d,0.0d]` (normal Minecraft gravity and pickup still apply), and the evaluator excludes any implausibly distant success rather than silently counting it.

**Accepted clean independent fixture audit:** world `MDL_DropAudit_dclean02`, **new test seed 8804**, 32 identical cases for each frozen policy/controller, max 22 decisions ×5 ticks. Every episode reset kills old dropped entities and clears inventory; a no-movement negative control saw zero logs, and a fixed-W positive test retrieved a fresh fixture log. No policies learned during audit and no suspected distant pickups were excluded.

| Frozen policy/control | Real item pickups /32 |
|---|---:|
| No-action control | 0 (one dedicated negative-control fixture) |
| Untrained seed 4401 | **2/32** |
| Random W/A/S/D | **16/32** |
| Always W | **21/32** |
| Trained `p01` | **30/32** |
| Trained `p02` | **24/32** |

`logs/drop_pickup_audit_dclean02.csv` contains 160 scenario rows, SHA-256 `9fd470bc0d82cdb443ea1a3b05b6b6b6a85e3b302e9fc8240c505c81f3203846`. This is direct evidence of learned *off-center collection movement*, but only for a simple floor-based **controller-summoned** item with engineered color features.

**Combined Minecraft mining + learned movement:** `experiments/log_mine_and_pickup_audit.py` loads previously trained, frozen aiming model `rgb02` and a separate frozen pickup model (`p01` or `p02`). The first model receives RGB-derived features and chooses mouse yaw/pitch or ATTACK. On ATTACK, the test holds left click for a **fixed 100 ticks** to mine a *controller-placed oak-log block* in survival mode, then passes control to the second model choosing W/A/S/D in 5-tick batches—**no scripted post-attack W in neural-model modes**. All scored actions are ordinary Minecraft inputs; the controller's `/tp`, `/setblock`, `/kill` and inventory clear occur **only during each episode reset**. Score is **real mined `minecraft:oak_log` in observed inventory**. Each experiment reuses one centered 3-block trunk and the same template world, not general terrain.

| Movement after learned camera/ATTACK | `f01` (seed 8816, 24 cases) | `f02` (seed 8817, 32 cases) | Total |
|---|---:|---:|---:|
| Untrained policy | 8/24 | 13/32 | **21/56** |
| Hardcoded always W | 23/24 | 26/32 | **49/56** |
| Trained `p01` | 21/24 | 22/32 | **43/56** |
| Trained `p02` | 22/24 | 29/32 | **51/56** |

In `f02`, a no-attack negative control walked into the intact trunk and produced **0 oak logs**. In both completed audits, model `rgb02` elected ATTACK in all evaluated cases, the policies were frozen, and no old item entities survived reset. Each movement result was audited across **identical held-out starting camera angles**. `f01` saved 96 rows (`logs/full_log_pickup_audit_f01.csv`, SHA-256 `9e5a1c6405f77f9067c5d86d6d99857cc36469e8920b2d612fdc919f12d1c1d0`); `f02` saved 128 rows (`logs/full_log_pickup_audit_f02.csv`, SHA-256 `e01a1e1d25e0f0207411ead0aae2797d2c9d3a32fdaf32abf2222d1f381f6b26`). In `f02`, `p02` learned movement was **almost entirely W** (155 W decisions), consistent with the fixed centered-target setup; its 51/56 versus hardcoded W's 49/56 **does not establish statistically meaningful superiority**. In contrast the offset-item audit shows meaningful lateral-action learning (p01 used W/A/D).

**Limitations/next:** separate aim and pickup models are trained separately, not a fully joint end-to-end agent. Camera is not aesthetic; the agent still must aim at actual blocks and move through normal game mechanics. Trunk remains controller-placed in a flattened world, brown-color threshold is engineered, controller knows test geometry to compute scalar *training* shaping, and fixed 100-tick attack hold is still scripted. Neither stage can yet find natural trees, learn crafting or navigate an unseen random seed. Next create randomized *actual trunk positions* and player positions, test if learned movement beats W beyond simple center-aligned log, and gradually replace hand-engineered visual features/privileged scalar shaping with learned robust perception and legitimate event rewards.

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
