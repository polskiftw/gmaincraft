# Handoff for a brand-new ChatGPT conversation

**Last verified: 2026-11-02.** Canonical code: [polskiftw/gmaincraft](https://github.com/polskiftw/gmaincraft), branch `main`. **Newest milestone:** learn movement W/A/S/D from real RGB features (2 independently trained models; fresh fixture audit p01 **30/32**, p02 **24/32**, always-W 21/32); then **combine frozen RGB aim/ATTACK and pickup policies** on *real mined logs* (two independent camera-angle sets, trained p02 **51/56**, hardcoded W **49/56**, untrained movement 21/56). Earlier numerical-navigation and RGB log-aiming milestones remain preserved; `docs/EXPERIMENTS.md` records all results. The initial uncleaned aim 12/14 and aborted first drop-distance audits are **invalid evidence**. No general world-seed Minecraft speedrunning. Do not ask for old test logs.

## Newest: CUDA training operational after bridge restart

CUDA now actually WORKS: UVM devices visible, `cuInit(0)=0`, Torch CUDA
allocation passed. A 25-step synthetic CNN batch-64 model benchmark measured
~6.82x GPU versus CPU update throughput. Real Minecraft `cuda01`: 12 joint
PPO training episodes / 3 GPU PPO updates, 4 frozen eval episodes, 220 ticks
each, 0/12 train logs and 0/4 eval logs. Both local model and JSONL saved;
Git-ignored. Frozen policy exhausted episodes in just 3 long actions,
indicating potential fixed 80-tick attack-hold macro exploitation; **next
investigate before large GPU experiments**. Gamescope post-child exit 139
remains. Historical CUDA-unavailable remarks below are now superseded.

## Latest checkpoint continuity check (2026-11-02)

`warmcheck01` successfully loaded the old joint model, trained one more
100-tick **CPU** Minecraft episode with 23 transitions, ran PPO's second
update and saved a valid **atomic** 11 MiB checkpoint with optimizer state.
No log acquired; this used repeated scenario RNG, so it is just an interface
smoke, not independent training progress. New code rejects source train/policy
seed repetition. For future work use `--init-model OLD_TAG` **with a new
`--run-tag`, `--train-seed` and `--policy-seed`**. This warms weights and Adam,
but is not an exact interrupted-episode restore. Preserve existing models
and follow `docs/JOINT_LEARNING.md` rather than rerunning a used tag.

## Newest handoff: joint visual PPO and RTX GPU, 2026-11-02

Source additions: `src/gmaincraft/vision_ppo.py`, `ppo_math.py`,
`joint_curriculum.py`, `experiments/joint_log_ppo.py`, optional
`requirements-gpu.txt`, C GLX probe, GPU preflight, CPU/GPU training
benchmark. Read `docs/JOINT_LEARNING.md` and `docs/GPU_ACCELERATION.md`
first. Joint agent uses two RGB84 screenshot frames, 888,236-parameter
CNN/PPO, 11 vanilla timed macros including a **fixed 80-tick mining hold**.
Not trained to craft, explore general terrain, or speedrun. Verified
CPU real-game pilot `jointsmoke01`: 2 training+2 heldout Minecraft log
scenarios with PPO gradient update, 0/2 logs each. Frozen natural-seed pilot
`jointtransfer01` seed 161803398, 0/120 ticks; no errors in Python child.
These runs establish pipeline only; large-scale RL research remains pending.

RTX 4070 SUPER hardware OpenGL **confirmed** via nested XWayland GLX probe
(`GL_RENDERER=NVIDIA ...`, 4.6 NVIDIA). Torch 2.15.0+cu132 installed only
into project `.venv` and works on CPU, but CUDA still **unavailable inside
current bridge**: `nvidia_uvm` host module loaded, no bound `/dev/nvidia-uvm`,
`cuInit(0)=999`. Host needs UVM node creation and bridge restart; instructions
in `docs/GPU_ACCELERATION.md`. Do not claim real GPU training completed.
Use `--device cuda` for fail-fast once devices are visible, verify actual
allocation first with `tools/gpu-preflight.py`. No modifications to host
launcher, Gentoo packages or broader sandbox permissions were made.

## NEW course correction (2026-11-02)

Read `docs/COURSE_CORRECTION.md` and the highest-priority block of `TODO.md`
first. Existing trained aim/pickup models remain valuable *diagnostic* baselines
but have not transferred to general Minecraft. New source modules include
`agent_interface.py` and `natural_eval.py`, plus the real seeded-world runner
`experiments/natural_world_benchmark.py`. A pilot natural Minecraft seed
314159265 was generated and run: 0 logs for idle, random, forward+attack and
frozen `legacy_probe` within 100 ticks; no Python/game test error. Pilot
spawn positions differed (original source default spawnRadius=10); **do not
claim paired benchmark validity**. New `--prepare` fixes source spawnRadius=0
and verifies natural noise terrain + exact worldgen seed. A SECOND real-game
smoke test on new seed 271828183 ran idle and forward+attack for 25 ticks
per mode; **both actual start positions matched exactly (0.5,77,0.5), yaw 0,
pitch 0**, and both recorded zero logs. This verifies paired starting pose
only for these two cases; longer multi-seed testing remains open. Gamescope still reports exit
139 after Python child completes. Use normal `git` from Gentoo Shell on
`/workspace`, commit/push; never expose scoped private Git credentials.

## First actions for the next assistant

1. Read `AGENTS.md`, `TODO.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/EXPERIMENTS.md`, inspect relevant code/tests and recent GitHub commits.
2. Recognize the last verified milestone: all three camera-relative numerical policies scored **50/50 after training and 50/50 on an additional unseen target/yaw set**. Each trained for 300 real Minecraft episodes with distinct policy/train/eval/extra/action seeds. Each training CSV shows **299/300 wins**.
3. **NEXT:** the separately learned W/A/S/D pickup skill is DONE (fresh fixture audit 30/32 p01, 24/32 p02 vs W21/32), and combining frozen aim+pickup collected real mined logs (p02 51/56, W49/56). **Variable trunk/player positions and an initial shared visual PPO controller now exist**, but only zero-win smoke tests completed; next train/evaluate actual improvements against baselines in independent natural worlds. Preserve old checkpoints. The user does NOT want human-looking camera behavior; odd gaze is fine if vanilla crosshair/reach and observation rules are followed. Retain no-attack/dropped-item cleanup safeguards.
4. Later remove privileged numeric goal-direction inputs for visual/perceivable goals. Independently seeded *Minecraft worlds* remain untested.
5. Commit code, tests and all status docs together to GitHub, then give the user safe `git pull --ff-only` commands. Preserve local ignored checkpoint and world artifacts.

## Real Minecraft orientation evidence (2026-11-02)

| Run | Old fixed yaw | Old random yaw | Fresh before | Fresh after 300 episodes | Extra unseen target/yaw set | Learning time |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 50/50 | 24/50 | 9/50 | **50/50** | **50/50** | 30.16s |
| `rep_b` | 50/50 | 16/50 | 7/50 | **50/50** | **50/50** | 26.67s |
| `rep_c` | 50/50 | 19/50 | 7/50 | **50/50** | **50/50** | 25.17s |

Seeds (policy/train/eval/extra/action): reference 2718/81003/60211/99501/771; `rep_b` 2719/81004/60212/99502/772; `rep_c` 2720/81005/60213/99503/773. Local-only valid outputs: `checkpoints/orientation_policy.npz`, `checkpoints/orientation_policy_rep_b.npz`, `checkpoints/orientation_policy_rep_c.npz`, three corresponding `logs/orientation_training*.csv` files (300 rows each) and three disposable `MDL_OrientationLab_v1*` worlds copied from the same `DemoWorld` template. All model weights are finite (4,3) matrices. This is **not** general Minecraft world-seed success, no end-to-end speedrun, no visual recognition, no camera-turning action.

`experiments/orientation_learner.py` now supports `--run-tag` with write-once world/log/model filenames, and `--policy-seed`, `--train-seed`, `--eval-seed`, `--extra-seed`, and `--action-seed`. Existing default run behavior remains unchanged. Five new seed/name tests supplement the five existing numerical policy tests.

## NEW learned RGB movement and integrated mining milestones (2026-11-02)

- New sources: `src/gmaincraft/log_pickup.py`, `experiments/log_pickup_learner.py`, `experiments/log_pickup_evaluator.py`, `experiments/log_mine_and_pickup_audit.py`, `tests/test_log_pickup.py`; common MCio action wrapper now maps W/A/S/D and attack. Python project unit suite expanded to **24 passing tests**; real-GPU Minecraft trials completed in this same MCP workspace.
- Two new independent 150-episode movement trainings: RGB screenshot → **14 handcrafted oak-item features** → random-weight 4×14 softmax network → real W/A/S/D actions. `p01` seeds policy/train/eval 4401/6601/8801, 123/150 training pickups, 14.59s learner time; `p02` seeds 4402/6602/8802, 109/150 training pickups, 15.56s. Training item is **controller-summoned as an in-game dropped item**; the subsequent corrected independent audit uses the updated zero-initial-Motion fixture; controller shapes scalar reward from item spawn position, which policy NEVER sees.
- **Clean independent spawned-item audit:** world `MDL_DropAudit_dclean02`, seed 8804, 32 trials/controller, no-movement negative 0, untrained 2/32, random 16/32, always W 21/32, trained p01 **30/32**, trained p02 **24/32**. `logs/drop_pickup_audit_dclean02.csv` 160 rows (ignored). First `dclean01` audit is INVALID/incomplete due conservative distance-from-moving-item guard; do not quote it as success. Reset kills old items before each test.
- **Combined actual block mining + policy-chosen collection:** pretrained `rgb02` selects camera/ATTACK from screen, fixed 100-tick Minecraft left-click mine, then frozen `p01`/`p02` chooses W/A/S/D from screen. **No scripted collection movement in neural-controller modes.** Two fresh test sets: seed 8816 (24 cases), 8817 (32 cases), all controller commands restricted to reset, score is real mined oak_log inventory. Over **56 identical cases per controller**: untrained movement **21/56**; scripted W **49/56**; trained p01 **43/56**; trained p02 **51/56**. No-attack negative control yielded zero in the second audit. In this fixed centered-trunk setup even trained p02 mostly chooses W and its 2-win edge over scripted W is **not a meaningful proven advantage**.
- **NOW:** initial variable-trunk/player RGB joint PPO prototype exists, 0/2 held-out successes after just two training episodes. Next evaluate actual improvement over baselines, stronger visual perception and general survival on distinct random-world seeds. Detailed data/limitations: `docs/EXPERIMENTS.md`.

## NEW learned RGB camera/attack milestone (2026-11-02)

- Two fresh 500-episode Minecraft training runs: `experiments/log_visual_aim_learner.py`, policy seeds 5501/5502, train seeds 7723/7724, original camera eval seeds 8813/8814, learner time 18.01s/28.71s. Models 5×12 linear softmax initialized from random weights; actions look left/right/up/down/ATTACK. Features come from **actual RGB screenshot via engineered oak-brown mask/spatial bins**; no target coordinates or player yaw/pitch reach the policy. Controller shaping uses **scalar privileged angular-progress reward** (not fair reward in free-form Minecraft). In the **original camera-only evaluation**, after agent-chosen ATTACK the harness held left button and moved forward with **scripted W pickup**. The newer integrated evaluator (above) replaces that scripted movement with a separately trained neural pickup policy.
- **Important measured evaluation bug:** initial soft resets cleared player inventory but left *dropped items from earlier trials*. Thus 12/14 initial scores (and nonsensical untrained 11/14) are INVALID. The reset now sends **controller-only** `kill @e[type=minecraft:item]` before restoring the target/teleport; real item drops from the current ATTACK are still obtained normally.
- Independent, **clean audited frozen evaluation** uses fresh copied `MDL_LogAimAudit_clean01`, new angle seed 8815, 20 cases/model, no-attack negative control 0, untrained model 0/20, trained `rgb01` **15/20** and trained `rgb02` **18/20** real oak-log inventory acquisitions. `logs/log_aim_audit_clean01.csv` (60 rows) and two 5x12 checkpoints are **ignored local files**. Trained models are saved in `checkpoints/log_aim_policy_rgb01.npz` and `log_aim_policy_rgb02.npz` (see ledger for SHA-256). Both original numerical orientation/navigation checkpoints remain unchanged. At this **earlier camera-only stage**, general visual perception and pickup movement were unlearned; the newer learned pickup and integrated evaluations are reported in the section above. Independent world seeds, general tree recognition and speedrunning are still untested.
- Local one-shot command for another **new** audit tag: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_evaluator.py --run-tag audit02 --models rgb01 rgb02 --eval-seed 8816 --eval 20 --visible`. This creates only a disposable copied world, never overwrites checkpoints, and performs zero training. Continue with ordinary Git operations inside the connected shell.

## Updated interaction curriculum and honest status

- **User camera rule (2026-11-02):** camera pose/appearance should never receive human-likeness rewards; the agent may stare at sky or move sideways/backward, but must still use vanilla first-person line-of-sight, crosshair reach and mouse-based interactions where needed.
- `experiments/log_interaction_probe.py`: **VERIFIED scripted vanilla log mining/pickup in actual Minecraft**, not RL. In a *new disposable copied world* (`MDL_LogInputProbe_pilot01`), the controller placed a log column, then no commands were sent after staging. MCio mouse deltas +120,+50 pixels changed yaw +18.00° and pitch +7.50°, inverse deltas restored camera. Holding left attack 100 ticks then walking forward 11 ticks yielded **1 real oak log** in observed inventory; 20.15s total launch+test. Gamescope exit 139 occurred only after printout. No `/give`. Four new pure-helper tests were added.
- **DONE (bounded scope):** two models learned camera-turn and ATTACK choices from screenshot color features (15/20, 18/20 versus untrained 0/20); **now a separate learned RGB-based pickup policy also exists** and an integrated camera-ATTACK-plus-movement audit works on real mined logs (p02 51/56 across two angle seeds). **FOLLOW-UP:** new variable-log joint CNN/PPO exists as an unproven prototype; do not claim tested general tree skill from its zero-win smoke tests. This is NOT an independent-world Any% agent.

## Runtime and local environment

- Gentoo Linux/OpenRC, Plasma Wayland, i7-14700KF, RTX 4070 SUPER driver 595.104.02, 32GB RAM, Python 3.14.10.
- Host project `~/Documents/minecraft-discovery` is exposed as `/workspace` through a custom Gentoo Shell MCP Bubblewrap sandbox. Host `/home` remains **not exposed**. Network namespace is now intentionally **shared with host** for GitHub HTTPS/SSH access, including possible LAN and host-loopback connectivity. Repo-specific SSH write deploy key under ignored `mcio-data/git-auth/` allows normal shell `git push`; HTTPS fetch also works. GPU/DRM devices, selected read-only sysfs and one host Wayland socket are available.
- MCio Python controller `mcio_ctrl==1.5.2` with `cbor2==5.9.0`, Minecraft Java **1.21.3**, Fabric MCio **1.6.1+1.21.3**, Java 21 MCio runtime. Instance/template are `mcio-data/instances/DemoInstance` and `DemoWorld`. MCio ZMQ ports 4001/8001 require a working private localhost `/etc/hosts`.
- Inside MCP, Minecraft uses nested NVIDIA Gamescope/Xwayland. **Use `tools/run-mcio-viewer.sh`**, which builds the SDL2 no-host-mouse library and filters real mouse motion, clicks and scrolling (not MCio agent actions), keeping your pointer free. Example unique disposable test: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/orientation_learner.py --visible --run-tag viewer_demo --episodes 8 --eval 5`. Local Xwayland wrapper has `-ac` only for its private X11 socket. SDL2 simulated-input test and Minecraft MCio smoke 10/10 passed; physical desktop pointer hover verification is pending. See `docs/DEVELOPMENT.md`.
- **Gamescope exits SIGSEGV 139 after each completed Python run**; test results/checkpoints/CSVs were written and validated first. Direct MCio NVIDIA headless without Gamescope failed with GLFW/EGL in this sandbox.
- All checkpoints, saves, Python virtualenvs, local MCP launcher helpers, huge raw logs and model files are git-ignored; only source/docs/tests belong on public GitHub. Never access other host folders, alter global Gentoo settings or add a daemon for these experiments.

## Scientific boundaries

- Privileged arena controller may use `/tp`, `/fill`, `/setblock`, `/kill`, `/summon` **only for setup/cleanup**. Earlier numerical navigation policy receives a hidden goal vector; the newer **learned camera/ATTACK and W/A/S/D pickup policies receive only handcrafted RGB pixel features** and *privileged scalar training reward*, no player/item world coordinates. Controller randomizes camera on reset; neural aim policy controls camera through real MCio mouse inputs during play. Fixed 100-tick mining hold and the two-stage handoff are still scripted scaffolding.
- The 3 independent seed sets are different *numeric RNG seeds*, not different Minecraft world seeds. All three use a small flattened arena. These are strong curriculum checks but do not prove survival, perception, crafting, Nether/End progression or cross-world generalization.
- A soft reset does not fully restore block/entity/RNG world state. Preserve saved checkpoints and `DemoWorld` across future experiments; avoid overwriting known-good assets.
