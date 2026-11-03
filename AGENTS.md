# AGENTS.md — project instructions for coding assistants

This file is the operating contract for anyone (including a future ChatGPT conversation) editing **polskiftw/gmaincraft**. Treat the GitHub repository, not the conversation transcript or an assumed local checkout, as the authoritative codebase.

## MANDATORY FIRST TASK FOR A NEW CHAT: MCio speed regression (highest priority)

**Start with [`docs/NEXT_SESSION.md`](docs/NEXT_SESSION.md), not a new PPO
run.** The user explicitly requires **diagnosis and remediation of the
Minecraft/MCio game-step throughput regression BEFORE resuming AI training**.
Earlier normal real-game runs took ~0.6-0.8 s/300 ticks, current ~3.9-4.0
s/300; profile showed ~3.876 s inside MCio game send/receive, ~0.09 s CUDA
inference and ~0.015 s RGB conversion. A no-Torch control is also slow;
small viewport, 6-vs-12 game-tick action intervals and changing chat scale
did not fix it. The comparisons must be matched before claiming causality.

**Order of operations:** connect to Gentoo Shell `/workspace` -> inspect
repo/docs and GPU preflight -> reproduce/trace slow per-tick Minecraft I/O
with minimal no-ML control -> fix/quantify and rebenchmark under matched
conditions -> update evidence/docs/tests -> only THEN continue the valid
from-scratch RGB-only RL course correction toward unrestricted first-log
mining + inventory pickup. Do not spend hours on model training first.
If root cause is unresolved, preserve precise profiler results and request
a user decision before accepting the slower speed as the new baseline.

Use direct `git` from the shell (including commit/push), keep all ignored
worlds/checkpoints/data unchanged, and never widen the restricted sandbox or
modify host drivers without demonstrated need and user agreement. New
conversation must have the Gentoo Shell bridge connected/authorized; do not
claim it is connected if it is not. Update `TODO.md` and `docs/HANDOFF.md`
with each verified milestone. See `docs/VISUAL_SELF_SUPERVISION.md` for valid
vs contaminated experiments; **`v3event01` is INVALID evidence**.

## NEWEST — clean visual observations, self-supervision and real events (Nov 2, 2026)

**Read `docs/VISUAL_SELF_SUPERVISION.md` BEFORE the next feature or training
run.** Connected Gentoo Shell `/workspace` has GPU CUDA and direct Git;
keep using the shell, not the GitHub connector. All previous ignored worlds,
checkpoints, datasets and user files must remain intact. The active mission
remains from-scratch learned Any% on unseen Minecraft Java seeds, not scripted
completion. **No unrestricted first-log skill has been demonstrated.**

The original actual 84x84 RGB observations contained large **red MCio
command-chat overlays**. `vision_settings.ensure_clean_visual_options` now
atomically sets `chatScale:0.0` with `chatVisibility:0` in dedicated local
`DemoInstance/options.txt` before MCio launch: keeps HUD and normal camera
controls. Do NOT use `chatVisibility:2`: Minecraft teleports then consistently
reports pitch/yaw=0, invalidating seeded resets. `sendCommandFeedback false`
did not fix error chat. The user's own Minecraft options are untouched.

New `experiments/visual_dynamics_collect.py` captures the agent's OWN random
normal Minecraft action/RGB pairs only, locally, with distinct train/validation
scenario+action RNG seeds. 13 valid training episodes (507 before/after pairs)
and 4 held-out validation episodes (136 pairs) were obtained; collector
initially planned 14 train episodes but timed out after 13, so **do not claim
14**. New fully random-weight CUDA inverse visual predictors achieved held-out
**camera 42.6% (majority 23.5%)**, attack **84.6% (baseline 52.2%)**,
movement **15.4% (baseline 14.7%)**. Camera-locked movement-only dataset
achieved **38.5% vs 14.7%**; a causal GRU action-effect diagnostic achieved
movement20.6%, camera44.9%, attack81.6% on the mixed dataset. **Offline
self-supervised diagnostics are NOT yet a recurrent RL policy or first-log
success.** `--vision-init-tag` in V2/V3 training supports self-collected
raw-RGB CNN initializer, not outside pretrained models; downstream learning
advantage remains unverified.

Minecraft mining statistics (`minecraft:mined` and `minecraft:picked_up`) now
have private evaluator caching at EACH MCio game tick in `MCioAgentBridge`;
policy `AgentView` stays only RGB. **CRITICAL BUG HISTORY:** `v3event01`
training result INVALID: per-episode stats cache began at 0 and then received
cumulative absolute mine counts, producing fake 1+2+3+4+5+6=21 mined counts.
DO NOT treat `v3event01` as clean evidence or warm start from it. Fixed with
`CumulativeMinedTracker` scoped to a shared world arena; verified actual
mine/no-attack/mine **1/0/1** delta across resets in `statfix01`. A clean
`v3eventfixed01` warm-start from prebug checkpoint `v3clean01` ran 10 CUDA
training episodes, **3 real mined oak blocks, 0 inventory logs**. Frozen
held-out paired restricted stationary audit: `v3clean01` mined4/4, picked0/4;
`v3eventfixed01` mined4/4, picked2/4. Fully unrestricted near is still 0/4
for both. Different training budgets prevent causal attribution to reward.
New natural seed31415927 with identical spawn: random and frozen V3 each
0/240-tick logs; pipeline verified, no general-world transfer proven.

`--decision-ticks 12` is separately schema-tagged V3 (6 is V2); old V1/V2
checkpoints remain supported. `v3clean01` started 28 episodes but timed out
after 15 recorded; LAST CHECKPOINT was after 14. Never claim 28 complete.
An unexplained MCio I/O **slowdown to ~3.9s per 300 game ticks**, independent
of Torch, viewport, chat scale or 6/12 action interval, is under investigation.
Profiling shows MCio game I/O, NOT GPU inference (~0.09s). Do not blindly
change host services, driver, sandbox mounts or GPU settings. When 90-second
bridge calls time out, inspect valid JSONL+atomic checkpoints and terminate
ONLY orphaned processes owned by the experiment.

**NEXT:** pursue evidence-backed sample-efficient visual motion modeling,
credit assignment and genuine unrestricted first-log acquisition. Compare
against matched scripted 8/12 controlled-near baseline, then unseen genuine
natural survival seeds; no infinite PPO blindly. Use own RGB/action data,
not human/external pretrained models. Document every result in the ledger,
commit+push code/docs, NEVER Git-add local models or world saves.

## NEWEST — factorized visual policy and real learning-efficiency audit (Nov 2, 2026)

**Read `docs/FACTOR_POLICY_REVIEW.md` BEFORE any more training.** V2 is a
new, checkpoint-INCOMPATIBLE 888,750-weight CUDA RGB PPO actor with 3
categorical heads (movement/look/attack) controlling ordinary vanilla
Minecraft inputs simultaneously every 6 game ticks. V1 code/checkpoints
are preserved. The new `experiments/continuous_mine_probe.py` demonstrated
a real log after 20 consecutive 6-tick holds; no hidden commands during
scored play. Stale-inventory contamination at reset was found by the positive
controls, then FIXED: `VariableLogArena.reset` waits for both correct pose
and zero wood inventory, with 2 extra settled ticks.

**Evidence:** fair/simple same near-scenarios (seed 3737) 12-case control:
attack-then-forward **8/12**, random **0/12**, privileged geometry oracle
**11/12** (NOT a fair policy). Frozen V2 models `factorv2pilot01`,
`attackrl01`, `aimnear01` each **0/12** on these EXACT same cases. A
restricted attack-only curriculum produced **3/32** training and **2/4**
held-out logs when look/movement were disabled, but it does **not** transfer
to unrestricted play. `aimnear01` sequential and `mixstable01` interleaved
curriculum runs also failed fully unrestricted near tests; learned policies
often endlessly turn or walk away. Do not claim that more PPO epochs fixes
this or that CUDA speedup implies Minecraft skill.

New `experiments/joint_log_factorized.py` supports staged or interleaved
curriculum, privileged SCALAR shaping only in the placed-log training lab,
and optional `--save-rollouts`. Rollout data are validated, compressed,
versioned, and stored in **ignored local** `runs/joint_factorized_TAG/`,
containing only agent RGB, normal actions, scalar rewards/values,
head masks and terminal flags; **NO hidden coordinates**. Do not blindly
reuse stale PPO rollouts as new on-policy experience. `recordpilot01` did
real CUDA rollout, saved/read a 40-transition 135670-byte local dataset.

Frozen V2 can now be evaluated with `natural_world_benchmark.py --modes
joint_factorized --factorized-model TAG`. Actual natural seed 246813579,
all modes start (16.5,124.0,0.5), 0/360-tick logs for random,
forward-attack and V2 `mixstable01`. Checkpoints hashed in output. This is
pipeline proof only, NOT world transfer or speedrunning competence.

**Next scientifically justified step:** use held-out paired logs and
recorded agent visual data to investigate perceptual features, temporal
memory and more sample-efficient off-policy/representation learning BEFORE
scaling new GPU training. Keep actual game success as the gate; if basic
first-log learning fails, DO NOT proceed to piles of crafting subskills.
Never overwrite existing ignored files; Git fetch/commit/push via isolated
Gentoo Shell `/workspace`, no GitHub connector necessary.

## Latest: CUDA works after bridge restart (2026-11-02)

User created host NVIDIA UVM device nodes and restarted Gentoo Shell. GPU
preflight **passed**: `cuInit(0)=0`, CUDA tensor allocation on RTX 4070
SUPER and local PyTorch 2.15.0+cu132. `tools/benchmark-training.py` measured
**879.69 CUDA vs 128.99 CPU updates/s** for synthetic batch 64 (6.82x);
this is model training only, not Minecraft environment speed.

Actual CUDA real Minecraft `cuda01`: 12 training episodes x220 ticks,
3 GPU PPO updates, 4 frozen held-out x220 ticks, **0/12** and **0/4** log
acquisitions. All completed, local checkpoint/JSONL saved. Frozen policy
used only three long actions per episode, suggesting action-duration collapse
toward the fixed 80-tick mining macro; investigate before expensive long
training. Current CUDA works but UVM devices may need recreation after
future host boot. See `docs/GPU_ACCELERATION.md` and `docs/JOINT_LEARNING.md`.

## Freshest improvement: durable learning checkpoints

`joint_log_ppo.py` now atomically writes local `.pt` checkpoints including
Adam moments and source metadata. For continuation, use `--init-model` with
**new** `--run-tag`, `--train-seed` and `--policy-seed`; this is new-world
warm-start, not exact mid-episode replay. `warmcheck01` CPU smoke completed a
100-tick episode / second PPO update and saved optimizer state; **0 logs**,
scenario RNG repeated earlier seed in this *pilot*, so not fresh evidence.
Later code now rejects repeated train/policy RNG seeds. See
`docs/JOINT_LEARNING.md` for canonical commands. Never delete or overwrite
older saved checkpoints to make a rerun work.

## 2026-11-02 additional joint PPO + NVIDIA status (latest)

**Read `docs/JOINT_LEARNING.md` and `docs/GPU_ACCELERATION.md` before continuing.**
A new 2-RGB-frame CNN actor-critic (888,236 parameters) trains random weights
from full collected Minecraft trajectories with timed-action GAE / clipped PPO.
11 actions jointly select movement/look/jump/attack; 80-tick mining hold is
still a **scripted motor primitive**, not learned duration. Controller resets
can plant logs and supply scalar geometric shaping ONLY in the **training
arena**. Frozen natural-world evaluation is a separate `joint_ppo` mode.

Verified real game pilot `jointsmoke01`: 2 CPU-train + 2 held-out episodes at
140 ticks; **0/2 success each**, but actual Minecraft rollout, PPO update,
local checkpoint and independent evaluation all completed. Another natural
seed `161803398` ran frozen `jointsmoke01` for 120 ticks with 0 logs. **Do NOT
claim successful transferable learning or Any% progress from these smoke tests.**
Existing old checkpoints and saves are intact; local outputs stay ignored.

GPU probe inside the same nested Gamescope/XWayland GLX display actually
reported `NVIDIA GeForce RTX 4070 SUPER` OpenGL 4.6 / driver 595.104.02,
**not CPU llvmpipe**. System game simulation still uses CPUs. PyTorch
2.15.0+cu132 is installed **project-locally** in `.venv` and CPU forward/
backward was tested. Host `nvidia_uvm` module was loaded but **current bridge
still lacks `/dev/nvidia-uvm`**; `cuInit(0)=999`, `torch.cuda.is_available()`
false. User received root host-side `mknod` commands and must restart the
bridge to bind devices once created. DO NOT claim or silently assume CUDA
training worked; use `tools/gpu-preflight.py`, and `--device cuda` for strict
failure until genuine GPU tensor allocation succeeds. Never widen filesystem
sandbox to fix CUDA. Original shell already passes UVM devices when available.

## 2026-11-02 priority correction and direct shell/Git

**Use Gentoo Shell `run_shell` for repository work when connected.** It
provides scoped `/workspace` access, working `.venv`, MCio, real Minecraft,
network and ordinary `git` fetch/commit/push via a **repo-scoped** SSH deploy
key. Run Git directly inside the shell rather than requiring the GitHub
connector. Always inspect `git status` first, preserve local ignored worlds,
model checkpoints, logs, private keys and user modifications. Do not display
private keys. Push commits normally and verify `git status --short --branch`.
If shell is disconnected, use GitHub connector only if explicitly available;
never pretend to have tested a runtime or pushed a commit when unable to do so.

**Read [docs/COURSE_CORRECTION.md](docs/COURSE_CORRECTION.md) before new
feature work.** Current highest priority is a defensible generalization
benchmark on independent natural survival worlds, a reusable RGB-only
observation/action boundary, longer-horizon learning, and contiguous wood-to-
crafting progression. Do not add narrow skill scaffolding without a clear
route to a better held-out first-log or end-to-end survival score. Do not
replace the user's learn-from-random-weights goal with a scripted planner.

**New code:** `src/gmaincraft/agent_interface.py` (only the `AgentView`
immutable RGB array is passed to visual policies; `AgentAction` contains only
normal key/button/mouse input), `src/gmaincraft/natural_eval.py`, and
`experiments/natural_world_benchmark.py`. The controller may privately inspect
inventory to score after real gameplay. There is no admin/cheat channel in the
policy action contract. The interface is not an untrusted-code security
sandbox. Pilot natural-world result was 0 logs in 100 ticks for all 4 modes;
old source spawnRadius=10 made that smoke test **unpaired**. New world
preparation fixes spawnRadius=0, requiring another actual test. Never treat
old linear policies as a scalable fully autonomous agent; the `legacy_probe`
mode explicitly uses scripted mining timing and phase switching.

## Start every new task this way

1. Read `README.md`, **`TODO.md`**, `docs/HANDOFF.md`, `docs/ARCHITECTURE.md`, and (for measured claims) `docs/EXPERIMENTS.md` from the current branch. Inspect relevant source files, their imports, and recent commits/PRs before proposing changes. Do not claim to have read them if the connector is unavailable.
2. Determine the *last verified* milestone, the *current pending* test, and the next unchecked item in `TODO.md`. Distinguish user-supplied actual run output from forecasts, synthetic tests, and code review.
3. Take user-provided run results seriously; extract meaningful summaries and errors. Do not assume a test passed merely because source exists, syntax compiles, or a toy simulation worked.
4. Make the smallest coherent code change. Preserve confirmed behavior and keep test worlds, production inputs, and model weights safe.
5. **Update `TODO.md` in the same change whenever progress, next steps, blockers, or scope change.** Update `docs/HANDOFF.md` whenever the restart instructions or latest status change. Update `docs/EXPERIMENTS.md` when a run is actually measured or an earlier claim is corrected. Update `README.md` when user-facing commands or layout change. These documents must not contradict each other.
6. Verify with appropriate unit/syntax tests; explicitly note runtime tests you cannot perform on the user's Gentoo machine.
7. Commit/source-control the changes on GitHub when authorized and give the user `git pull --ff-only` or other precise Git commands to bring their working copy up to date. Do not default to scattering downloadable Python attachments.

## Goal and honest scope

Long-term: Minecraft Java Any% agent that learns from random weights, discovers behavior through experience, and generalizes to unseen random seeds, without a scripted solution. The current experiments are **scaffolding**, NOT a pretrained or general speedrunner.

Current proven stages (2026-11-02): real MCio controller, benchmarks, reward checks, numerical camera-relative navigation (3 independent policy/train RNG seeds, each 50/50 on held-out and extra yaw-target tests); **learned RGB-feature mouse/ATTACK** (2 fresh 500-episode training runs, fresh-world audit 15/20 and 18/20 real mined oak logs, untrained 0/20); **learned W/A/S/D pickup** (2 fresh 150-episode training runs, separate zero-drift spawned-item audit: untrained 2/32, random 16/32, always-W 21/32, learned p01 30/32, learned p02 24/32). **Two-policy real mined-log combination** (two separately seeded camera-angle audits, 56 trials/model): untrained movement 21/56, fixed W 49/56, learned p01 43/56, learned p02 51/56. The two neural stages are NOT jointly trained and pickup chooses movements only **after** an agent-selected ATTACK; attacking is held for a fixed 100 ticks. Fixed-centered trunk in flat copied world, handcrafted brown-color RGB features, privileged controller-only scalar shaping and reset commands. Better than always-W on off-center spawned-item pickup, **not a proven advantage over W in integrated centered-tree tests**. No learned general tree recognition, cross-world generalization or Any% gameplay. **NOW:** a variable-player/trunk joint visual PPO prototype exists and is smoke-tested, but first-log success remains unproven; next validate real learning on held-out natural worlds and CUDA training. Never reward human-like camera aesthetics. Preserve local old navigation/aim and new pickup checkpoints. See `docs/EXPERIMENTS.md` and `TODO.md`.

Important distinctions:
- **Camera rules (user decision 2026-11-02):** never reward natural-looking camera movement, looking forward, staying level or other aesthetics. Allow strange angles, walking backward/sideways and fast turns if they help. Camera orientation matters only because it changes perceived pixels and determines actual vanilla raycast interactions; the agent must still aim to break/place/interact, within vanilla reach. Use normal MCio mouse/cursor inputs for agent actions, never controller teleport commands for agent turning.
- The `navigation_learner.py` policy receives a numeric goal direction and assumes a fixed camera yaw. `orientation_learner.py` also gets a privileged goal vector in camera-relative coordinates. Both are carefully limited **curriculum experiments**, not vision or real discovery.
- The new `log_pickup_learner.py` uses a **controller-summoned dropped item** only to train/evaluate the movement subskill; it must not be counted as natural wood gathering. The combined `log_mine_and_pickup_audit.py` uses a controller-placed *block* and real vanilla mining and pickup, no `/give` or scripted movement during scored phases. Both RGB encoders are **hand-engineered** and the trainer gets **privileged scalar progress**, not hidden coordinates as policy observations.
- The reward probe uses Minecraft `/give` to test its detector, not to assist an agent. Controller-admin commands `/tp`, `/fill`, `/setblock`, `/effect` are for episode setup/cleanup; a learning policy must never issue them during scored gameplay.
- A persistent-process **soft reset** restores only selected player properties and does not undo mined blocks, mobs, dropped items, exploration, or RNG/world state. Do not call soft-reset trials fully independent random worlds.
- A network's randomly initialized weights and actual Minecraft rollouts are central requirements. Don't replace experimental RL with a hardcoded policy/planner while claiming learning.
- A single pre/post evaluation on 20 targets is an encouraging smoke test, **not** statistical generalization proof. Use multiple runs/seeds and properly held-out scenarios before strengthening claims.
- The ultimate task is much harder than these prototypes; single-GPU training feasibility, sample complexity, and end-to-end completion remain uncertain.

## Environment and filesystem constraints

Verified user's machine: Gentoo/OpenRC, Plasma Wayland, XWayland compatibility path, NVIDIA RTX 4070 SUPER 12GB, i7-14700KF, 32GB RAM, Python 3.14.10. User prefers project-local setups in `~/Documents/minecraft-discovery`, no `sudo` (if elevated action ever needed, explain `su -c`), no background daemons, and no Portage/desktop integration unless explicitly approved. Do not request access to other local directories.

- Repository: `https://github.com/polskiftw/gmaincraft`, default branch `main`. **Normal Git works directly from shell**: HTTPS fetch and a repository-scoped write deploy key via `ssh.github.com:443` for pushes. Prefer `git fetch`/`pull --ff-only`, local commits and `git push` over relaying Git blobs through connectors. Credentials exist only under git-ignored `mcio-data/git-auth/`; never expose private key data or commit it.
- Local project root: `~/Documents/minecraft-discovery` (an existing working directory, may be attached to Git).
- Source scripts: `experiments/*.py`. Reusable code moves into `src/gmaincraft/` only when it earns its abstraction; tests go in `tests/`.
- Local-only instance/template: `mcio-data/instances/DemoInstance`, source `DemoWorld`. Scripts overwrite **only** their explicitly named disposable trial worlds; never change template `DemoWorld` or PrismLauncher worlds.
- MCio controller: `mcio_ctrl==1.5.2`, Minecraft 1.21.3 Fabric, MCio mod 1.6.1+1.21.3 observed in logs. Keep `cbor2>=5.7.1,<6` because CBOR 6.x has a known incompatible decoding behavior; document before changing.
- For the original non-sandbox Gentoo desktop, use per-process `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 ...` to run MCio's GLFW viewer through XWayland; don't change global graphical settings.
- **Gentoo Shell MCP sandbox (verified 2026-11-02):** isolated `/workspace`, host home hidden; **network now deliberately shared with host to permit outbound Git/DNS; LAN/host-local services are potentially reachable**; read-only NVIDIA sysfs, GPU/DRM devices, one Wayland socket, private localhost-only `/etc/hosts`; accelerated Minecraft launched through local nested Gamescope/Xwayland with `--visible`. The local-only `mcio-data/xwayland-sandbox-wrapper.sh` sets `-ac` for nested Xwayland only. **Use `tools/run-mcio-viewer.sh`** to prevent physical desktop mouse motion/click/scroll from reaching the nested game's camera; it compiles a project-local SDL2 shim and unsets `LD_PRELOAD` before spawning the Python/Java game child. Synthetic SDL2 input test and MCio 10/10 smoke succeeded; live physical hover verification is still pending. See `docs/DEVELOPMENT.md`. Gamescope exits with code 139 after the experiment child shuts down; the Python experiment completed, logged and saved a valid checkpoint first. Direct MCio headless failed with EGL/GLFW in this sandbox. Do not weaken sandbox boundaries or expose other host folders to fix these issues.
- Running headless on a separately configured host is allowed, but still needs a working graphics stack; benchmark active simulation versus full launch/exit overhead.
- Model checkpoints, worlds, local logs, large videos, profile files and binary outputs are not source code and remain Git-ignored. Don't add them or API tokens to the public repo. Never commit raw huge console logs full of irrelevant Java stack traces.

## Development standards

- Keep runnable entrypoints documented with exact commands and flags. Avoid breaking the existing root-relative runtime paths when moving files.
- Prefer simple, inspectable, reproducible code; use explicit RNG seeds and record Minecraft/mod/Python versions, training/evaluation seeds, action spaces, reward definitions, sample counts and end-to-end timings.
- Keep evaluator separate from trainer; freeze policy during evaluation; use held-out targets/world seeds and ensure train/test leakage checks.
- Start with source-only or isolated unit tests; full Minecraft launches and GPU integration tests must be labeled and run only when requested/available. No CI job should require Mojang downloads.
- Never silently change the objective from learn-from-scratch to scripted speedrun or from fair observations to hidden-world state.
- Treat user feedback and factual results as primary. If a GitHub write fails, report it; do not claim the repo was updated.

## Completion checklist for every substantive change

- [ ] Code added/updated in the correct directories, not only a chat attachment.
- [ ] Syntax/targeted unit tests run or limitations explicitly stated.
- [ ] `TODO.md` updated in the same commit to reflect the new state.
- [ ] `docs/EXPERIMENTS.md` updated if actual measured outcomes changed.
- [ ] `docs/HANDOFF.md` updated for the next fresh-chat continuation.
- [ ] Documentation/commands updated if paths, dependencies, or flags changed.
- [ ] Commit/PR linked and user given safe local Git synchronization commands.

**In a new chat:** do not ask the user to retell the entire project. Read the repository and request only a specific missing run result or environment detail.
