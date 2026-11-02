# AGENTS.md — project instructions for coding assistants

This file is the operating contract for anyone (including a future ChatGPT conversation) editing **polskiftw/gmaincraft**. Treat the GitHub repository, not the conversation transcript or an assumed local checkout, as the authoritative codebase.

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

Current proven stages (as of **2026-11-02**): MCio actions/observations, reset/speed benchmarks, reward plumbing, fixed-yaw learning, and camera-relative movement with randomized initial camera yaw. **Three independent model and training/evaluation RNG seeds** produced after-training 50/50 held-out and 50/50 additional unseen target/yaw outcomes each (300 episodes/run; 299/300 successful training episodes each). The policies still receive a privileged numeric goal vector, use only W/A/S/D and have **no learned camera control or vision**. The trials all use the same saved-world arena template, not different random Minecraft world seeds. Source now supports `--run-tag` and five seed flags, with 5 new unit tests. **NEW VERIFIED 2026-11-02:** real RGB-feature camera/attack RL from random weights, two independent 500-episode training seeds. On a fresh dropped-item-cleaned 20-angle test: untrained **0/20** oak logs, trained **15/20 and 18/20**, no-attack negative control zero. Actions: agent chooses four mouse-look directions or ATTACK; **a scripted W finisher still collects drops**. Policy receives ONLY 12 engineered oak-brown RGB features; controller angular reward is scalar-only and privileged. Stale item drops contaminated initial *unvalidated* tests; reset must run `kill @e[type=minecraft:item]` before every episode, and future evaluations must retain no-attack controls and actual inventory success. **NEXT:** learn movement/pickup, then robust learned visual perception; human-like camera aesthetics are NOT the objective. Preserve all three older orientation checkpoints plus both new RGB models (ignored locally). Check `docs/EXPERIMENTS.md` and `TODO.md` for exact results.

Important distinctions:
- **Camera rules (user decision 2026-11-02):** never reward natural-looking camera movement, looking forward, staying level or other aesthetics. Allow strange angles, walking backward/sideways and fast turns if they help. Camera orientation matters only because it changes perceived pixels and determines actual vanilla raycast interactions; the agent must still aim to break/place/interact, within vanilla reach. Use normal MCio mouse/cursor inputs for agent actions, never controller teleport commands for agent turning.
- The `navigation_learner.py` policy receives a numeric goal direction and assumes a fixed camera yaw. `orientation_learner.py` also gets a privileged goal vector in camera-relative coordinates. Both are carefully limited **curriculum experiments**, not vision or real discovery.
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
