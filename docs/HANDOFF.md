# Handoff for a brand-new ChatGPT conversation

**Last verified: 2026-11-02.** Canonical source: [polskiftw/gmaincraft](https://github.com/polskiftw/gmaincraft), branch `main`. **The orientation experiment has now been run successfully in real Minecraft**, so do **not** ask the user to supply an orientation log again. See `docs/EXPERIMENTS.md` for the exact run evidence and caveats.

## First actions in a new chat

1. Read current `AGENTS.md`, `TODO.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, and `docs/EXPERIMENTS.md`; inspect the source of any experiment you modify.
2. Confirm the latest verified milestone: 300 Minecraft training episodes, camera-relative movement policy **9/50 before -> 50/50 after** on 50 held-out target/yaw scenarios, **50/50** on an additional unseen target/yaw set. Original policy fixed yaw **50/50**, randomized yaw **24/50**. Training CSV: **299/300 wins**. Learner time: **30.16s**. Only one training seed/world.
3. **NEXT:** parameterize independent RNG seeds (policy initialization, training, held-out evaluation, extra evaluation) and separate output paths **before** repeating the experiment. Do not overwrite the validated `checkpoints/orientation_policy.npz`.
4. Replicate with several independent seeds; capture success/variance/steps/compute costs, label all same-world arena tests honestly. After replication, teach actual camera-turning action and ultimately move away from privileged numeric goal vectors.
5. Commit code/tests/status docs together, verify real Minecraft runs when available, and give the user a `git pull --ff-only` command.

## Goal and scope

Long-term from-scratch reinforcement-learning agent for Minecraft Java Any% on unseen random world seeds, without scripted speedrun logic or privileged hidden-world inputs. **We are not close to a general visual speedrunner.** These are controlled training curricula using known numeric goal direction, flat arena and W/A/S/D actions only. Randomized camera yaw is set by the experiment controller at spawn; policy has no mouse/camera turning action.

## Verified runtime and infrastructure

- Gentoo/OpenRC, Plasma Wayland, NVIDIA RTX 4070 SUPER (driver 595.104.02), i7-14700KF, 32GB RAM, Python 3.14.10.
- Project root on host: `~/Documents/minecraft-discovery`, available via Gentoo Shell MCP as `/workspace`. Its Bubblewrap sandbox **hides `/home`**, keeps network namespace loopback-only and mounts selected GPU/DRM/sysfs resources plus one Wayland socket.
- Minecraft Java **1.21.3**, Fabric MCio **1.6.1+1.21.3**, `mcio_ctrl==1.5.2`, `cbor2==5.9.0`; MCio-managed instance `mcio-data/instances/DemoInstance` and template `DemoWorld`.
- The MCP sandbox launches accelerated game **through nested Gamescope with local Xwayland wrapper**; see `docs/DEVELOPMENT.md`. Direct NVIDIA/MCio headless EGL failed here. Nested Gamescope exits with SIGSEGV **139 after the experiment child exits**, although orientation results/checkpoint/CSV were completed and validated.
- **Local-only, ignored assets:** `checkpoints/navigation_policy.npz`, `checkpoints/orientation_policy.npz` (validated saved model), `logs/orientation_training.csv` (300 rows), local world saves, and the custom sandbox launcher/wrapper under `mcio-data/`. They are **not on GitHub**. Do not delete these or copy personal Minecraft saves.
- Prior verified foundations: MCio actions/observations, clean-world-copy experiments, throughput benchmarking, local soft resets (world mutations persist), controller item-novelty detection, original navigation 7/20 -> 20/20 after 120 episodes/12.8s training.

## Orientation evidence (2026-11-02)

| Evaluation | Wins | Final distance (mean) | Ticks among wins (mean) |
|---|---:|---:|---:|
| Original fixed yaw | 50/50 | 0.72 | 25.6 |
| Original randomized yaw | 24/50 | 7.03 | 41.2 |
| New random initial weights | 9/50 | 7.22 | 71.3 |
| New AFTER 300 training episodes | 50/50 | 0.66 | 25.6 |
| New on extra unseen target/yaw set | 50/50 | 0.65 | 26.0 |

Policy seed 2718; training scenarios seed 81003; evaluation scenario seed 60211; extra seed 99501; action RNG seed 771. Trial world `MDL_OrientationLab_v1`; never overwrote template `DemoWorld` or original navigation checkpoint. Full protocol and SHA-256 of artifacts are in `docs/EXPERIMENTS.md`.

## Safety and scientific integrity

- Minecraft admin commands (`/tp`, `/fill`, `/setblock`, etc.) belong solely to the privileged experiment controller for resets/arena setup, never to scored policy actions.
- A randomized target/yaw in the **same** saved-world arena does **not** demonstrate cross-seed survival or visual recognition. Do not conflate in-game learner time with total process start/stop time.
- The 100% evaluations are two 50-case sets after a **single** trained model on a controlled curriculum; quantify multi-seed robustness and separate trainer from frozen evaluator before stronger claims.
- Original host checkpoints, worlds, personal data, and local raw logs remain git-ignored; do not add them to the public repository. No host root/sudo access or system services should be required.
