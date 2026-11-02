# Handoff for a brand-new ChatGPT conversation

**Last verified: 2026-11-02.** Canonical code: [polskiftw/gmaincraft](https://github.com/polskiftw/gmaincraft), branch `main`. We have **three independently initialized/trained real Minecraft camera-relative orientation policy runs** (details in `docs/EXPERIMENTS.md`). Do **not** ask the user for the old orientation run log again.

## First actions for the next assistant

1. Read `AGENTS.md`, `TODO.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/EXPERIMENTS.md`, inspect relevant code/tests and recent GitHub commits.
2. Recognize the last verified milestone: all three camera-relative numerical policies scored **50/50 after training and 50/50 on an additional unseen target/yaw set**. Each trained for 300 real Minecraft episodes with distinct policy/train/eval/extra/action seeds. Each training CSV shows **299/300 wins**.
3. **NEXT:** implement **learned visually guided wooden-log acquisition**, starting with camera yaw/pitch and vanilla mining/collection. **User explicitly does not want human-looking camera behavior**; aim/turn only where perception and the game mechanics reward it, without cosmetic gaze penalties. Keep randomized held-out episodes and report genuine observed inventory acquisitions. The input probe already proved real mouse deltas and log pickup; do not mistake it for trained behavior.
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

## Updated interaction curriculum and honest status

- **User camera rule (2026-11-02):** camera pose/appearance should never receive human-likeness rewards; the agent may stare at sky or move sideways/backward, but must still use vanilla first-person line-of-sight, crosshair reach and mouse-based interactions where needed.
- `experiments/log_interaction_probe.py`: **VERIFIED scripted vanilla log mining/pickup in actual Minecraft**, not RL. In a *new disposable copied world* (`MDL_LogInputProbe_pilot01`), the controller placed a log column, then no commands were sent after staging. MCio mouse deltas +120,+50 pixels changed yaw +18.00° and pitch +7.50°, inverse deltas restored camera. Holding left attack 100 ticks then walking forward 11 ticks yielded **1 real oak log** in observed inventory; 20.15s total launch+test. Gamescope exit 139 occurred only after printout. No `/give`. Four new pure-helper tests were added.
- **NEXT:** learn camera alignment and actual attack/pickup from RGB frames rather than using numeric goal coordinates, ensuring controller reset commands are never policy actions. Report if only input-path or visual preprocessing is demonstrated, not learned behavior.

## Runtime and local environment

- Gentoo Linux/OpenRC, Plasma Wayland, i7-14700KF, RTX 4070 SUPER driver 595.104.02, 32GB RAM, Python 3.14.10.
- Host project `~/Documents/minecraft-discovery` is exposed as `/workspace` through a custom Gentoo Shell MCP Bubblewrap sandbox. Host `/home` remains **not exposed**. Network namespace is now intentionally **shared with host** for GitHub HTTPS/SSH access, including possible LAN and host-loopback connectivity. Repo-specific SSH write deploy key under ignored `mcio-data/git-auth/` allows normal shell `git push`; HTTPS fetch also works. GPU/DRM devices, selected read-only sysfs and one host Wayland socket are available.
- MCio Python controller `mcio_ctrl==1.5.2` with `cbor2==5.9.0`, Minecraft Java **1.21.3**, Fabric MCio **1.6.1+1.21.3**, Java 21 MCio runtime. Instance/template are `mcio-data/instances/DemoInstance` and `DemoWorld`. MCio ZMQ ports 4001/8001 require a working private localhost `/etc/hosts`.
- Inside MCP, Minecraft uses nested NVIDIA Gamescope/Xwayland. **Use `tools/run-mcio-viewer.sh`**, which builds the SDL2 no-host-mouse library and filters real mouse motion, clicks and scrolling (not MCio agent actions), keeping your pointer free. Example unique disposable test: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/orientation_learner.py --visible --run-tag viewer_demo --episodes 8 --eval 5`. Local Xwayland wrapper has `-ac` only for its private X11 socket. SDL2 simulated-input test and Minecraft MCio smoke 10/10 passed; physical desktop pointer hover verification is pending. See `docs/DEVELOPMENT.md`.
- **Gamescope exits SIGSEGV 139 after each completed Python run**; test results/checkpoints/CSVs were written and validated first. Direct MCio NVIDIA headless without Gamescope failed with GLFW/EGL in this sandbox.
- All checkpoints, saves, Python virtualenvs, local MCP launcher helpers, huge raw logs and model files are git-ignored; only source/docs/tests belong on public GitHub. Never access other host folders, alter global Gentoo settings or add a daemon for these experiments.

## Scientific boundaries

- Privileged arena controller may use `/tp`, `/fill`, `/setblock`, `/effect` **only for setup/cleanup**. Learned policy is restricted to W/A/S/D so far, receives scalar reward and an exact camera-relative numeric target vector. Random starting yaw is controller-set, not action-controlled.
- The 3 independent seed sets are different *numeric RNG seeds*, not different Minecraft world seeds. All three use a small flattened arena. These are strong curriculum checks but do not prove survival, perception, crafting, Nether/End progression or cross-world generalization.
- A soft reset does not fully restore block/entity/RNG world state. Preserve saved checkpoints and `DemoWorld` across future experiments; avoid overwriting known-good assets.
