# gmaincraft

**A Minecraft reinforcement-learning research playground.** The long-term ambition is an agent trained from randomly initialized weights that can complete Minecraft Java Edition Any% (defeat the Ender Dragon and reach the exit) efficiently on unseen random world seeds. It should learn from experience, rather than follow a hand-scripted speedrun or peek at forbidden game state.

**Current reality (2026-11-02):** Minecraft Java 1.21.3 + MCio and three seeded numerical-navigation runs are verified. Two random-initialized, RGB-feature mouse+ATTACK models learn to aim at a controlled oak trunk (audited 15/20 and 18/20 real Minecraft logs). Two independently trained RGB-feature W/A/S/D policies learn to approach a **dropped** oak log: in a fresh 32-case offset-item audit, the best reaches **30/32**, versus **21/32** for just holding W. Combining the learned aiming and collection policies yields **51/56 real mined-log inventory pickups** over two fresh camera-angle sets for pickup model `p02`, versus **49/56** with scripted W. This does **not** establish a meaningful advantage in the centered-tree setup, and it still uses hand-engineered pixel masks, controller-only scalar rewards, fixed 100-tick attack holds, a single flat-world template, and separate training for the two neural stages. No general tree recognition or Any% speedrunning.

This repository is the canonical home for code and continuity across ChatGPT conversations. Start with [AGENTS.md](AGENTS.md), [TODO.md](TODO.md), and [docs/HANDOFF.md](docs/HANDOFF.md).

## Repository layout

```text
gmaincraft/
├── AGENTS.md                    Agent workflow and rules for future chats
├── TODO.md                      Living, ordered project task list
├── docs/
│   ├── ARCHITECTURE.md          Design, fairness, and research direction
│   ├── DEVELOPMENT.md           Installation, commands, testing, migration
│   ├── EXPERIMENTS.md           Verified experiment history and limitations
│   └── HANDOFF.md               Copy/paste new-chat resumption guide
├── experiments/
│   ├── random_agent.py          Random actions + clean-world restarts
│   ├── benchmark_agent.py       Windowed and headless full-restart benchmarks
│   ├── persistent_benchmark.py  Fast soft resets in one Minecraft process
│   ├── reward_probe.py          Per-episode first-seen item rewards
│   ├── navigation_learner.py    First real in-game policy-gradient learning
│   ├── orientation_learner.py   Camera-yaw generalization experiment
│   ├── log_interaction_probe.py Vanilla mouse/mine/pickup instrumentation test
│   ├── log_visual_aim_learner.py RGB-feature policy learns look/attack in toy arena
│   ├── log_visual_aim_evaluator.py Fresh-world audited frozen camera policy
│   ├── log_pickup_learner.py     RGB-feature W/A/S/D pickup reinforcement learning
│   ├── log_pickup_evaluator.py   Fresh-world spawned-item pickup audit + controls
│   └── log_mine_and_pickup_audit.py Audited real mined-log two-policy combination
├── src/gmaincraft/              Home for reusable code as it matures
├── tests/                       Minecraft-independent unit tests
└── requirements.txt             Pinned MCio compatibility dependencies
```

`LICENSE` is the existing Apache-2.0 license, preserved from the original repository.

## Quick start

Prerequisites: Linux with a working OpenGL/NVIDIA configuration, Python 3.12+, and a Java-capable Minecraft installation. The verified machine uses Gentoo Linux, Plasma Wayland with XWayland, Python 3.14.10, an RTX 4070 SUPER, and an i7-14700KF. MCio downloads/uses a separate Minecraft Java **1.21.3** Fabric client; the user's normal PrismLauncher installation is separate and untouched.

For a **new directory**:

```bash
git clone https://github.com/polskiftw/gmaincraft.git
cd gmaincraft
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/mcio demo --mcio-dir "$PWD/mcio-data"
```

The initial MCio demo provisions `mcio-data/` and creates the `DemoInstance` and `DemoWorld` used as templates by these experiments. It may download the Minecraft client/runtime on the first run. Press Q in the demo's viewer to exit it.

**Already have `~/Documents/minecraft-discovery` from this project's earlier experiments?** Do **not** delete it or clone over it. Instead attach that existing directory to this repository using the safe commands in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). This preserves `mcio-data/`, `.venv/`, `checkpoints/` and `logs/`.

### Next learning goal: acquire a log, not a human-looking camera

Our camera rule is **functional, not cosmetic**. The agent need not look forward, keep its head level or move its view smoothly. It may face skyward while traveling; it must use legitimate mouse aiming and vanilla reach mechanics when breaking, placing, interacting or fighting. The intended curriculum is rendered-pixel observation → agent-selected camera/movement/attack → actual oak-log pickup. No hidden target coordinates as policy observations.

The **scripted mechanics probe** verified vanilla mouse and log mining. The **learned camera/ATTACK** stage now chooses where to look and when to attack from engineered screenshot features, and a **separately learned pickup** policy chooses W/A/S/D from item-appearance pixels. Both train with controller-only *scalar* shaping while the learned policies see no hidden target positions. In two fresh integrated tests, combining `rgb02` aiming and `p02` pickup obtained **51/56 genuinely mined logs** with *no scripted post-attack movement*; simple always-forward pickup scored **49/56**, so a meaningful improvement for the centered trunk remains unproven. Off-center dropped-log tests show a clearer advantage for learned movement. The camera is free to look bizarre as long as the game rules are followed. See [the experiment ledger](docs/EXPERIMENTS.md); the next work is varying trunk/starting positions, richer scenery and learned vision.

### Run an experiment

After the MCio demo has finished and you are back at a shell:

```bash
cd ~/Documents/minecraft-discovery
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/navigation_learner.py --visible
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/orientation_learner.py --visible
```

**For the Gentoo Shell MCP sandbox**, use `tools/run-mcio-viewer.sh` rather than calling Gamescope directly. It builds a project-local SDL2 shim that keeps the real desktop mouse **unlocked and unforwarded**: mouse motion/click/scroll events are filtered before reaching the nested Minecraft window, while MCio agent actions still work. Example disposable run: `tools/run-mcio-viewer.sh .venv/bin/python -B experiments/orientation_learner.py --visible --run-tag viewer_demo --episodes 8 --eval 5`. Test the shim alone with `tools/test-gamescope-mouse-filter.sh`. Physical hover behavior still needs user verification; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Run **one** MCio experiment at a time: they use the same local action/observation ports and disposable Minecraft worlds. The orientation experiment requires a local `checkpoints/navigation_policy.npz` from an earlier navigation run. It does **not** include that checkpoint in Git. The original host can use headless NVIDIA mode, but the isolated Gentoo Shell MCP environment was verified with **nested Gamescope + `--visible`**, not MCio direct headless mode; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Default reruns overwrite `checkpoints/orientation_policy.npz`. New reproducibility flags `--policy-seed`, `--train-seed`, `--eval-seed`, `--extra-seed`, and `--action-seed` require a unique `--run-tag` to protect that checkpoint. Tagged runs create separate worlds, checkpoints and CSVs, and refuse existing tags.

Each experiment is a direct-entry Python prototype, not a production package. Its runtime paths are relative to the repository root even though the scripts live in `experiments/`.

### Verify or repeat visual aiming

Inside the Gentoo Shell sandbox, use a unique run tag and the mouse-safe viewer. This creates a separate disposable world and outputs only git-ignored model/CSV data:

```bash
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_learner.py --run-tag rgb03 --episodes 500 --eval 14 --policy-seed 5503 --train-seed 7725 --eval-seed 8816 --visible
```

To **evaluate existing checkpoints without further training**, use another new tag:

```bash
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_visual_aim_evaluator.py --run-tag audit02 --models rgb01 rgb02 --eval-seed 8816 --eval 20 --visible
```

The evaluator creates a fresh copied world, clears dropped items before every case, verifies no inventory pickup without an attack, and requires a real observed oak log for success. These are still toy single-trunk tests, not independent Minecraft world seeds.

### Train/audit pickup and the full two-policy interaction

Use a unique tag for every run. In the MCP sandbox, all three examples launch the actual Minecraft game through the mouse-safe viewer; model files, logs, and worlds remain git-ignored:

```bash
# Train a NEW RGB-feature W/A/S/D policy on controller-summoned dropped logs
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_pickup_learner.py --run-tag p03 --episodes 150 --eval 20 --policy-seed 4403 --train-seed 6603 --eval-seed 8805 --visible
# Audit existing pickup policies against untrained, random, and always-forward
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_pickup_evaluator.py --run-tag dclean03 --models p01 p02 --eval-seed 8806 --eval 32 --visible
# Real mined-log combined test: agent selects camera+attack, then agent selects movement
tools/run-mcio-viewer.sh .venv/bin/python -B experiments/log_mine_and_pickup_audit.py --run-tag f03 --aim-model rgb02 --pickup-models p01 p02 --eval-seed 8818 --eval 32 --visible
```

The pickup-only trainer **summons a dropped item as a controller-owned test fixture**; only the *combined* audit obtains a real mined log. Re-running a used tag is refused. No model weights are included in the public repository.

## Verified milestones

- Random-action baseline controlled the actual game and restored the starting world between episodes.
- Full-restart benchmark: 195.0 active steps/sec windowed; 185.5 active steps/sec headless (two small trials each).
- Persistent-process benchmark: 284.5 active steps/sec over 5,000 steps; 123.5 steps/sec measured end-to-end; soft resets around 0.08–0.11s. **Soft resets do not restore the world.**
- First-seen item reward detector passed all five controller-injected item tests. **Not a learned item-acquisition policy.**
- Simple navigation learner: 7/20 successes before training, 20/20 afterwards on the same 20 held-out targets, after 120 training episodes in 12.8 seconds. Fixed yaw, privileged goal direction, no vision.
- Vanilla log interaction probe (2026-11-02): camera deltas worked, oak log mined via held left mouse button, item entered observed inventory after moving closer; 1 log verified, no `/give`. Controller placed trunk in disposable world; scripted test, NOT learned perception.
- **Learned W/A/S/D collection** (2026-11-02): two fresh 150-episode Minecraft trainings. On an independent 32-case random-offset dropped-item test: untrained 2/32, random 16/32, fixed-forward 21/32, p01 **30/32**, p02 **24/32**. The policy receives only engineered RGB pixel features; the trainer uses privileged item-location scalar progress, and items are controller-summoned fixtures.
- **Combined real mined-log policy stages** (2026-11-02): `rgb02` chooses camera/attack; pickup policy chooses movements, **no scripted W**. Over two frozen camera-angle audits (24+32 cases), untrained pickup 21/56, fixed W 49/56, trained p01 43/56, trained p02 **51/56** real Minecraft inventory log pickups. In centered-tree setups the W baseline is already excellent; two extra wins do not prove a robust advantage.
- **Learned camera/attack curriculum** (2026-11-02): two random-weight linear RGB-feature policies trained for 500 episodes each, chose look/attack, then earned **15/20 and 18/20 real oak logs** on the same independent 20-camera-angle set in a freshly reset/loot-cleaned arena, versus 0/20 untrained. A stale-item contamination in the initial evaluation was diagnosed and fixed before these audited results; see `docs/EXPERIMENTS.md`.
- Camera-relative orientation learner (**real Minecraft**, 2026-11-02): initial 9/50, 7/50, 7/50 across three independently initialized/trained numerical policies. All three reached **50/50 held-out and 50/50 extra unseen** target/yaw scenarios after 300 training episodes each (30.16s, 26.67s, 25.17s learner time; 299/300 training wins per run). Policy never moved the camera or saw visual goal pixels; privileged numeric goal direction was supplied.

All measurements are preliminary and hardware/setup specific. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the actual scope and caveats.

## Safety, scope, and version control

`mcio-data/`, `.venv/`, `logs/`, `checkpoints/`, recordings, Minecraft worlds and model binaries are ignored by Git. The Python source and documentation are the source of truth; **Git alone cannot recreate an uncommitted trained checkpoint or world save**. Preserve those locally or arrange an explicit backup if needed.

No daemons, startup services, `sudo`, Portage changes, or unsolicited filesystem access are needed. Minecraft `/give`/`/tp`/`/fill` commands are for controller-led setup and testing, not for the learning policy. Avoid accidentally turning privileged game state into agent observations.

For exact commands, local migration, unit tests, and known Linux compatibility notes, consult [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
