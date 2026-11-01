# gmaincraft

**A Minecraft reinforcement-learning research playground.** The long-term ambition is an agent trained from randomly initialized weights that can complete Minecraft Java Edition Any% (defeat the Ender Dragon and reach the exit) efficiently on unseen random world seeds. It should learn from experience, rather than follow a hand-scripted speedrun or peek at forbidden game state.

**Current reality:** the MCio control interface, benchmarks, reward detector, and first simple neural navigation experiment have worked on a real Minecraft 1.21.3 client. We have **not** trained a general Minecraft player or completed a speedrun. The camera-orientation experiment exists but its real-run results are **pending** as of 2026-11-01.

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
│   └── orientation_learner.py   Camera-yaw generalization experiment
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

### Run an experiment

After the MCio demo has finished and you are back at a shell:

```bash
cd ~/Documents/minecraft-discovery
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/navigation_learner.py --visible
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/orientation_learner.py --visible
```

Run **one** MCio experiment at a time: they use the same local action/observation ports and disposable Minecraft worlds. The orientation experiment requires a local `checkpoints/navigation_policy.npz` from an earlier navigation run. It does **not** include that checkpoint in Git. Omit `--visible` for headless NVIDIA mode.

Each experiment is a direct-entry Python prototype, not a production package. Its runtime paths are relative to the repository root even though the scripts live in `experiments/`.

## Verified milestones

- Random-action baseline controlled the actual game and restored the starting world between episodes.
- Full-restart benchmark: 195.0 active steps/sec windowed; 185.5 active steps/sec headless (two small trials each).
- Persistent-process benchmark: 284.5 active steps/sec over 5,000 steps; 123.5 steps/sec measured end-to-end; soft resets around 0.08–0.11s. **Soft resets do not restore the world.**
- First-seen item reward detector passed all five controller-injected item tests. **Not a learned item-acquisition policy.**
- Simple navigation learner: 7/20 successes before training, 20/20 afterwards on the same 20 held-out targets, after 120 training episodes in 12.8 seconds. Fixed yaw, privileged goal direction, no vision.

All measurements are preliminary and hardware/setup specific. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the actual scope and caveats.

## Safety, scope, and version control

`mcio-data/`, `.venv/`, `logs/`, `checkpoints/`, recordings, Minecraft worlds and model binaries are ignored by Git. The Python source and documentation are the source of truth; **Git alone cannot recreate an uncommitted trained checkpoint or world save**. Preserve those locally or arrange an explicit backup if needed.

No daemons, startup services, `sudo`, Portage changes, or unsolicited filesystem access are needed. Minecraft `/give`/`/tp`/`/fill` commands are for controller-led setup and testing, not for the learning policy. Avoid accidentally turning privileged game state into agent observations.

For exact commands, local migration, unit tests, and known Linux compatibility notes, consult [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
