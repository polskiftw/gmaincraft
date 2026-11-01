# Handoff for a brand-new ChatGPT conversation

**Last verified status: 2026-11-01.** The canonical code is in [polskiftw/gmaincraft](https://github.com/polskiftw/gmaincraft), branch `main`. The current outstanding item is the **real run result** from `experiments/orientation_learner.py`; it has not yet been supplied. Do not infer success.

## First actions for the next assistant

1. Open the current GitHub versions of `AGENTS.md`, `TODO.md`, `README.md`, this file, and `docs/EXPERIMENTS.md`. Inspect the script being debugged.
2. Ask for or process the user-provided orientation log; the user intends to bring it in the new chat.
3. Look for these output labels: `Original (fixed camera)`, `Original (random camera)`, `New BEFORE training`, `New AFTER training`, `New extra unseen cases`, `Learning time` and `New checkpoint`.
4. Compare before/after and the extra unseen scenarios without overclaiming. Investigate errors first.
5. Update `docs/EXPERIMENTS.md`, `TODO.md`, and this handoff with verified results. Implement the next prioritized milestone in a GitHub commit and provide `git pull --ff-only` instructions; do not rely on old chat-generated attachments.

A user can start the new chat with:

> Read my GitHub repository `polskiftw/gmaincraft`, especially `AGENTS.md`, `TODO.md` and `docs/HANDOFF.md`. We were about to review the orientation-learner result. Here is the run output: [paste].

## Project intention

Experimental agent that starts from random neural-network weights and eventually learns Minecraft Java Any% (kill Ender Dragon, exit) on unseen random seeds without hand-programmed speedrun logic or unfair hidden-world inputs. The objective is ambitious and unsolved here. Initial learning experiments intentionally simplify the task and supply a target-direction vector; this is not end-to-end visual speedrunning.

## Verified local machine and setup

- Gentoo Linux / OpenRC; Plasma Wayland and XWayland workaround.
- i7-14700KF, NVIDIA RTX 4070 SUPER 12GB, 32GB RAM; NVIDIA driver 595.104.02 reported previously.
- Python 3.14.10; host Java 25.0.4.1; MCio-managed Fabric/Minecraft logs show a Java 21 runtime.
- Existing project directory: `~/Documents/minecraft-discovery`.
- Local Python venv `.venv/`, installed `mcio_ctrl==1.5.2` with a fixed `cbor2>=5.7.1,<6`.
- Local MCio data `mcio-data/`; instance `DemoInstance`; original template world `DemoWorld`, Minecraft Java **1.21.3** and MCio mod **1.6.1+1.21.3**. User's main PrismLauncher and newer Minecraft installation are separate.
- Visual viewer on Wayland initially was invisible; use `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11` for MCio launcher/client. The viewer then was black until `cbor2<6` was installed.
- Missing `libflite.so` narrator error and remote Mojang profile lookup timeouts are observed and nonfatal; do not modify Gentoo on their account unless a test actually blocks.
- User prefers copy/paste commands, no `sudo`, no daemons, no global integration, no unrequested access outside the project directory.

## Experiments already verified

- Random controls and a genuine fresh saved-world copy restart worked.
- Full restart: approx 195 active steps/s visible, 185.5 active steps/s headless. Headless full-run 2,000 steps in 55.5s.
- Persistent-process soft resets: 5 × 1,000 steps, 284.5 active steps/s average and 123.5 end-to-end, ~0.08–0.11s between episodes. **World remains mutated between resets**.
- Reward probe: oak log first seen +1, repeat +0, stone pickaxe +1, reset, oak log +1. Commands injected the items solely for unit-style verification.
- Navigation learner: before 7/20 on held-out target positions; after 120 real Minecraft training episodes 20/20; average successful time 98 -> 24 ticks; 12.8s training. Goal direction supplied as numeric vector; yaw fixed. Local `checkpoints/navigation_policy.npz` and `logs/navigation_training.csv` exist on the user's machine, **not** on GitHub.

## Orientation experiment ready, outcome pending

Script: `experiments/orientation_learner.py`, imported `navigation_learner.py` module. Needs `checkpoints/navigation_policy.npz` from the previous run. It tests old weights with fixed and random camera orientations, trains a new random-initialized egocentric-goal policy for 300 episodes, then evaluates on the held-out targets and a new unseen set. Uses disposable `MDL_OrientationLab_v1`, preserves original `DemoWorld` and original checkpoint.

Run locally (from project root):

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 .venv/bin/python experiments/orientation_learner.py --visible
```

After the user has attached Git to the existing directory, this command replaces the old root-level script path; the Python source is in `experiments/`, with outputs still in the root-level `logs/` and `checkpoints/`.

## Integrity cautions

- MCio setup/admin `/fill`/`/tp`/`/give` is outside the policy's action space, never to be used to solve a scored challenge.
- Training/evaluation on one small arena cannot prove unseen-world generalization.
- Soft reset ≠ exact world reset.
- Git tracks source/docs/tests, not models, logs, saves or recordings. An unavailable local checkpoint must be recreated by training or explicitly brought over by the user.

## Immediate next step

**Review actual orientation results; then decide whether to fix camera generalization or teach learned mouse/camera turning and fair object observations.** Update all status docs and the TODO in the same GitHub change.
