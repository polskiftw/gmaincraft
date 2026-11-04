# November 3: V5 / V6 real Minecraft wood-learning comparison

**Status: experiments completed, no convincing learned wood-harvesting skill yet.**
This report documents direct evidence from completed local checkpoints and
Git-ignored real Minecraft 1.21.3 journals. It supersedes the earlier V4
continuation recommendation to blindly train more of the same policy.

## CUDA fixed after kernel 6.18.60-gooned reboot

The first post-reboot shell saw graphics NVIDIA modules but no `nvidia_uvm`.
The user loaded the existing `nvidia_uvm` module on the Gentoo host and
restarted the `bubblewrap`-based Gentoo Shell bridge. Verified **inside** the
restarted shell: `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` now exist;
`torch 2.15.0+cu132` reports CUDA available; real CUDA tensor compute passed
on RTX 4070 SUPER. No need to rebuild the kernel or patch host permissions.
The sandbox launcher already conditionally passes the NVIDIA UVM nodes
through, but a running sandbox cannot retroactively gain those mounts.

## V5: `wood_target_v5` (same V4 typed observation/motor; fresh weights)

- Retains the ordinary move/look/attack + 6/24/72-tick *learned* duration
  heads and original V4 player-own + bounded first-hit camera input.
- Adds strictly bounded **first visible vanilla crosshair contact** reward
  (`.9` per unique actually targeted wood block, at most 4096/world).
- Doubles the earlier *own visible crack progress* reward, reduces ordinary
  exploration/irrelevant mining incentive, increases actual wood-mining and
  owned-inventory-wood rewards. No path hint, auto-aim, item spawning,
  privileged world survey, scripted mining phase or pretrained weights.
- Uses a separate new reward-profile metadata version so V4 checkpoints
  cannot silently resume under altered reward settings.

**V5 training:** `targetv5pilot1103` completed **10 new true noise worlds**,
600 six-tick game-time units each (**6,000 units / 36,000 ticks**), **41 CUDA
PPO updates**, 149 blocks mined, 1,078 newly sampled visible wood surfaces,
**one real mined-and-inventory log** (episode 0 only), seven deaths and
zero invalid crack readings. Average measured 2.54ms game I/O per tick;
no MCio performance regression. The original V4 pilot **also** picked up
wood on this same first train seed: this one shared result is not evidence
that the V5 change made it learn more effectively. V5 had only 21 wood
crosshair decisions and nine attack-while-targeting-wood decisions across
all ten worlds, with ZERO `wood_attack_without_look` policy decisions in
training. The central targeting/consistent mining problem remains.

## V6: `wood_engagement_v6` (new observation schema; fresh weights)

Separate from V5, a second from-scratch experiment added two explicit
player-knowable booleans/scalars derived **only from its current camera**:
`crosshair_is_wood` and visible-wood fraction within the same top-24
first-hit camera budget. A new exactly versioned observation/checkpoint
schema rejects incompatible V4/V5 weights. V6 also gives one-time reward
for actual crosshair wood contact + attack engagement (not a steering
script) and uses stronger wood crack rewards. These two visibility features
summarize IDs already observable to the policy; they do not expose unseen
blocks. The real engine decides whether an attack works.

**V6 training:** `engagev6pilot1103` completed **10 natural noise worlds**,
**6,000 game-time units / 36,000 Minecraft ticks**, 41 CUDA updates, 147
ordinary mined blocks, 974 new visible wood surfaces, eight deaths, **zero
wood mined and zero inventoried**. Average measured 2.62ms/tick; 143 pure
checks passed. Adding recognizability features and an attack-contact reward
has NOT yet produced reliable wood gathering.

## Matched held-out Minecraft survival controls

All frozen trials used the same user-approved memory-only observation
boundary, normal vanilla actions, 600 six-tick units (3,600 ticks) per
world, a disjoint never-trained Minecraft seed stream, no optimizer updates
and world-specific first-hit memory. Each two-world group tests matched
seed indices, **not necessarily matched stochastic policy trajectories**.

| Frozen controller | Held-out indices | Blocks mined | True wood inventory pickups | Deaths |
|---|---:|---:|---:|---:|
| V4 grown, 138 updates | 4-5 | 74 | 0 | 0 |
| V5 early, 12 updates | 4-5 | 33 | **1** | 2 |
| Uniform random | 4-5 | 40 | **1** | 2 |
| V4 grown, 138 updates | 6-7 | 17 | 0 | 2 |
| V5 grown, 41 updates, default sampled RNG | 6-7 | **61** | **1** | 1 |
| Uniform random | 6-7 | 15 | 0 | 4 |
| V6 grown, 41 updates | 6-7 | 55 | 0 | 0 |
| V4 grown, 138 updates | 8-9 | **105** | 0 | 0 |
| V5 grown, 41 updates, default sampled RNG | 8-9 | 21 | **1** | 1 |
| Uniform random | 8-9 | 19 | 0 | 1 |
| V6 grown, 41 updates | 8-9 | 22 | 0 | 2 |

**Critical stochastic controls:** replaying frozen V5 on seeds 6-9 with
independently chosen policy RNG seed `813777` gave **zero wood on all four
seeds** (21 then five ordinary blocks); greedy V5 took no ordinary block
mining on seeds 6-7 (and no wood). Greedy V6 on seed 6 likewise mined no
blocks and no wood. This makes it unsafe to call V5 a robust, learned
wood finder just because the default sampled rollout obtained wood on a
few held-out seeds. V4 often beats V5/V6 on ordinary mined blocks; this
metric is insufficient as a proxy for the desired inventory objective.

Raw (Git-ignored) audit journals:

- `logs/fresh_stream_targetv5pilot1103.jsonl`, checkpoint
  `checkpoints/fresh_stream_targetv5pilot1103.pt`
- `logs/fresh_stream_engagev6pilot1103.jsonl`, checkpoint
  `checkpoints/fresh_stream_engagev6pilot1103.pt`
- `logs/fresh_eval_v5pilot45_cuda_1103.jsonl`
- `logs/fresh_eval_v5train6k_eval67.jsonl`, `logs/fresh_eval_v5train6k_eval89.jsonl`
- `logs/fresh_eval_v5_rng813777_67.jsonl`, `logs/fresh_eval_v5_rng813777_89.jsonl`
- `logs/fresh_eval_v5_greedy_eval67.jsonl`
- `logs/fresh_eval_v5rand_eval67.jsonl`, `logs/fresh_eval_v5rand_eval89.jsonl`
- `logs/fresh_eval_v4grown_eval67.jsonl`, `logs/fresh_eval_v4grown_eval89.jsonl`
- `logs/fresh_eval_v6_eval67.jsonl`, `logs/fresh_eval_v6_eval89.jsonl`,
  `logs/fresh_eval_v6_greedy_6.jsonl`

All historical V2/V3/V4/V5 and V6 checkpoints, world artifacts and raw
journals remain in ignored directories. The V5/V6 modifications passed
143 pure tests (including GPU checkpoint round-trip). A known *Gamescope
parent* segfault can occur after a successful child save; success is
determined by the Minecraft child journal and verified model state, not
the parent process exit status.

## Decision and suggested next experiment

**Do not launch 24,000 extra units of this same V5 or V6 profile blindly.**
The primary weakness is rare simultaneous **wood crosshair alignment +
attack + no camera movement**, and the learned policy's **greedy** behavior
can collapse. The next major iteration should measure continuous correct
crosshair-on-wood durations and physically accumulated break progress,
not simply reward first-time glances or attack-button presses. Consider
explicit memory of *already visually targeted* block progress, adaptive
choices for keeping camera steady while attacking, and a better action
credit scheme; no automatic target selection or privileged world query.
Always use new versioned observations/rewards and a scratch model, then
compare inventory pickups across many unseen seeds and multiple policy RNG
seeds against uniform random and frozen V5/V4. Only wood inventory and
actual mined item events count as first-log acquisition.
