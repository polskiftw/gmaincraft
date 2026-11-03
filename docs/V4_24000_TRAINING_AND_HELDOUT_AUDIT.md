# V4 24,000-unit training and next two-seed held-out audit

2026-11-03. Completed real Minecraft 1.21.3 noise survival, no privileged
world map or phase switching. Local raw journals and saved weights are
Git-ignored and retained, not committed to Git.

## Full training result

`persistpilot1103` fully completed 40 distinct training seeds, 600 six-tick
units / 3,600 Minecraft ticks each: 24,000 total game-time units, 138 PPO
updates, 419 ordinary blocks broken, 4,882 newly first-hit sampled wood
surfaces, 3 real wood blocks mined and 3 actual wood inventory pickups,
17 death event samples, 0 silently fabricated wood successes. The true
wood mined/picked events were at seed episodes 0 (units 355/361), 34
(units 571/572), and 37 (units 273/275), each with a real inventory +1
and Minecraft item-pickup statistic +1. `logs/fresh_stream_persistpilot1103.jsonl`
ends with `stage=completed`, `episodes=40`, `decisions=24000` and totals;
checkpoint is `checkpoints/fresh_stream_persistpilot1103.pt`. Historical
600-unit original snapshot remains untouched at
`checkpoints/fresh_stream_holdv4_pilot600.pt`.

**Failure mode to tackle:** the duration head increasingly selected 72-tick
holds (training episode 0: 34/115; episode 39: 46/66). Long holds helped
break many more *ordinary* blocks near end (episode 32: 48; episode 34:
47), but are not inherently targeted at wood. Entropy in each quarter of
PPO updates decreased from averages ~5.45 to ~4.53. This is measured
behavioral concentration, not proof of policy improvement in the task.
The last episode's game I/O was 2.875ms per game tick, so there is no
obvious return of the old multi-second per-tick MCio regression.

## Independent held-out seeds 4 and 5 (frozen, matched game-time)

Each controller played the same two *previously untrained* eval seeds:
8457808129004096636 and -7504319717094273383, each for 600 units
(3,600 game ticks). Normal survival actions, no evaluator intervention,
no optimizer updates during frozen trials. **These two seeds are too few
for conclusive performance statistics.**

| Frozen controller | ordinary blocks mined | real wood mined | real wood inventory | deaths |
|---|---:|---:|---:|---:|
| Original V4 pilot checkpoint, updates=4 | 16 | 0 | 0 | 3 |
| Grown V4 checkpoint, updates=138 | 74 | 0 | 0 | 0 |
| Uniform-random four-head action/duration | 40 | 1 | 1 | 2 |

The uniform-random controller mined and **actually collected** wood on
seed index 4. That means 3 pickups in 40 train seeds alone cannot establish
learned generalization, and neither frozen model beat random at wood
collection on this pair. Frozen grown-v4 first-hit new wood sightings 787
vs pilot409 vs random609, but **wood sightings are not wood inventory**.
Grown V4 attacks while aimed at wood were 3 total (only seed5), and the
policy almost never holds still to attack wood (`wood_attack_without_look`
1 in seed5); random chose 9 wood-attack decisions across the pair. Real
frozen episode telemetry and mean I/O are preserved in their journals:

- `logs/fresh_eval_v4grown4_5cpu.jsonl`
- `logs/fresh_eval_v4pilot4_5cpu.jsonl`
- `logs/fresh_eval_v4random4_5cpu.jsonl`

An evaluation launch initially failed *before world gameplay* because
PyTorch could not initialize CUDA inside the current sandbox even though
`nvidia-smi` could see RTX 4070 SUPER. `/dev/nvidia-uvm` is absent from
this isolated shell, after host reboot to 6.18.60-gooned. Do not conclude
from this alone that the host driver is broken. New optional
`GMAINCRAFT_EVAL_DEVICE=cpu` runs frozen inference without changing model
weights, observations or Minecraft input contract. The three frozen tests
above completed on CPU; model metadata checks still passed, no checkpoint
was modified. The headless Gamescope *parent* still exits 139 after the
child saves and the evaluator independently verifies a complete journal.
Do not change host NVIDIA modules or permissions from this project sandbox.

## Next actual engineering / training decision

**Do not blindly resume this checkpoint for another 24,000 units.** Current
run is complete; a different reward, duration distribution, or motor schema
requires a new named/versioned from-scratch training run. Next prioritize
wood-targeting and same-block mining persistence instead of more incidental
ordinary mining. Before changing rewards, inspect crosshair-wood + attack
with no-look and crack-stage trajectories and implement nonprivileged
policy-facing history if needed. Keep genuine first-hit-only visibility
and the identical train/frozen observation schema, no scripted wood search,
pre-aimed fixtures, or evaluator phase handoff. Then compare on more
matched unseen seeds vs *both* current frozen V4 and matched random,
with wood **inventory** as primary success metric and ordinary mined,
crack progression and death as diagnostics.

If GPU training is desired, first verify `torch.cuda.is_available()`
**inside the same isolated Gentoo Shell connection** and ensure the sandbox
exposes the CUDA device nodes. The current journal/checkpoint are safe.
