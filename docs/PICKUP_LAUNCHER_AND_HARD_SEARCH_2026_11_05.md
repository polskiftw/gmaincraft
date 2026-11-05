# Nov 5: repaired pickup launch and real hard-search generalization

## User run investigation — no training data was lost

The exact earlier command `tools/run-numeric-pickup-v2.sh train egolearn1105b`
DID train and commit **all 300 actual staged Minecraft episodes** (258 real
held oak pickups, 3456 double-Q optimizer updates, checkpoint cursor 300,
253 of the staged drops initially in camera view). Its complete journal is
`logs/numeric_pickup_egolearn1105b.jsonl`, checkpoint in
`checkpoints/numeric_pickup_egolearn1105b.pt`; neither is overwritten.
The old wrapper propagated signal exit **139** from Gamescope as though the
learning process had failed. This is a *post-child compositor teardown*
SIGSEGV: raw output shows Minecraft stopped and `Primary child shut down!`,
then Wayland broken pipe and Gamescope signal 11. It is not an indication
that Python/Minecraft failed to save the episode or model. Root cause in
native compositor not yet fixed; do not suppress unrelated Python errors.

## Repaired kit launcher — tested live

`tools/run-numeric-pickup-v2.sh` now uses
`tools/stream-numeric-pickup.py` to keep a complete per-run raw console
and present short READY/PULSE/SUMMARY signals. It waits for the child
return code without printing the misleading shell `Segmentation fault` on
the terminal. Crucially, the independent
`tools/verify-numeric-pickup-v2.py` audits every recorded episode, actual
held target inventory, no scored commands, contiguous episode cursor,
model schema, optimizer/checkpoint alignment and frozen no-update rules.
Only a completed full run with a verified committed checkpoint and the
KNOWN compositor teardown code 139 can report successful training; a
partial/missing/corrupt journal or unrelated abnormal exit fails.

New commands:

```bash
# Confirm user's already finished training without relaunching:
tools/run-numeric-pickup-v2.sh status egolearn1105b

# New unique tag (never clobbers previous models/worlds):
tools/run-numeric-pickup-v2.sh train another_tag --episodes 300

# Detached unattended training in isolated Minecraft:
tools/run-numeric-pickup-v2.sh start hardrep1105b \
  --episodes 1500 --case-seed 531151 --policy-seed 971153 \
  --case-profile search_hard_v2

# Safe progress / stop checks:
tools/run-numeric-pickup-v2.sh status hardrep1105b
tail -f logs/numeric_pickup_hardrep1105b_launcher.console
tools/run-numeric-pickup-v2.sh stop hardrep1105b
```

`stop` is graceful and only available with a held run lock; `resume`
requires the same original case seed, policy seed, model horizon and
case profile. Increase `--episodes` ONLY if intentionally extending an
old run. Never start two Minecraft jobs simultaneously on DemoInstance.

## Hard item search: less obviously visible drops

New opt-in `--case-profile search_hard_v2` retains the exact same numeric
player-observation, vanilla motor and double-Q network. The ONLY change
is the research **staged** item placement: full world-yaw randomization,
3.0–5.8 block distances, roughly half initially behind the player and
much broader lateral angles. Case randomization stays out of policy
input; all scored actions are ordinary mouse and movement controls.
The default `standard_v2` sequence/metadata is unchanged, preserving
exact old checkpoints and reproducibility. Hard training will be an
independently initialized, version-tagged run, never a silent mutation
of old 300-world training.

Frozen model `egolearn1105b` (300 **standard** cases) was scored on two
new independent hard seeds with 100 matched real-game Minecraft episodes
per control, no optimizer updates in any of the frozen runs:

| Hard seed | frozen greedy | random | always forward | initially visible |
|---|---:|---:|---:|---:|
| 881735 | 88 | 24 | 18 | 31 |
| 882735 | 93 | 26 | 17 | 40 |
| **Both (200)** | **181** | **50** | **35** | **71** |

The same old frozen model also scored 100/100 on easy heldout seed 912771.
Hard pooled **90.5% vs 25% random** passes the strict two-seed provisional
inventory success gate; 57/69 initially offscreen items were acquired in
seed 881735. This is **still staged dropped logs**, not natural-world
Minecraft wood mining/pickup or composition and not independent-network
replication. Next step: compare the fresh hard-trained network on new heldout
hard seeds against the frozen standard-trained network and controls, then
work toward mined-drop collection in a natural unmodified survival episode.

## New independent hard-profile training (launched, not a planned command)

`hardrep1105b` was started detached via the repaired `start` CLI with
1500 practice episodes, run seed531151 and policy seed971153. The actual
Python process arguments and first committed journal entries explicitly
confirmed `search_hard_v2` and the distinct seeds; held lock and Java
process were observed. This is independent scratch training, NOT resuming
old standard-trained weights. Live `status` checks tolerate legitimate
race conditions between the worker's atomic checkpoint and its JSONL
commit; the **final** completed validation remains strict. On completion,
run fresh unseen hard-evaluation comparisons, then investigate natural
mined-log transfer. No extra Minecraft jobs concurrently.
