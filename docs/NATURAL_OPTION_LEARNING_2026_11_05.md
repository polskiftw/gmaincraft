# Nov 5: from practice pickup to a trainable mine/pickup option selector

## Why the 12-world report was rejected, not celebrated

The previously launched `nptr25a` suite actually finished both 12-world
**frozen EVALUATION** arms. Raw outcomes: mining-Q shadow 16 naturally mined
wood blocks and 13 real inventory wood gains; player-visible pickup override
13 wood blocks mined, 10 real inventory gains and **8 gains temporally during
416 learned pickup motor choices**. However three pair members were NOT the
same initial player pose despite identical seeds: evaluation indices31,34,35.
The old `tools/run-natural-pickup-transfer.sh` also failed to run the final
analyzer because its script bytes were edited while Bash was running.
`logs/natural_transfer_nptr25a.invalid_unpaired.json` was preserved as a
DIAGNOSTIC report, `comparison_valid=false`. NO paired uplift claims.

This exposed a separate deeper problem: paired tests can diverge even when
initial player poses match and no pickup takeover happens (native stochastic
world simulation and sampled Q action trajectories). Do not treat two same
seed traces as a deterministic counterfactual. Verify first camera/player
observation fingerprint and frozen Q sampler RNG; require multiple fresh
worlds and independent action RNG seeds before claiming improvement.

## Source snapshots and bounded attempt

- `FreshWorldManager(...,paired_source_tag=TAG)` creates ONE untouched,
  vanilla noise-generated **pre-game source** world in an owned `MDL_*`
  namespace for each natural seed. Each arm gets a distinct live copy of
  the exact same natural world files. No teleport/summon, no manual item
  placement, no server/admin commands during scored policy play. The
  shared source is preserved for forensic auditing; no user worlds touched.
- After the first 12-world invalid batch, on indices31/32 a repaired
  two-world smoke `npfix31a` passed full strict pose matching and published
  its report (both arms 0 logs over a short 120-unit horizon).
- The opt-in `visible_memory_budget_v2` is a separate version from earlier
  `visible_memory_v1`: it cannot chase ONE repeatedly visible/remembered
  identical dropped item forever. At most 96 real Minecraft ticks of
  effort per item until own inventory gain or a genuinely different
  visible item. It uses only own visible position/item ID, elapsed ticks,
  and held inventory. This is still a NONLEARNED evaluation-only baseline,
  not a strategy given to the future agent.
- `npbud28a` is a valid shared-source two-world diagnostic on indices28,29:
  shadow mining acquired 3 wood; budgeted pickup arm acquired 0 and made
  0 pickup motor choices. Since neither arm took over, **the difference is
  baseline rollout variation, NOT damage caused by pickup skill**. The
  experiment specifically demonstrates why the *input trajectory*, not
  just starting pose, must be audited.
- New launchers snapshot their own shell source bytes under the experiment
  tag and run that immutable copy; in-flight script edits cannot break
  the final analyzer as happened in `nptr25a`.

## Trainable high-level option Q: no expert routing labels

`src/gmaincraft/natural_option_q_v1.py` defines a **versioned, real-data-only
2-option double-DQN**, with mild conservative Q penalty against unobserved
option actions:

- Option 0: frozen Q wood miner; option 1: frozen learned dropped-item pickup
  motor. The learned option network eventually chooses *at every decision*,
  without an `if visible wood then collect` trigger; it can also choose
  pickup search when an item is not currently in camera view.
- Input: the existing 15-feature camera/own remembered pickup reading and
  eight own-camera/player-status V9 mining fields (crosshair wood, mining
  progress, health/hunger, own held logs, etc.). NO hidden world map, seed,
  evaluator pickup stats, scripted goals or correct action labels.
- Reward TRAINING ONLY: actual own wood inventory gain (+25 per bounded log)
  and actual own Minecraft-mined wood event (+3 per bounded block), small
  game-time penalty and player death penalty. Stats are **outcome labels**
  applied after the actual Minecraft action; the network never sees them
  as an observation or scripted control decision.
- Transitions are a real `(own_before, chosen_option, vanilla_duration,
  own_after, real_inventory_gain, real_mining_event, death, terminal)`.
  Every row must match an actual committed Minecraft action. The network
  is scratch initialized and can be fully serialized with replay/optimizer.
- `--record-option-experience` opts in to compact action-level traces in
  completed natural-world journals, preserving actual chosen motor and
  sensor observations. The trace is not live RGB and contains no oracle.
- `--option-data-namespace train` runs **FROZEN source Q/pickup policies**
  solely to COLLECT real self-experience on new TRAIN stream indices,
  disjoint from future untouched EVAL seed namespace. It does NOT update
  the frozen sources or evaluate claimed generalization.
- `experiments/train_natural_option_offline.py` rejects uncommitted logs,
  wrong seed namespaces, unknown schemas, altered first-eye signatures,
  missing action records, imagined inventory gains, model drift, fewer
  than 128 real decisions, or datasets with too few authentic outcome/
  pickup examples. It produces `checkpoints/natural_option_TAG.pt` only
  after actually training on observed transitions. Synthetic unit tests
  DO NOT generate real training artifacts.
- `--pickup-selector learned_option_v1 --natural-option-model TAG` is an
  opt-in **frozen** rollout of this learned chooser in REAL Minecraft.
  `shadow_learned_option_v1` measures the exact source Q actor with same
  item sensor and loaded option weights, but never lets the option choose.
  The new learned network only earns trust after independent REAL fresh
  natural-world improvement over that control, and no evaluator-only labels.

## Reproduction when prepared data exists

```bash
# Collect first-hand real Minecraft TRAIN-namespace player outcomes, not
# another staged bedrock pickup practice treadmill. New unique tag always.
tools/run-natural-pickup-transfer.sh start npdata2ka 2000 12 600 1639377 \
  qv2memf1104 hardrep1105b budget_v2 data_train

# Monitor; don't start a second Minecraft while the suite lock is held:
tools/run-natural-pickup-transfer.sh status npdata2ka

# When BOTH arms have completed and authentic positive experiences exist,
# train an option selector using the actual committed action traces:
.venv/bin/python experiments/train_natural_option_offline.py \
  --tag option1105a --source npdata2ka_s --source npdata2ka_p \
  --steps 320 --batch-size 64

# ONLY AFTER genuine training succeeded: frozen heldout EVAL on separate
# never-trained EVAL-world indices, both arms share same source world bytes:
tools/run-natural-pickup-transfer.sh start npopt60a 60 12 600 1639381 \
  qv2memf1104 hardrep1105b learned_v1 eval option1105a
```

### Evidence standards before claiming learned composition

**No** claiming that staged success implies Any%, or that an observational
trace is proof of a causal takeover improvement. Final experiments must
verify actual inventory events, source and option model checksum, no
optimizer updates in scored heldout play, same pre-game snapshot and
initial player pose, source Q action RNG signatures, and new disjoint
heldout world seeds. If there is no real event data, report a gap, not a
fabricated trained policy. Two options do not yet solve exploration,
crafting, tool-building or learned full speedrun planning.

## November 5 deeper reliability: separate Q RNG for every matched world

The two-world real TRAIN-trace smoke `npdatap2a` successfully committed
actual action-level records from four genuine Minecraft survival episodes,
with sum(action.ticks) exactly equal to game ticks and player inventory
outcomes reconciled. It also exposed a crucial carry-over in earlier
**sampled** frozen Q rollouts: because two arms took different numbers of
Q actions in the preceding world, their global PyTorch sampling RNG
started the NEXT world at a different state. The naive same-seed approach
was inadequate even with identical shared world source and pose.

New `paired_frozen_q_sampler_seed` is a separate, SHA-256-derived,
versioned **evaluator-only action RNG** for each tuple of seed root,
policy RNG seed and episode index. It resets only within opt-in paired
survival evaluations, leaving all historic checkpoints/rollouts unchanged.
The first Q sampler RNG hash and exact initial camera/V9 player-observation
fingerprint are logged and verified per actual world. This eliminates the
prior-world action-count RNG confound while preserving honest in-game
stochasticity; it does **not** guarantee deterministic Minecraft outcomes.
A matched Q-only repeat is still useful to quantify nondeterminism.

### Better-than-hardcoded observational training data

The versioned `explore_v1 data_train` mode uses an **independent seeded
random behavior policy** to actually sample BOTH frozen options at states
with current player-visible items and at states with no known item. It
oversamples rare item opportunities (55% option-pickup sampling if an
item is visible/remembered, 8% otherwise) to reduce wasted genuine
Minecraft test ticks, but **never supplies a correct option label, a
hidden destination, an oracle reward, or a known speedrun strategy**.
The training objective uses only the resulting actual player inventories
and mined-block outcomes. Its stochastic choice is not a deployed policy.

```bash
# Stronger genuine real-world exploration data on a fresh TRAIN-only
# set of world indices, separate from previous frozen heldout tests:
tools/run-natural-pickup-transfer.sh start npoexp2ka 2002 12 600 1639393   qv2memf1104 hardrep1105b explore_v1 data_train

# After this fully completes with genuine positive outcomes:
.venv/bin/python experiments/train_natural_option_offline.py   --tag option1105a --source npoexp2ka_s --source npoexp2ka_p   --steps 320 --batch-size 64
```

This is a **new trainable research tool**, not proof the option network
selects useful actions yet. The separate frozen `learned_v1 eval MODEL`
mode is already implemented but must be compared against its shadow arm
on independent unseen EVAL seeds before any promotion.

## Unattended real-data to real trained-model handoff (opt-in)

The launcher accepts an optional tenth positional `AUTO_TRAIN_OPTION_TAG`
**only** for `data_train`. When BOTH actual Minecraft source streams have
independently completed and the per-game real own-inventory/action journals
pass validation, it invokes `train_natural_option_offline.py` on those
TRAIN-only committed journals under the same detached, uniquely named
workstation job. It does NOT train the source Minecraft miner or pickup
networks, and never consumes heldout EVAL worlds. If there aren't sufficient
positive real outcomes for the new option model, offline training exits
with an explicit error and **DOES NOT publish fake/random trained weights**.
No automatic claim of improved Minecraft gameplay occurs until an entirely
separate learned-vs-shadow heldout natural evaluation passes.

Example: `tools/run-natural-pickup-transfer.sh start npoexp2ka 2002 12 600 1639393 qv2memf1104 hardrep1105b explore_v1 data_train '' optdata2ka`.
The empty ninth argument means no pretrained option model (we are TRAINING
a new one). When real data qualifies, the checkpoint name will be
`checkpoints/natural_option_optdata2ka.pt`, otherwise it remains absent.
The suite's own diagnostic report is explicitly labeled TRAIN-data rather
than a causal heldout test, and may report initial-state mismatches without
discarding genuine single-world experiences.
