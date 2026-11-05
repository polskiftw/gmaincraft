# Nov 5: first observed natural survival wood/pickup skill transfer

## Result and reason for a separate natural-world gate

`hardrep1105b`, the independently scratch-trained 1500-episode **staged
item-search** network, COMPLETED with 1170 actual own-inventory oak pickups
and 17892 genuine replay-Q updates. It did **not** establish stronger pickup
performance than the older 300-episode `egolearn1105b` network: on two NEW
hard, matched heldout case seeds 883735 and 884735 (200 trials each model),
`hardrep1105b` frozen greedy 86+88=**174/200**, old `egolearn1105b`
85+88=**173/200**. Thus additional identical staged curriculum training is
not justified. All frozen model checkpoints and scored worlds preserved.

## New genuinely natural experiment

`src/gmaincraft/natural_pickup_transfer.py` introduces an **opt-in, honest
and explicitly UNLEARNED** player-visible item opportunity switch. It is
an evaluation diagnostic only, not a solved skill manager:

1. Freeze the existing real self-play Q wood mining actor `qv2memf1104`
   with its exact V9 first-hit block/own-memory observations and ordinary
   V7 motor, with NO optimizer updates or evaluator progression signals.
2. Freeze the *separate* learned item motor `hardrep1105b`. Its inputs are
   exactly the same real, geometrically camera/ray-filtered dropped-item
   record and own player XYZ, yaw, inventory, timestamped sightings it used
   in staged training. The ItemEntity sensor is requested on the same real
   game-action tick as the first-hit surfaces and mining status.
3. Two control modes in ordinary, genuinely noise-generated **survival**:
   `shadow`: read visible dropped items but ALWAYS let the original Q miner
   act; `visible_memory_v1`: when the actual player's own item sensor first
   sees a wood drop, let the frozen learned pickup motor act while the
   visible or recently remembered item remains relevant (maximum 90 ticks
   without another sighting). Return to frozen Q mining otherwise. If own
   inventory actually grows, forget old consumed item. This is an explicitly
   **hardcoded sensory opportunity switch**, NOT learned option arbitration.
4. During scored play there are NO `/summon`, `/setblock`, `/tp`, `/give`,
   hidden item coordinates, seed-to-policy input, stat-triggered phase
   changes, evaluator teleports, or privileged item-distance hints. Actions
   are normal mouse and W/A/S/D input only. This is the same player
   observation permitted in train and scored play; the composite itself is
   currently eval-only because no independent high-level selector trained.
5. Preserve strict old behavior when `--pickup-model` is omitted, preserve
   old Q and numeric checkpoint schema, and keep the extra packet sensor
   enabled in BOTH matched arms. Loading frozen item weights restores
   PyTorch CPU/GPU RNG state to avoid perturbing sampled mining Q actions.
   The pickup actor itself never receives game stat totals or unseen world
   entities, and both actors are never updated in frozen evaluation.
6. Always audit own inventory pickups, actual wood mined, actual camera
   item sightings, option engagements, exact pickup primitive counts,
   `wood_inventory_gain` observed during pickup motor, deaths and held logs.
   Q/V7 action counters exclude unmatched pickup primitives; their separate
   `unfactored_primitive_actions` counter records the real motor count.

### First REAL smoke (seed 24, 600 six-tick units, 3600 real ticks)

Natural vanilla noise world seed `8554419461571164028`, frozen mining Q
`qv2memf1104`, frozen pickup `hardrep1105b`, independent sampled-Q action
seed 1639377, memory-only zero-RGB transport, no gameplay commands:

| Same natural world | Old Q-only shadow | Player-visible option switch |
|---|---:|---:|
| Wood actually mined | 3 | 5 |
| Genuine inventory wood pickup events | 2 | 4 |
| Deaths | 1 | 0 |
| Observed visible item sample ticks | 64 | 8 |
| Actual pickup motor actions | 0 | 13 |
| Inventory gains *during pickup motors* | 0 | **0** |
| Q optimizer updates | 0 | 0 |

**Interpretation:** the system can now execute BOTH frozen skill policies
inside uninterrupted natural Minecraft and its camera-visible wood sensor
fires in a real vanilla world. The observed inventory difference is a
one-world trajectory difference; there is currently **NO evidence the 13
pickup actions themselves caused a log to enter inventory**. It would be
misleading to claim learned skill composition or causal advantage. The
short run motivates paired additional worlds, not a longer old rewards run.

## Reproducible unbiased multiple-world evaluation

Use `tools/run-natural-pickup-transfer.sh start PREFIX START_INDEX WORLDS
UNITS_PER_WORLD Q_ACTION_RNG [Q_MODEL] [PICKUP_MODEL]`. The wrapper launches
one sequential mining-Q-only **shadow** study followed by the identical
natural Minecraft seeds with camera-visible pickup actions enabled. It
holds a separate suite lock, refuses existing tags, preserves BOTH full
world journaling and console logs, and validates completed no-update
journals, exact seed/time budget and no admin gameplay before summarizing.
This is an evaluation, NOT training the separate Q or pickup weights.

```bash
# Run a NEW detached paired study, with entirely unused heldout indices:
tools/run-natural-pickup-transfer.sh start nptr25a 25 12 600 1639377

# Later, while the user machine performs the run:
tools/run-natural-pickup-transfer.sh status nptr25a
tail -f logs/natural_transfer_nptr25a_launcher.console

# After BOTH arms finish:
.venv/bin/python tools/analyze-natural-pickup-transfer.py \
  nptr25a_s nptr25a_p
```

**NEXT research:** evaluate the actual real source of inventory gains, then
learn a trainable high-level chooser or joined policy from player-visible
natural-world state/experience WITHOUT a literal visibility-if-then phase
switch. A proper skill manager must learn when to mine vs collect vs
explore and be judged on fresh natural seeds, with a genuine inventory
outcome and independently trained network replication. Do not substitute
staged teleport/summon success or a single favorable seed for that proof.

## Important sensor boundary

The geometric item filter is validated for stone occlusion and camera
turn-away, but actual recognizable small objects through transparent
materials/leaves/low light are not completely calibrated. A candidate is
not an X-ray target or a long-range target oracle. Do not call the current
memory eyes fully human-equivalent yet.
