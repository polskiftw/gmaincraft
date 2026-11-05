# November 5: evidence-gated numeric item-pickup training kit

## Why this replaces indiscriminate activity-count training

Prior staged numeric actor-critic, trained for 200 actual Minecraft practice
cases, collected 109 real items. On 100 truly disjoint staged heldout cases:
**random 56/100; frozen greedy 55/100; frozen sampled 54/100**. The greedy
model selected `forward24` on all 147 motor decisions, which is not a
conditional skill. This evidence cannot justify further identical PPO runs.

Frozen Q wood natural-world evaluations, on indices 16..23 with independent
policy RNGs: longf 5+2 real wood versus memf 1+8; different stochastic
results, neither reliably dominates. The current problem is item collection
and meaningful conditional movement, not increasing `new_visible_wood`.

## New versioned practice kit

- Existing files/code/checkpoints preserved; new `egoreplay_v2` CLI flag
  changes ONLY the new practice learner, not the V9/Q natural wood policy.
- The controller stages real one-log `ItemEntity` drops **before** each
  practice episode in a disposable bedrock world. Each item angle is drawn
  relative to an independent uniformly randomized world yaw; many items
  require lateral movement or turning rather than a fixed forward hold.
- The actor receives only the opt-in geometrically camera-verified current
  dropped-item sensor, its *own* XYZ/yaw, old timestamped sightings, its
  own movement feedback, previous action and current inventory. Relative
  target position is rotated into the player's camera direction, avoiding
  a pointless need to rediscover movement as Minecraft world yaw changes.
- **No** fixture case/random seed/hidden item location/teacher action,
  screenshots, commands or chunk maps enter the network. All scored control
  is ordinary `W/A/S/D`, mouse look or idle, 6/12/24 vanilla ticks.
- A replay-backed, scratch-trained double-Q network uses real actual player
  movements and inventory acquisitions for feedback. Potential-based
  progress uses the *agent's own* remembered visible item position, not
  distance to the staged fixture target. Real pickups give +25; episodes
  store typed before/action/after/reward/duration/terminal transitions.
  Successful self-collected transitions are retained separately; NO
  artificial expert actions or imagined successful states are manufactured.
- Fresh model/optimizer/RNG/replay checkpoints committed atomically after
  each episode and verified against the journal on resume. Unique world/log
  tags refuse accidental overwrite. Training and frozen evaluation use the
  same sensor and action schema. Old kit accessible as `legacy_v1`.

## Actual real-game results on November 5

One NEW network `egopilot1105a`, trained from scratch on 60 actual staged
practice episodes (25 actual oak logs acquired, 612 Q updates), then FROZEN.
Two separately seeded 100-case evaluations and complete no-optimizer-update
journals, run with identical stage cases per control:

| Unseen practice case seed | random | always forward | learned greedy | learned sampled |
|---|---:|---:|---:|---:|
| 912771 (100 cases) | 48 | 53 | **72** | 76 |
| 992771 (100 cases) | 48 | 44 | **67** | not measured |
| **Pooled 200 matched cases** | **96** | **97** | **139** | n/a |

Pooled greedy **69.5%**, random **48%**, fixed forward **48.5%**.
`tools/gate-numeric-pickup-v2.py` uses actual inventory and complete frozen
logs, 2+ independent case seeds, exact scenario matching, no optimizer
updates, and demands 10+ percentage-point improvement over BOTH controls;
this one network gives a provisional PASS with **21-point margin** over
strongest control. It does not claim cross-run statistical confidence,
natural world collection, mining-to-pickup composition, or completion of
Minecraft. Initial item currently visible in 84/100 cases of each suite;
the remaining 16/100 are off-camera due to randomized bearings, which is
useful but still easier than natural forests and many simultaneous drops.

The headless Gamescope **parent process exits 139** after the Python
experiment has printed a valid `completed` summary. This also occurred on
older practice tests. The journal, episode count and frozen update counters
are the decisive validation; a separate compositor teardown issue remains.
A forced-forward evaluation initially failed at case 10 with a stale
inventory reset race; the pre-scored reset retry now handles this boundedly.
The rerun `efwd2_1105a` completed 100/100 and is used in the gate.

## Reproduce and next experiments

```bash
# Read current results, no Minecraft launch:
.venv/bin/python tools/gate-numeric-pickup-v2.py \
  egoeval_greedy_1105a erand1105a efwd2_1105a \
  egreed2_1105a erand2_1105a efwd3_1105a

# Start a NEW 300-practice-episode learning run, never overwrite old:
tools/run-numeric-pickup-v2.sh train egolearn1105b

# Resume same model/journal after interruption:
tools/run-numeric-pickup-v2.sh resume egolearn1105b

# Example frozen eval: choose a new tag AND a genuinely unseen case seed:
tools/run-numeric-pickup-v2.sh eval egoval1105b \
  --model-tag egolearn1105b --episodes 100 --max-actions 24 \
  --case-seed 771151 --policy-seed 881152 --device cpu --control greedy
```

**Next milestone:** independently retrain at least two other network
initializations and evaluate on new case seeds and larger case difficulty;
monitor wall/time per REAL pickup, failure-to-see, failures due to leave-area,
behavior conditioned on item bearing, and holdout vs strongest fixed baseline.
Then test **naturally mined** item entities in a continuous unmodified
survival episode without evaluation-triggered phase switching or commands;
only afterward merge item eyes into the larger speedrun agent's VERSIONED
policy and skill selector. Do not initiate another long wood-Q run until
that transfer is measured. This is not yet full-world autonomous training.
