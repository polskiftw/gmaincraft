# First autonomous wood mining + inventory pickup in an unscripted natural Minecraft world

**November 3, 2026. Actual vanilla-noise Minecraft Java 1.21.3 / CUDA / Gentoo.**
This is the first recorded genuine **natural-world** inventory log in this
project without a placed/preaimed tree or evaluator-scripted mine/pickup
phase. It is NOT a speedrun, not a guaranteed skill, and only one of four
held-out seeds yielded wood in a very small frozen evaluation. All raw
reports, disposable worlds, checkpoints and prior models are Git-ignored
and preserved locally. Run tags below identify the exact evidence.

## Round 2: diagnose `woodv3_01` before changing anything

Original `speedrun01` v2: 24,000 six-tick choices, 250 neural updates,
10 different noise worlds, 106 ordinary blocks mined, 759 unique visible
wood surface sightings, 0 mined wood and 0 inventory wood. Action collapse
in earlier controlled frozen run: 452 of 480 actions were forward+jump;
only 12/480 attempted attack. V3's higher entropy and lower learning rate
improved action diversity but did not yet acquire wood:

| 24,000 six-tick training choices | v2 `speedrun01` | v3 `woodv3_01` |
|---|---:|---:|
| True ordinary blocks mined | 106 | 338 |
| Newly camera-sampled wood surfaces | 759 | 2,895 |
| Wood blocks mined | 0 | 0 |
| Items of wood acquired | 0 | 0 |
| Death event samples | 3 | 17 |
| Neural updates | 250 | 250 |

V3 trained with genuine real Minecraft 1.6–2.6ms/tick world-level game I/O,
no new hangs. Across 24,000 decisions v3 attempted attack 13,442 times,
was aimed at wood while attacking only 250 times, and attempted attack
with no look delta while wood was targeted only 39 times. Its longest
consecutive attack + no-look decision streak was four decisions (24 ticks),
whereas a wooden log mined by hand can require substantially more than
that. This is an observation about *action persistence*, not a claim that
crosshair alignment always survives other movement.

V3 frozen on the exact first two held-out seeds for 600 game-time units
(3,600 ticks each) mined 15 ordinary blocks and 0 wood. V2's earlier
matched 240-unit/seed test is not a same-duration 600-unit comparison.

## Versioned V4: let the neural agent choose its own action duration

The v4 `wood_persistence_v4` policy has a new, incompatible-on-purpose
checkpoint schema:

`natural_memory_online_v2_first_hit_objects_ppo_variable_duration`

It retains the **same exact typed player-own and bounded camera first-hit
memory observations** as v2/v3, and adds a **fourth learned categorical
head** selecting one of **6, 24, or 72 Minecraft game ticks** to hold the
agent's chosen ordinary move/look/attack. Camera delta is applied only at
the start of a hold. The controller does NOT choose a target, steer,
select a mining phase, force attack, place blocks or teleport the player.
This is not a hand-written wood-mining skill. Train and frozen gameplay
use the **same** direct-memory observations and motor contract.

During each longer hold, the controller samples real Minecraft player
and world-surface observations every six ticks for rewards and actual
inventory scoring. The actor is **not** offered intermediate choices or
extra knowledge; a single macro has one neural decision and one PPO
transition. Rewards within a hold are discounted by the real game tick
offset and SMDP `timed_gae` discounts its continuation by the actual
hold duration. The *per-world and per-run decision budgets* still count
six-tick **game-time units**, so a 600-unit v4 episode has exactly 3,600
game ticks just like a 600-decision v3 episode, regardless of how many
long neural actions v4 selected. Pulse telemetry reports both
`decisions` (six-tick units) and `policy_actions` (actual choices), plus
duration counts. A player death or stop request can end a hold early.

V4 retains the real `wood_progress_v3` one-time visible crack-stage
reward and reward/inventory truth. Its optimizer uses lr=0.00015,
entropy coefficient=0.04, and **32 policy transitions per update** (v3
uses 96 six-tick decisions per update). The new reward profile, action
schema, duration choices, optimizer update cadence and six-tick horizon
are checkpointed and exact-resume protected. The old three-head v2/v3
network and saved model schemas remain loadable; they are not overwritten.

## The first natural wood: independent live Minecraft evidence

`tools/run-fresh-treadmill.sh start persistpilot1103 600 600 wood_persistence_v4`
ran on naturally generated Minecraft noise seed **1671001014072077565**,
spawn `(0.5,69,0.5)`. It started with randomly initialized neural
weights, played **600 six-tick game-time units** (3,600 actual game ticks)
through **115 policy actions**, performed **4 CUDA PPO updates**, and
saved the model, optimizer, RNG and committed game-event journal.

- At **game-time unit 355**, `uncommitted_game_event` recorded an actual
  `minecraft:mined` wood block counter increase: `new_wood_mined=1`,
  `new_mined_blocks=1`, `wood_inventory=0`.
- At **unit 361**, the real inventory increased: `wood_inventory=1`,
  `wood_inventory_gain=1`; Minecraft's own pickup statistic also advanced
  (`new_wood_picked_stat=1`). This is NOT a surface-sighting proxy.
- The committed episode confirms **27 mined blocks including 1 wood**,
  **1 acquired wood item**, 114 newly camera-observed wood surfaces, 79
  novel visitation cells, 0 death events, 0 invalid crack readings, and
  2.481ms/game-tick MCio I/O. Client startup10.436s, worldgen6.294s.
- Original vanilla input packets had NO trainer commands. World generation
  used `minecraft:noise`, no placed oak fixture or privileged target state.

A real result from a nearly untrained policy is **not** proof that its
weights have learned to prefer wood. A lucky exploratory macro can also
succeed; only repeated matched frozen tests and learning curves can
establish superiority.

Historical original 600-unit checkpoint was copied without alteration to
`checkpoints/fresh_stream_holdv4_pilot600.pt` (Git-ignored), SHA256:
`90dca0d360c51d9255303358dfaadf9b13c29fd1c018541e30f63bf8dc78e362`.
The original `persistpilot1103` checkpoint has the same SHA before
continuation. This snapshot remains an independent frozen control if
future training changes the original checkpoint.

## Frozen unseen-seed audit with exact in-game-time parity

Both controllers use identical full64 seed stream root917211,
vanilla `spawnRadius=0`, the same four held-out world seeds and initial
positions, 600 six-tick units / 3,600 game ticks per seed, and identical
vanilla motor options. No evaluator commands or training updates occurred
during gameplay. Frozen V4 is the original four-update **pilot** model.
The random baseline selects all four move/look/attack/duration heads
uniformly. V3 frozen was run on the first two of these same seeds at the
same exact tick horizon.

| Heldout pair | Frozen V4 trained pilot | V4 uniform random |
|---|---:|---:|
| Seeds 0 and 1, normal blocks mined | 34 | 60 |
| Seeds 0 and 1, actual wood mined / collected | **1 / 1** | 0 / 0 |
| Seeds 2 and 3, normal blocks mined | 56 | 46 |
| Seeds 2 and 3, actual wood mined / collected | 0 / 0 | 0 / 0 |
| **Total 4 seeds, blocks mined** | **90** | **106** |
| **Total 4 seeds, wood mined / inventory** | **1 / 1** | **0 / 0** |

Frozen V4 wood events in unseen seed0: wood mined at six-tick unit147,
actual inventory item gained at unit149. Both event types were independently
observed in the game; `new_optimizer_updates_in_eval=0`. V3 frozen in
seeds0 and1 under the same 600-unit horizon mined 15 nonwood blocks and
collected 0 wood. **Four held-out seeds are not enough to establish a
learning advantage.** V4 random mined more total nonwood blocks, so do
not conflate inventory success with overall action quality.

Raw journal tags:

- `logs/fresh_stream_persistpilot1103.jsonl` — natural pilot and checkpoint
- `logs/fresh_eval_persistfrozen_1103.jsonl` — frozen V4 heldout indices0,1
- `logs/fresh_eval_persistrandom_1103.jsonl` — random V4 indices0,1
- `logs/fresh_eval_persistmore_1103.jsonl` — frozen V4 indices2,3
- `logs/fresh_eval_persistrandom_more_1103.jsonl` — random V4 indices2,3
- `logs/fresh_eval_v3longmatched_1103.jsonl` — old frozen V3 indices0,1

A new optional last argument `EVAL_SEED_START` on
`tools/run-fresh-evaluation.sh` selects disjoint held-out indices without
replaying earlier worlds, e.g. `... wood_persistence_v4 4`. This start
index NEVER enters actor observations. `evaluated_episodes` distinguishes
worlds actually played from absolute seed cursor in eval summaries.

## Resume safety and next long experiment

Independent live V4 `persistresume_1103` ran to **120** game-time units
(one true new Minecraft world, 1 CUDA update), then resumed to **240**
units on the NEXT unseen train seed. Both journal/atomic model validations
passed; update count2 and no previous seed was silently repeated.
The old v3 frozen model also ran correctly under the new source code,
so the new optional V4 policy did not invalidate the old checkpoint.
**134 pure tests pass**. No host NVIDIA settings, unrelated saves,
pre-existing weights, or old historical mods were changed.

**Recommended exact next user run:** CONTINUE the successful V4 pilot
instead of destroying its optimizer/RNG state or retraining blindly:

```bash
cd "$HOME/Documents/minecraft-discovery"
tools/run-fresh-treadmill.sh resume persistpilot1103 24000 600 wood_persistence_v4
```

The numeric target is **24,000 total six-tick gameplay units**, not
24,000 extra from the saved 600. The per-world horizon MUST remain 600
to honor exact model+seed metadata. This plans 39 additional *new*
vanilla noise survival seeds (40 total), until interrupted or the bound
is reached; prior completed worlds are retained/cleaned according to
existing safe owned-world policy. It is foreground-only, not a daemon.

After the long run, frozen-evaluate both the **grown** V4 and preserved
**600-unit snapshot** versus the random duration baseline on several
matched held-out seeds; compare actual wood inventory per world, action
head/duration entropy, death frequency and game ticks, not just wood
sightings or cumulative ordinary mined totals. Never infer that more
training necessarily improves generalization. Next major engineering
priority remains bounded visibility-checked dropped item entity memory,
then completely learned item use, crafting, other dimensions and Any%.
