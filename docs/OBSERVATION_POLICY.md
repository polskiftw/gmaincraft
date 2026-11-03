# What the Minecraft speedrunner is allowed to know

**Authoritative design intent, clarified by the user on November 2, 2026.**
This document supersedes older blanket phrases such as "RGB-only", "pixel-only
fairness", "no memory reads", or "all Minecraft stats are privileged". Older
experiment descriptions can still truthfully say *those specific models* saw
only pixels. Those are implementation histories, **not project requirements**.

## The actual goal

Train a general-purpose Minecraft Java Any% speedrunner from randomly initialized
weights that learns by interacting with normal survival worlds, eventually
finishing on **unseen random seeds**. The agent should act using ordinary
in-game actions (walking, looking, mining, interacting, using inventory, etc.),
but it **does not need to perceive its world exclusively through screenshots**.

The desired observation is a **mixture of game-engine/memory-derived structured
state and optionally RGB/audio**, filtered by an explicit *skilled-player
knowledge boundary*. Reading Minecraft memory is **deliberately allowed in
both training and scored gameplay**. The constraint is on **what information**
it learns from or receives, not **how** that information was retrieved.

The agent also need not look or move like a human; optimize success and then
in-game time. Do not hand it a hardcoded human speedrun, expert demonstrations,
external pretrained model, or a magic destination/path to win.

## A useful observation budget (proposed, not yet implemented)

The following are candidate direct-read **policy inputs in both training and
inference**, without requiring the agent to parse the screen or open F3 every
time. Each field needs documented provenance, semantics, and version:

| Source of information | Examples of permitted structured inputs | Condition |
|---|---|---|
| Own player state | XYZ position, facing/yaw/pitch, dimension, movement status, health, hunger, armor, experience, air, active effects, time since input | Information ordinarily visible through HUD/debug display or directly knowable about one's own character; exact numeric values can come from game memory |
| Player-owned inventory | Hotbar, full inventory, equipment/offhand, item counts/types, selected slot, durability | Data from the player's own accessible inventory and UI; direct memory access is fine rather than making the model perform OCR |
| Progress and remembered knowledge | Items acquired/used, blocks mined, deaths, achievements, elapsed game ticks, explored/previously observed places, its own action history and memory | Ordinary progress facts a skilled player could track; event counters must be synchronized and accurately attributed |
| Local environment | Biome, day/night or light, weather, current targeted block/entity, known reachable interaction, nearby player-perceivable entities | Use a player-knowledge/visibility filter where the game would otherwise reveal hidden information |
| Perceivable blocks and entities | **Block IDs/properties and entity types/positions** read from Minecraft's internal world data, but only for objects the agent could legitimately detect in the current view or remember from earlier observation | Eligibility determined by camera, geometry, occlusion, range and perception research below; NOT a bulk world-data dump |
| Additional optional perception | RGB frames, ordinary game sounds, UI rendering, agent's own remembered map | Valuable optional channels, not a mandatory pixel-only bottleneck |

**Important:** this is an intended *observation design*, not a claim these
fields are wired into the neural policy already. Vanilla-visible/F3-equivalent
information may be read numerically **without showing the F3 screen or
reconstructing pixels**. Likewise, using the actual item ID/count from one's
own inventory is fine even when the inventory GUI is closed.

### What still should not be handed to the playing agent

- Exact unobserved underground ore veins, block structures behind walls,
  hidden mobs/entities, or container contents before opening/observing them.
- World seed or coordinates of an unvisited stronghold/portal, an oracle route,
  hidden goal location, other global map knowledge before discovery.
- Automatically refreshed world facts about places the agent has left and
  cannot currently perceive. It **may remember** the last legitimately
  observed state, including uncertainty when that state can change.
- Administrative edits/teleports, item spawning or impossible movement as
  the **agent's gameplay actions**. Using controller commands *outside scored
  play* for training fixture setup, reset, and independent evaluation is
  documented separately.

This is **not an objection to memory reading**. It is a distinction between
skilled-player-knowable state and an all-knowing X-ray/world-seed oracle. If
later experiments suggest useful adjustments to this boundary, discuss the
tradeoff with the user and version the observation contract rather than
quietly imposing new restrictions.

## Visible/actionable block selection — an explicit future research task

The user specifically wants us to **investigate and derive the thresholds**
for how much local block/entity state is supplied, **not arbitrarily choose a
radius or field-of-view cutoff now**. Treat visibility and interaction as two
separate questions:

1. **Can the player perceive this block/entity?** Study the actual camera
   frustum, angular coverage, on-screen projected size, line of sight,
   occluding blocks, transparency, lighting and geometry. An eligible block
   can be returned as structured ID, position relative to the player, surface,
   and relevant observable properties, read directly from engine memory. A
   memory record of a previously seen block is separate from a live reading.
2. **Can the player act on it now?** Independently apply the game's actual
   vanilla targeting, crosshair raycast, interaction/reach rules and collision
   checks. A tree can be visible at a distance without being in mining range;
   the agent must still approach and aim with normal inputs.
3. **How much should the model receive?** Design and benchmark a bounded
   observation representation: candidate object count, distance/visibility
   confidence, precision, temporal update rate and how to represent occlusion
   or partial visibility. Measure input bandwidth/throughput and learning
   performance; do not indiscriminately serialize whole loaded chunks.
4. **How to validate the threshold choices?** Generate varied real-game
   scenes (angles, distances, glass/leaves, water, darkness, moving entities,
   partial occlusion), compare engine-derived candidates with what a skilled
   player could reasonably perceive, check false X-ray positives and missing
   visible objects, and test held-out seeds. Use the evidence to choose and
   document the thresholds. Do not pretend a calibrated detector already
   exists or silently substitute arbitrary numbers.

The Minecraft mod/MCio path is a reasonable way to expose these fields **from
inside the game process**, rather than requiring brittle external RAM address
scanning. Inspect actual MCio/Fabric capabilities and add a small scoped
read-only observation API if necessary. Do not equate read-only memory
observations with server commands or changes to the game's mechanics.

## Training, gameplay and scoring must agree about available knowledge

- **Both training and normal gameplay:** the learning policy can consume the
  *same bounded player-knowable structured observations*, optionally plus
  screenshots/audio/history. Do **not** remove numeric state only in scored
  evaluation and then call the comparison fair.
- **Training controller / evaluator:** can have additional hidden world data
  for resets, scalar rewards, counter verification and independent scoring.
  Mark it as teacher/evaluator-only. It should **not** become a policy feature
  that magically locates an undiscovered target.
- **Scored gameplay:** no admin commands, supernatural movement or hidden
  oracle destinations; use normal survival game mechanics and meaningful
  independent random-world seeds.
- **Auditing:** record which policy fields were enabled, their provenance and
  filtering version, train/eval observation parity, block visibility decisions,
  rewards, checkpoint hashes and actual game/inventory outcomes. Preserve
  historical pixel-only evaluations as **separate baselines** rather than
  pretending they used the newer structured inputs.

## V4 milestone: memory as actual policy input in train AND frozen play

`player_knowledge_v1` is no longer merely a staged Python adapter: the
opt-in V4 factorized PPO actor (`src/gmaincraft/memory_factorized.py`)
consumes the same own position/yaw/pitch/health/inventory/stats-derived
20-float feature encoder during real Minecraft CUDA training and **normal
frozen gameplay**, alongside two 84x84 rendered RGB frames. Code paths are
`MCioAgentBridge.multimodal_view()` in `joint_log_factorized.run_episode`
(training/heldout arena) and `natural_world_benchmark.py` via
`natural_eval.MemoryFactorizedAgent` (seeded natural survival). The versioned
schema is `factorized_v4_memory_v1_rgb2_6tick`; old RGB-only experiments are
still separate. First V4 real games were 0 successful inventory pickups;
implementation ≠ learning competence. See `MEMORY_POLICY_V4.md`.

**World-object sensor is still absent.** Current MCio/Fabric 1.6.1 doesn't
include targeted block/entity hit or hunger in the base observation packet.
Do not substitute a raw loaded-chunk map or uncalibrated distance cutoff.
The empirical visible/actionable filter remains a future milestone.
Temporary V4 display distance of **4 chunks** is a graphics/performance
choice, NOT an empirically calibrated memory-visible block cutoff; see
`MINECRAFT_RENDER_PERF.md`. The same rendering profile must be recorded for
fair side-by-side scored comparisons.

## Implementation note (November 2, 2026): first versioned adapter

`src/gmaincraft/player_knowledge.py` now implements an opt-in,
`player_knowledge_v1` snapshot via `MCioAgentBridge.player_knowledge()`:
position, yaw/pitch, health, owned inventory and known stats derived from
MCio packets; protected as bounded immutable values. This has unit and live
Minecraft inventory-parity tests. **No policy/trainer has adopted the new
schema yet**, and no visibility-filtered blocks, entities, hunger or biome
observations are currently exported. The historical RGB interface remains
unchanged. See `MCIO_THROUGHPUT_2026_11_02.md` for live measurements.

## What the existing code really does (as of this design note)

`mcio_ctrl.network.ObservationPacket` **already exposes** `player_pos`,
`player_yaw`, `player_pitch`, `health`, `inventory_main`, `inventory_armor`,
`inventory_offhand`, rendered `frame` and optional `options` (including
statistics where configured). `MCioAgentBridge` privately receives these,
but the **current** `src/gmaincraft/agent_interface.py` converts the packet
into an `AgentView` containing **only RGB pixels**. That restriction was the
assistant's original prototype decision, **not what the user requested**.

No engine-derived visibility-filtered block list, player-knowable state
schema or paired structured-state vs pixel-only trained comparison has been
implemented yet. The old CNN/PPO and self-supervised pixel experiments are
valid historical research records but **must not drive the whole project
back into a vision-only design**.

## Project order remains deliberate

**Priority 0:** fix or conclusively characterize the existing MCio simulation
throughput regression (~300 game ticks currently ~3.9-4.0 s versus historical
~0.6-0.8 s under not-yet-matched conditions), with no-neural matched
benchmarks. `docs/NEXT_SESSION.md` is the runbook.

**Then:** design/implement a versioned **structured player-knowledge
observation**, starting from existing MCio player state, inventory and
statistics. Research visible/actionable-block selection and tune eligibility
from experiments; do not hardcode unverified thresholds in documentation.
Add sensible baselines and direct-memory observation parity in train/eval.

**Then:** revisit the learning algorithm and integrated log-find/mine/collect
skill under this intended sensor model, before scaling to continuous crafting
and eventually Any%. Keep earlier checkpoints, worlds and tests intact.
