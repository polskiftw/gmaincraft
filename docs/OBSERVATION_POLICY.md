# What the Minecraft speedrunner is allowed to know

## Current visibility research evidence (NOT trained policy inputs)

Independent real-game screenshot/memory comparisons, a valid multi-seed
sensor throughput study and a Java exact-vertical-camera fix are reported
in [`VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`](VISIBLE_SURFACE_CALIBRATION_2026_11_03.md).
The earlier apparent total loss of visibility was a lab synchronization
error (server block updates lagged MCio frame samples); its results are
invalid. The opt-in 25x17 first-hit ray profile improves some small or
off-center object recall but still misses truly rendered surfaces, especially
behind glass/leaves. The sensor is **not yet a calibrated recognizability
boundary** and MUST NOT silently become the player's oracle for small,
dark, unrendered, hidden or occluded blocks. Structured game-memory policy
inputs remain the goal in both training and scored gameplay once validated.

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

## New on-demand experimental surface reads (NOT trained yet)

A normal MCio sync action may explicitly request a fresh grouped first-hit
sample on the last policy-decision tick. Missing observations contain no
old world-object data. This **scheduling optimization** was validated
in real Minecraft; it does not solve human-visible eligibility, rendering,
transparency or entity selection. See `SURFACE_REQUEST_V1.md`. Do not
silently add this information to V5 or score it under an unverified schema.

## Experimental off-crosshair objects are NOT policy observations yet

The opt-in `gmaincraft_surface_grid_v0` and `gmaincraft_surface_objects_v1`
MCio/Fabric memory streams now sample bounded **first-collision camera rays**
and group them into block positions and visible ray indices; see
[`SURFACE_SENSOR_V1.md`](SURFACE_SENSOR_V1.md). The new real Minecraft
fixture controls passed 6/6, including off-center oak detection and no
hidden oak behind stone, but glass blocks even visually visible wood,
lighting/FOV/size thresholds remain uncalibrated, and the per-tick read
adds measurable overhead. **No V5 or other learner consumes these streams.**
Do NOT treat raw colliders as universally visually perceivable objects or
turn on the new data silently during scored evaluation. Future V6 inputs
must use validated eligibility, identical train/play contracts and
fresh requested samples with measured overhead. Old V5 status+crosshair
remains untouched and the zero-log model still has no acquired skill.

## NEWEST tested memory interface: V5 own status + vanilla crosshair hit

This user-authorized, reversible, isolated MCio/Fabric 1.6.1 patch now
provides an actual `gmaincraft_status_v1` **read-only** memory option:
player hunger/saturation/armor/air, selected hotbar slot, dimension, plus
ONLY the real vanilla client **crosshair-targeted BLOCK** ID, BlockPos,
face and distance from `Minecraft.client.hitResult`. A crosshair miss
reports no target or coordinates; it does not scan loaded/unseen chunks.
No seed, unvisited structure, hidden ore, closed container or admin action
is available to the playing policy. [Sensor implementation, reversible
build, reach/occlusion tests](MCIO_PLAYER_SENSOR.md).

V5 `factorized_v5_status_v1_player_v1_rgb2_6tick` receives this option
through `MCioAgentBridge.multimodal_status_view()` BOTH in actual CUDA
training and frozen normal survival gameplay, combined with the existing
immutable own-state `player_knowledge_v1` and RGB. Exact block IDs are
encoded by a trainable byte-sequence CNN initialized from scratch; no
hand-coded oak-only world vocabulary and no outside model weights. V5
scored **0/4** actual training log pickups and **0/8** held-out frozen
pickups; no general autonomous play claimed. All historical RGB/V4
architectures and weights remain isolated and unchanged.

**Empirical distinction:** oak targeted at 1.5/2.5/3.5-block front-face
distances and not 4.5–7.5 in tested straight-line fixtures; glass intercepts
the targeted ray, though a person can see through transparent glass. A
crosshair-only hit is **not** the full population of blocks visible on
screen. Real FOV/occlusion/lighting/projected-size range calibration for
a bounded multi-object detector is STILL a future task; do not expand the
memory budget using an arbitrary radius or leak all loaded chunks.

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

**Historical V4 limitation:** the original unmodified MCio/Fabric base
packet omitted hunger and targeted block/entity hits. The new V5 opt-in
patch fixes own hunger and vanilla targeted BLOCK, **not** full
player-perceivable blocks/entities, and older V4 still omits these fields.
Never substitute a raw loaded-chunk map or uncalibrated distance cutoff.
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

## Historical initial code state (BEFORE the V4/V5 milestones above)

`mcio_ctrl.network.ObservationPacket` **already exposes** `player_pos`,
`player_yaw`, `player_pitch`, `health`, `inventory_main`, `inventory_armor`,
`inventory_offhand`, rendered `frame` and optional `options` (including
statistics where configured). `MCioAgentBridge` privately receives these,
but the **current** `src/gmaincraft/agent_interface.py` converts the packet
into an `AgentView` containing **only RGB pixels**. That restriction was the
assistant's original prototype decision, **not what the user requested**.

At the time of the initial design note no player-knowable observation
schema or engine-derived block hit was yet implemented. V4 and V5 now add
those opt-in player state / actionable-target interfaces, but a broader
visibility-filtered world-object list and decisive controlled V5-versus-
RGB learning improvement remain **unproven**. Historical CNN/PPO and
self-supervised pixel diagnostics are preserved but must not turn the
project back into a screenshot-only design.

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
