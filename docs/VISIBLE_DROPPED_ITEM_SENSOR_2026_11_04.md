# Visible dropped-item entity sensor: experimental v1 (Nov 4, 2026)

## Why

The current natural survival learner can mine wood and read the vanilla
crosshair/own inventory but does not receive information about dropped
ItemEntity objects. It can mine a tree and walk away from a log on
the ground simply because its learned observation has no dropped item.
More identical training on the wood-crack reward will not solve this
observation gap by itself.

## Design (RESEARCH ONLY, not a trained observation)

A separate, opt-in Minecraft Fabric sensor has been implemented:
- Environment: GMAINCRAFT_VISIBLE_ITEMS_V1=1 and
  GMAINCRAFT_VISIBLE_ITEMS_ON_DEMAND_V1=1.
- Python MCioAgentBridge.apply(request_visible_items=True,
  return_view=False) requests this action's current item observation.
- Candidates are from a local maximum 24-block zone, bounded further
  by the render distance; a candidate must fit in the camera projection
  and have at least one unobstructed Minecraft VISUAL ray among five
  samples on its bounding box. Very tiny projections below ~2 pixels
  are rejected. At most 16 nearest items are reported.
- Each accepted item has only a registry ID, XYZ centroid, camera
  screen UV, distance and number of visible samples. Never give
  unknown hidden item positions, NBT, stack counts, full loaded-world
  entity enumerations, loot tables or privileged pickup commands.
- No-request observations must not replay old item data. The parser
  fails closed on absent or malformed records.
- Existing V9/Q neural policy and all trained checkpoints are unchanged.
  Feeding this sensor into learning will require a new versioned
  observation schema and fresh train/eval agreement.

## Verification and limits

Pure Python request/shape tests and a real pinned Fabric Gradle build
passed. After the ongoing frozen heldout Minecraft evaluations exit,
install with tools/build-mcio-player-status.sh --install (which will
back up the current mod), then run the unique disposable fixture:

GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh \
.venv/bin/python -u experiments/mcio_visible_items_probe.py \
--run-tag itemv1lab_1104a --visible

It stages drops and stone/glass/leaf blockers using controller commands
only in a test fixture, NOT inside scored self-play. Test for exposed
item visible, stone-occluded item hidden, camera-turned-away item hidden,
and empty/unrequested no-stale results. Glass/leaves are measurements,
not pre-assumed success. The normal RGB frame is used only as a lab
compatibility witness, never as policy input.

Important uncertainty: finite geometric rays and an exact registry
ID do not automatically prove pixel-level human recognizability,
especially behind leaves/glass, in fog/darkness or at low pixel size.
Do not connect this candidate sensor to trained policies until
calibrated. Future work: real mined-log dropped-entity pickup and a
learned, goal-conditioned navigation/pickup skill with vanilla inputs.
