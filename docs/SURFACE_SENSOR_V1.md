# Experimental camera-surface memory, raw grid v0 and grouped blocks v1

## Later measured visibility/throughput study

For 23 independently staged real-Minecraft visual scenes, stone/glass/leaf
controls and three natural noise seeds, consult
[`VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`](VISIBLE_SURFACE_CALIBRATION_2026_11_03.md).
A separate explicit fine 25x17 research profile exists in addition to the
backward-compatible 13x9 default; the tracked patch also fixes straight
up/down camera ray geometry. Earlier fixed 13x9 experiments below remain
valid historical baselines, not current general visibility guarantees.

**Live verified in real Minecraft Java 1.21.3 MCio/Fabric, 2026-11-02/03.**
These are OPT-IN research sensors only. **NO trained model currently consumes
these block lists.** The canonical policy knowledge boundary is
[`OBSERVATION_POLICY.md`](OBSERVATION_POLICY.md), not every loaded Minecraft
block. Do not claim that this is a validated human-visible block detector.

## Newer on-demand optimization

The on-demand action-triggered grouped sensor has now been built and tested
in real Minecraft. See [SURFACE_REQUEST_V1.md](SURFACE_REQUEST_V1.md).
It avoids full per-tick scans and does not replay stale block observations.
The experiments below are the earlier continuous-sampling baseline.

## Design and reproducible build

The pinned `twoturtles/MCio` 1.6.1 Fabric patch
`patches/mcio-1.6.1-player-status.patch` retains the verified V5
`GMAINCRAFT_STATUS_V1=1` own status and actual vanilla crosshair. Two
additional independent options are disabled unless a Minecraft subprocess
sets the corresponding environment variable:

- `GMAINCRAFT_SURFACE_GRID_V0=1` emits `gmaincraft_surface_grid_v0`:
  **13 columns x 9 rows**, at most 117 individual first-block collision
  hits with block ID, block XYZ, hit face, exact sampled ray index,
  distance and the block's sky/block light. Decode via
  `src/gmaincraft/mcio_surface_grid.py` before the first CBOR observation.
- `GMAINCRAFT_SURFACE_OBJECTS_V1=1` emits
  `gmaincraft_surface_objects_v1` and groups those SAME rays by Minecraft
  block position, with a face bitmask, nearest hit distance, 0..15 light,
  and the exact hit-ray indices. The bounded Python adapter
  `src/gmaincraft/mcio_surface_objects.py` calculates **sample occupancy**,
  screen-space sample center and sample bounds. These are neither exact
  rendered pixels nor a claim the user can identify the block visually.
  Duplicated rays do not repeatedly transmit a block's registry ID.

A `13x9` pinhole-style frustum uses the current Minecraft FOV **option**
(70 degrees in test fixtures), window aspect ratio, player eye position,
view direction and **current game render distance** (4 chunks = maximum
64-block sampled ray length in the tested setup). Every sample calls
Minecraft `Level.clip` with `ClipContext.Block.COLLIDER` and `Fluid.NONE`,
and includes ONLY the FIRST hit if its chunk is loaded. No raw chunk
iteration, full world map, hidden seed/structures, or hidden underground
ore list is sent. Rays through solid stone and glass stop at the surface.

**Important missing eligibility research:** true in-game dynamic FOV/projection
versus FOV option; visible blocks between grid rays; projected minimum
screen area; world fog; light **as the actual player sees it** (raw
sky-light is not visual brightness); looking vertically where the chosen
right/up basis degenerates; leaves, glass, transparent/non-colliding
shapes, fluids, darkness, particles and entities. A ray that hits a distant
opaque surface may not mean that Minecraft actually rendered enough pixels
to recognize its ID. Engine-side colliders are not equivalent to pixels.
Until calibrated, these fields are research observations and NOT policy
features in train or scored play. This prevents premature confidence in a
sample threshold and avoids silently creating a privileged observation.

Rebuild and install only in the isolated disposable DemoInstance:

```bash
cd /workspace
tools/build-mcio-player-status.sh --install
.venv/bin/python -m unittest discover -s tests -q
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/mcio_surface_grid_probe.py --run-tag FRESHGRID --grouped --visible
```

The build script verifies upstream source/patch bytes, refuses unknown JARs,
and saved all verified pre-upgrade versions in ignored
`mcio-data/mod_backups/`:

| Variant | SHA256 (actual local JAR) |
|---|---|
| Pristine upstream | `21368effcb3b579954deef345c4a8d965ebb667d82bab25723626da406224ab3` |
| Previous status V5 | `fff86beea0da5c66dc1e78ee8ae450260dace9d5f112120f9c6088388e77ffbb` |
| Initial sample grid v0 | `9ff0456aa6e37d7201477466eb97581ba31ddc5974e4358dbf3aeb7b06f245fc` |
| Current grouped+grid patch | `8d1152104e6646c59f963456b2d3c243b6c87b0e4b6d2ca9773b8fbb88a34fb3` |

`tools/build-mcio-player-status.sh --restore` restores the pristine mod;
do not replace JARs with Minecraft running. The trained V5 model still
only sees its older `status_v1` option and does not acquire world blocks
through this research-only opt-in. No user host packages or normal
Minecraft worlds were modified.

## Real on-screen versus memory experiment

Separate original ray-grid `surfacegrid_nov03a` and grouped-parity
`grouped_nov03a` fresh copied Minecraft worlds ran **real**
NVIDIA/MCio synchronized game frames. All commands for staging blocks,
wall and pose were strictly experimental controller-only fixture preparation,
not learned play. Full per-ray/per-object logs are local ignored files
`logs/mcio_surface_grid_<tag>.json`.

| Scene | Raw first-hit rays | Distinct grouped blocks | Oak samples | Actual center crosshair | Outcome |
|---|---:|---:|---:|---|---|
| Center oak | 57 | 29 | 12 | Oak | PASS: oak in grid and real crosshair |
| Oak 2 blocks off center at z+4 | 53 | 33 | 4 | Miss | PASS: perceivable candidate outside crosshair |
| Opaque stone wall ahead of oak | 112 | 16 | 0 | Stone | PASS: no hidden oak leaked |
| Transparent glass wall ahead of oak | 112 | 16 | 0 | Glass | PASS: collision filter blocks oak; **known false negative for visual visibility** |

The 6th live grouped check compared **every** grouped ray index, block ID,
XYZ, nearest distance and visible face to the raw ray stream in all four
scenes; exact parity passed. Four positive/negative scene controls plus
that grouped parity check all passed. No remote/hidden world data or
real survival score is inferred from these lab commands. The sample FOV
and ray layout still need an empirical sensitivity/false-positive analysis.

## Matched live throughput (no Torch; 300 synced ticks)

Tests used headless Gamescope with **real 320x180 NVIDIA RGB frames**, same
pinned fixed fixture at `(-82,64,-9)`, neutral/idle 300 actions, same
Minecraft graphics option state and fresh copied worlds. Each independent
launcher did two consecutive 300-tick repeats; **compare the second/warm
repeat**. No frame sequence gaps in any measured repeat; untouched input
inventory remains empty as expected for idle behavior. No pretraining or
policy inference. These are single-launch measurements, NOT a robust
multi-seed confidence interval or Java microprofiler.

| Run tag | Sensor flag | First repeat | Warm repeat | Game frame gaps |
|---|---|---:|---:|---:|
| `idlebase_v1jar_nov03a` | No added surface data, current JAR | 1.1256 s | **0.6746 s** | 0 |
| `idlebase_nov03a` | No added data, previous grid-only JAR | 1.2191 s | 0.6980 s | 0 |
| `idlegrid_nov03a` | Raw grid v0, grid-only JAR | 1.3475 s | **0.9497 s** | 0 |
| `idleobjects_nov03a` | Grouped blocks v1, current JAR | 1.4207 s | **0.8898 s** | 0 |

Grouped output in the fixed scene represented 35,100 raw ray hits as
17,100 grouped block records over 300 ticks. In this small sample,
grouping reduced warm time by ~6.3% relative to the raw format, but
still added ~32% versus the **same-current-JAR** no-surface baseline
(0.8898 vs 0.6746 s). Not all runs share one JAR build or runtime load,
so do NOT call either comparison a definitive causal speedup.

**NEXT performance gate:** measure on natural daylight/darkness and varied
scenes, consider *explicit observation-request sampling on policy decision
ticks* (every 6 in V5), rather than scanning and encoding the entire grid
on every game tick. Never silently send stale samples as if current; record
sample age and ensure an actual requested observation on each model decision.

## Gate before any new V6 learner

Calibrate actual rendered FOV, false X-ray leaks vs missing visible blocks,
transparent blocks, projected size, light/night, distance, entities and
multi-scene natural worlds. Determine sampled ray resolution and bounded
object budget **from measured recall, false positives and throughput**, not
from an arbitrary geometry radius. Verify same object view in train/frozen
and real survival; then attempt a new schema-tagged from-scratch learner
with action history / sustained attack and held-out real inventory scoring.
Current V5's true log collection remains 0 on prior held-out trials.
Do not burn large PPO runs just because an experimental sensor emits data.

Wrapper Gamescope exit **139 after the Python child prints `SAVED`** remains
an observed teardown problem and does not invalidate durable test JSON.
If the Python child fails or no result file exists, the experiment failed.
