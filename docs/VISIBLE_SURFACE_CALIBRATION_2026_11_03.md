# Real Minecraft camera-ray calibration: screen evidence, natural terrain and throughput

**Real Java 1.21.3 Fabric/MCio experiments, November 2–3 2026.**
The user wants direct Minecraft memory state bounded by skilled-player
knowledge. This study is EVALUATOR-ONLY evidence to calibrate future object
inputs; no model currently sees these block lists. It does NOT prove general
human-visible block recognition, entity detection, a complete map, successful
natural log pickup or Any% Minecraft speedrunning.

## Crucial invalid first experiment: server/client observation lag

`visgeom_1103a` and the early `visdiag_1103b` were **NOT valid visibility
calibration results**. MCio synchronized observations can advance many
frames within one actual Minecraft server tick. The controller-side `/setblock`
command is accepted before its server update is applied and replicated to
the client. Capturing just four MCio frames afterward often observed the
*previous* staged block, not the requested fixture, while initial paired
screenshots were also unstable. Its zero-detection results must NOT be used
to tune visible-block thresholds or conclude a Fabric sensor regression.

Fix: `experiments/mcio_visibility_calibration.py` explicitly waits 150 ms
of real wall-clock time after each controller-only scene command, then
executes 12 normal synchronized neutral ticks before its sample. A warm-up
with 42 frames validates same-scene screenshot stability. `visdiag_1103c`
passed four known target diagnostics at the expected pose: the 13x9 sensor
found all four genuinely rendered magenta blocks. A newer stability audit
`visstable_1103a` also compared each *post-placement* RGB frame with a
second frame after another real server tick: 4/4 unchanged, same target
membership, no stale position, and both frame and action sequence checks
passed. **Never silently remove this settlement/confirmation protection.**
Fixture server commands are only outside scored gameplay; no agent policy
received the hidden fixture position.

## RGB is an independent witness, not the policy's primary perception

Each lab condition places one opaque magenta concrete block into an otherwise
cleared, fixed camera-facing bedrock corridor, paired with the SAME world,
pose, graphics and time of day with that block absent. Real 320x180 Minecraft
RGB `get_frame()` is compared in
`src/gmaincraft/surface_visibility_audit.py`; the independent threshold
for a pixel change is **at least 40/255** in any channel. The last 28 HUD
rows are excluded. `pixels` also excludes an 11x11 crosshair square;
`pixels_without_crosshair_mask` includes that square to expose why small
centered distant objects can register zero in the crosshair-masked image.
For the study we show several pixel-count sensitivity slices (1, 20, 40,
80 and 150). **None is a validated model-input eligibility threshold.**
Shadows and lighting may alter pixels beyond the object, while recognizability
and visible details can fail even when a few pixels changed. The fixed
experiment runs at FOV option 70, render distance 4 chunks, VSync off and
FPS cap 260; these values are not calibrated universal player FOV/range.

The two tested profile options are:

- **Coarse (existing default)**: 13x9, at most 117 first-collider ray
  samples. Remains default for old script/score compatibility.
- **Fine (new explicit research setting)**: 25x17, at most 425 first-collider
  ray samples. Only enabled for one Minecraft subprocess by
  `GMAINCRAFT_SURFACE_RAY_PROFILE=fine`; both packet adapters validate
  the bounded shape (max 512 rays). The packet includes actual columns/rows,
  so future policy checkpoints MUST schema-tag this choice.

Both are **on-demand** via `SurfaceObjectsRequest(fresh=True)` on the final
normal action tick. Both report grouped first-hit block IDs and coordinates,
not an unobserved chunk inventory. The candidate count is bounded by rays,
not a global world-volume scan. Glass/leaves/fluids and pixel-size limits
remain unresolved. No Minecraft engine item spawning, teleportation or
administrative commands occur during natural-world samples.

## Controlled image-versus-ray evidence: 23 independently staged positions

`viscoarse_1103e` and `visfine_1103a`, both from the same exact pinned
Fabric build before the later vertical-camera fix. Each copied disposable
world staged 23 positions (offsets 0..8 blocks and depths 3..32) with a
settled camera pose of yaw=0/pitch=0. All test packets advanced frames,
all target-absent controls were empty, and untouched-scene RGB remained
stable. Actual raw results reside in ignored
`logs/mcio_visibility_<run-tag>.json`. Relevant paired outcome counts:

| Evidence slice | 13x9 coarse | 25x17 fine |
|---|---:|---:|
| Total staged target positions | 23 | 23 |
| Actual reported target block positions | **16** | **18** |
| Total rays hitting the target across 23 scenes | 28 | 90 |
| Rendered controls with >=80 changed pixels outside HUD/crosshair | 13 of 14 detected | **14 of 14** detected |
| Rendered controls with >=40 changed pixels outside HUD/crosshair | 13 of 15 detected | 14 of 15 detected |
| Rendered controls with >=40 changed pixels outside HUD (crosshair included) | 14 of 16 detected | 15 of 16 detected |

The finer grid recovered `offset1_depth10`, which independently changed
**182** screenshot pixels but was missed by coarse sampling. It also picked
up an extremely small `offset3_depth32` block (20 changed pixels, crosshair
included). It still missed `offset1_depth16` (56 changed pixels outside
crosshair) and some yet smaller/distant blocks. Thus 25x17 is an empirical
improvement on a few fixtures, **not full visual recall**. For the 14
stronger rendered controls (>=80 outside HUD/crosshair), coarse missed one
and fine missed none; denominator is tiny and these are not unseen natural
worlds. A first-hit ray returning a block may convey an exact registry ID
that is *not identifiable* in the actual rendered pixels. Do not conflate
ray hit or 1 pixel with human perceptibility. Centered far targets could be
completely hidden inside the crosshair exclusion yet change the raw image.

### Occlusion, transparency and darkness

`visoccl_1103d` (coarse) and `visocclfine_1103e` (fine) staged one target
behind each independently built barrier or changed its lighting.
Fine-profile target comparison (changed pixels **with crosshair included**):

| Scene | Changed RGB pixels | Sensor hit? | Grounded finding |
|---|---:|---|---|
| Solid stone wall | 0 | No | Correctly stops behind a fully opaque wall |
| Transparent glass wall | 182 | No | Known FALSE NEGATIVE: visible through glass, collision ray stops there |
| Oak-leaf wall | 815 | No | Known FALSE NEGATIVE: pixels differ through visually incomplete leaves, collider masks target |
| One-block slit in stone | 182 | Yes | Genuine through-opening detection, no wall penetration |
| Midnight near target | 154 | Yes | The rendered block still changed pixels in this fixture |
| Midnight distant target | 25 | No | Fine grid missed a small/low-contrast test object |

Raw Java sky/block light on some detected lab blocks was `0/0` even when
screen pixels changed; raw block light numbers are not actual screen
brightness and must not determine human visibility without more testing.
Glass is allowed to intercept the **actionable** vanilla crosshair; the
*perceivable* sensor will eventually require more careful transparency,
fluid, geometry, fog and illumination filtering. No X-ray leak was found
through the tested opaque stone barrier, but the six cases are not a proof
against all Minecraft block types or hidden world objects.

## Actual natural survival terrain: three independent noise seeds

`experiments/mcio_natural_surface_probe.py` validates the source world's
NBT **`minecraft:noise`** generator and exact signed world seed, copies it
to a new uniquely tagged disposable save, and runs **only 300 neutral vanilla
actions per repeat** with no admin commands or edits during either sample.
Two repeats per launch (first is cold chunk loading, second is warm) and
three separate launchers per seed: sensor OFF, on-demand 13x9 and on-demand
25x17. Each has the SAME seed, camera pose and 4-chunk fast graphics
profile, 320x180 genuine NVIDIA RGB, 50 validated frame decodes each
repeat, and **zero action/observation gaps**. Requested profiles receive
precisely 50 fresh observations/300 ticks, with none on other packets.
Every scene is a real Minecraft overworld, not a synthetic flat arena.

Warm 300-tick wall time from **one launch per seed/profile**:

| Natural seed / start XYZ | Surface OFF | 13x9 coarse | 25x17 fine | Mean distinct block surfaces per sample, coarse / fine |
|---|---:|---:|---:|---:|
| 31415927 / (-15.5, 76.0, 16.5) | 0.6761 s | 0.7398 s | 0.8383 s | 59 / 157 |
| 271828183 / (0.5, 77.0, 0.5) | 0.6668 s | 0.6473 s | 0.6719 s | 46 / 81 |
| 161803398 / (19.5, 63.0, 15.5) | 0.6299 s | 0.7158 s | 0.7289 s | 60 / 175 |

Across these three scenes, the fine profile produced substantially more
sampled *first-hit surface positions*, without proof that all 175 objects
are visually recognizable. It saw ordinary wood, leaves, stone, clay,
sand, mushrooms and grass depending on view; unlike the old pixel-only
prototype, this is the intended read-only engine/memory research path.
Relative performance is **scene- and launch-dependent**; e.g. coarse was
slightly faster than the sensor-OFF launcher in seed 271828183, which is
normal run-to-run variance, not magic negative overhead. First 100 cold
game ticks can cost 6–11 seconds while chunks initialize, whereas warm
300-tick repeats are often under 0.85s. Never combine cold-start and
warm simulation rates into claims about policy compute cost. A proper
confidence interval requires multiple runs and interleaved order.

The verified earlier default/fine natural seed sources were created in
prior sessions; they were **not newly generated unseen evaluation seeds**
in this session. Their individual copies were never scored as trained
agent gameplay, only as sensor/throughput controls. Full reports under
ignored `logs/mcio_natural_surface_<tag>.json` and the tag list:

- Seed 31415927: `natsens_base_1103b`, `natsens_coarse_1103a`, `natsens_fine_1103a`.
- Seed 271828183: `natsens_b2_1103a`, `natsens_c2_1103a`, `natsens_f2_1103a`.
- Seed 161803398: `natsens_b3_1103a`, `natsens_c3_1103a`, `natsens_f3_1103a`.

### Separate matched FLAT I/O control (no Torch)

One fixed idle flat world, the same mod build, 300 normal synced game ticks,
second warm repeat, 50 grouped requests when enabled:

| Mode | Warm 300-tick wall time | Frame gaps |
|---|---:|---:|
| Sensor disabled `visperfbase_1103a` | 0.7063 s | 0 |
| 13x9 on-demand `visperfcoarse_1103a` | 0.7554 s | 0 |
| 25x17 on-demand `visperffine_1103a` | 0.7650 s | 0 |

These timings use one launch each. More rays cost little *in this flat
scene* compared with natural forest scenes, but the fine profile is not
free and should not silently replace the default. NVIDIA/CUDA availability
was rechecked; no host driver, service, filesystem mount or user Minecraft
setting was changed.

## Straight-up/down camera basis bug FIXED and tested

Source inspection found a real mathematical degeneracy: when camera pitch
is exactly +90 or -90 degrees, `forward.cross(world_up)` is zero.
Normalizing it collapses all sampled rays to one direction, even though
a player can still see objects off center when looking vertically.
The tracked Java patch now uses the player's yaw to reconstruct a valid
horizontal camera basis ONLY at this singularity; all other camera
orientations use the original normalized cross product. The rest of the
first-hit, occlusion and render-distance logic is unchanged.

Actual live survival probe `vertical_nov03a` used ordinary relative mouse
input to reach exactly **-90 and +90 degrees pitch**, staged an overhead
*off-center* oak block through controller-only lab setup, and proved:
- pitch -90.0 and oak was seen at its exact fixture XYZ by **9 distinct**
  first-hit sample rays (not a collapsed center-only grid), fine shape25x17;
- pitch +90.0 and the floor produced **15 distinct surface objects**;
- all original 8/8 fresh-request / no-stale / action sync controls also
  passed, giving **12/12 real live checks**.

Report: `logs/mcio_surface_request_vertical_nov03a.json`. This verifies a
specific vertical camera degeneracy rather than proving accurate projection
at *every* pitch/FOV; further real renderer alignment remains necessary.

## Exact build, installation and replication

Tracked source-only patch: `patches/mcio-1.6.1-player-status.patch`, pinned
upstream tag `twoturtles/MCio 1.6.1+1.21.3`, original upstream revision
`e89bd1ea5187814a9a3b854dedf3573defac6f8a`. New/current build SHA256
`8cc58f899e7e824d0d67de7b74517b5390dbc3175375ec8c6b7560a2406135b7`.
The preceding fine-grid JAR (`887cbc6efcd0813d5c321006fa053c18f7fa00a4c78dfefb0ab2b02d16a19d37`)
is preserved at ignored `mcio-data/mod_backups/mcio-1.6.1+1.21.3.fine-pre-vertical.jar`.
The earlier on-demand, grouped, raw grid, status-V5 and **pristine** JARs
are also still backed up; `tools/build-mcio-player-status.sh --restore`
reverts this disposable mod. No other mod JAR or personal world was changed.

Run individual launches from `/workspace` with **fresh** tags and headless
Gamescope for fast, synchronized NVIDIA Minecraft RGB. Do not run two
Minecraft instances at once. Each live script refuses to overwrite a
prior world or ignored JSON report:

```bash
.venv/bin/python -m unittest discover -s tests -q
tools/build-mcio-player-status.sh --build
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_visibility_calibration.py --run-tag NEWGEOM --suite geometry --ray-profile fine --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_visibility_calibration.py --run-tag NEWOCCL --suite occlusion --ray-profile coarse --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_surface_request_probe.py --run-tag NEWVERTICAL --vertical-extremes --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_natural_surface_probe.py --run-tag NEWNAT --source-world MDL_NaturalSource_fixedspawn01_01 --profile fine --visible
```

Some Gamescope wrappers still terminate with **139 after the Python child
printed SAVED and closed its world**. A child traceback, failed validity
check or missing JSON report is an actual failure and must not be called
success merely because partial observations were printed.

## Legacy V5 crosshair/status compatibility

After installing the final modified Fabric JAR, the OLD opt-in
`gmaincraft_status_v1` crosshair probe ran on a newly copied real-world
fixture **with the surface-object flags absent**. `statuscompat_1103a`
passed **4/4** client status tests: actual oak under the crosshair,
turning away suppresses oak, camera reset restores oak, and opaque
stone intercepts it. Full ignored result:
`logs/mcio_target_statuscompat_1103a.json`. This is a sensor backward-
compatibility smoke, **NOT** a rerun or proof of successful V5 learned
inventory acquisition. Older checkpoints and policy weights remain intact.

## Decisions and next falsifiable gate

**Keep 13x9 as the backward-compatible default;** 25x17 is an explicit
research-only opt-in with better measured lab recall but variable natural
cost. The relative merits are not settled for a trained policy. Do not
silently convert either research sensor into privileged V5 inputs.

Next: compare actual camera renderer projection at multiple FOV settings,
light/biome/weather and dynamic transitions; learn which visible blocks
are genuinely recognizable under partial leaves/glass/water and other
non-opaque surfaces. Test natural camera motion and newly generated unseen
seeds. Implement bounded, versioned truly player-eligible block/entity
observations, identical for from-scratch training and frozen gameplay.
Then add player-visible block-damage progress / action-history credit and
validate continuous normal vanilla attack/mine/pickup on held-out seeds.
Until a scored policy acquires actual inventory logs without controller
intervention, **first-log collection and Any% remain unsolved**.
