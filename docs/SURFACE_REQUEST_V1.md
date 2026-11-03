# Synchronized camera observation requests

## Later real-scene calibration (see new results)

The exact same synchronized request protocol now supports a bounded,
**explicit** 25x17 research profile as well as the default13x9, and the
camera basis remains valid at +/-90 pitch. See
[`VISIBLE_SURFACE_CALIBRATION_2026_11_03.md`](VISIBLE_SURFACE_CALIBRATION_2026_11_03.md)
for all 23 controlled rendered-image controls, actual natural survival
seed timings, opaque/glass/leaf/night failures and the newer Java build.
Neither research profile is a policy input, and these tests cannot be
mistaken for an autonomous first-log skill.

## Protocol

This experiment adds opt-in grouped surface observations on specific Minecraft actions.
The option is research-only; V5 models and checkpoints are unchanged.

Set `GMAINCRAFT_SURFACE_OBJECTS_V1=1` and
`GMAINCRAFT_SURFACE_ON_DEMAND_V1=1` in the isolated Minecraft process.
A normal synchronized MCio `ActionPacket.options` may contain
`SurfaceObjectsRequest(fresh=True)`. The new grouped surface observation
is generated for that action only. Each request clears after an observation,
so unrequested packets never repeat old sampled positions.

Import `SurfaceObjectsRequest` from `gmaincraft.mcio_surface_objects`
before CBOR decoding. `MCioAgentBridge.apply(AgentAction(ticks=6),
request_surface_objects=True)` sends a request on the action's last tick.
Immediately read `bridge.surface_objects()`. Reading an unrequested packet
raises instead of returning stale data. The old continuous observation
mode is unchanged without the on-demand environment flag.

This verifies the synchronous MCio protocol, not asynchronous mode.
Game mechanics and ordinary mouse/keyboard inputs remain unchanged.

## Live Minecraft checks

`experiments/mcio_surface_request_probe.py` passed 8/8 checks in a new
copied survival world (`demandbridge_nov03a`): initial no-data; fresh oak
on request; synchronized final-tick reply; no data after unrequested input;
decoder fails closed; mouse turn removes old oak; one-tick request succeeds;
sequence IDs advance. Ignored report: `logs/mcio_surface_request_demandbridge_nov03a.json`.

The fixed idle scene at (-82,64,-9) used real 320x180 NVIDIA RGB, headless
Gamescope and 300 synchronous ticks. Each launch measured a cold and warm
300-tick repeat with no frame gaps. Only warm times are compared below.

| Benchmark tag | Surface schedule | Cold | Warm | Sample count |
|---|---|---:|---:|---:|
| `demandbase_nov03b` | Same mod, sensor off | 1.3648s | 0.7302s | 0 |
| `demand_nov03a` | Same mod, one per six ticks | 1.1818s | 0.7140s | 50 |
| `idleobjects_nov03a` | Previous mod, every tick grouped | 1.4207s | 0.8898s | 300 |
| `idlegrid_nov03a` | Previous mod, every tick raw | 1.3475s | 0.9497s | 300 |

The new on-demand path uses six times fewer scans and avoids most previous
per-tick serialization overhead. Warm sampled time 0.714s and no-sensor
0.730s are within between-run variability. Neither zero overhead nor a
statistically significant speedup is established. Replicate on natural worlds.

## Rebuild and reproduce

The tracked Fabric patch now includes `MCioClientSync.java` to receive
observation-request options. JAR SHA256 at the time of this milestone (superseded by the visibility-calibration build):
`7e89d4d7475e255ba48df70ba688ad7205dab3e96c5bbf2008c35ea4116e618b`.
The earlier grouped build `8d1152104e6646c59f963456b2d3c243b6c87b0e4b6d2ca9773b8fbb88a34fb3`
is backed up in ignored `mcio-data/mod_backups/` along with pristine,
status-V5 and grid-only JARs. Mod operations touch only DemoInstance.

```bash
cd /workspace
.venv/bin/python -m unittest discover -s tests -q
tools/build-mcio-player-status.sh --build
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_surface_request_probe.py --run-tag NEWBRIDGE --visible
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/mcio_throughput_probe.py --run-tag NEWPERF --ticks 300 --repeats 2 --fixed-fixture --idle-only --surface-objects --surface-on-demand --visible
```

Some nested Gamescope wrappers return 139 after the child writes its
`SAVED` report. Always inspect the actual JSON and the child completion.

Next: calibrate ray candidates against true rendered visibility, distance,
lighting, partial occlusion, transparent blocks, FOV and entities. Benchmark
multiple random natural worlds, then version the observation in both
training and frozen play and revisit learned sustained mining and pickup.
No learned general first-log collection has been demonstrated yet.
