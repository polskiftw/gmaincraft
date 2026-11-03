# MCio performance recovery and first bounded-memory smoke (November 2, 2026)

**Status:** Fast MCio throughput is reproducible **in the current environment**. The historical 4-second slowdown was not reproduced. This is a recovery/triage finding, **not** a proven root-cause fix. Do not attribute the earlier slowdown to VSync, graphics driver, PyTorch, system CPU governor, Gamescope, or a Minecraft mod without evidence.

## Reproduction method

New no-Torch `experiments/mcio_throughput_probe.py` copies a *new* disposable world from the existing `DemoWorld`, starts Minecraft 1.21.3+MCio in synchronous mode using the normal nested Gamescope viewer, constructs an equivalent small flat log fixture and resets the player (including clear inventory and item entities) before each measured repeat. Each 300-tick test holds vanilla attack 110 ticks, W 35 ticks, then idle 155 ticks. The probe checks every sent sequence is the last accepted action, receives every frame without sequence gaps, updates MCio stats at each step, decodes RGB periodically and scores actual oak log inventory. No Torch or policy decisions. Records send/receive timing and separates ~18 s game launch overhead from measured game steps.

The copied template world is the same source seed and fixture pattern but the spawn location and session runtime may vary between launches. **Within each process** repeats share world and fixture pose, but are not totally independent worlds (dropped entities reset; game RNG/chunk caches persist). This is sufficient to isolate pacing/warm-up, not to claim a fully controlled cross-machine factorial comparison with November 2 historical PPO rollouts.

| Probe tag | Options | Tick 1–300 | Tick 301–600 | Tick 601–900 | Real log, no gaps? |
|---|---|---:|---:|---:|---|
| `baseline_nov02a` | existing instance defaults | 1.2442 s | **0.6717 s** | — | yes (1 log each) |
| `baseline_nov02b` | existing instance defaults | 1.1726 s | **0.7091 s** | **0.6701 s** | yes (1 log each) |
| `capoff_nov02a` | temporary `enableVsync:false`, `maxFps:260` | 1.2017 s | **0.7002 s** | **0.6170 s** | yes (1 log each) |
| `knowledge_nov02a` | defaults + structured state built on every tick | 1.2299 s | **0.7206 s** | — | yes (1 log each) |

**New finding:** fresh-launch first measured episode consistently takes ~1.17–1.24 s; repeats after the first take ~0.62–0.72 s. VSync-off/maxFps260 did not deliver a stable enough improvement over unchanged defaults to justify setting options permanently. The original four-line `options.txt` was restored byte-for-byte after the temporary override. The 0.62–0.72-second results overlap the historical **~0.6–0.8 s** range; the 3.9–5 s slowdown reported in other runs remains unexplained as a transient or workload-specific effect. It was not reproduced in these matched no-Torch trials.

**Per-tick breakdown:** `baseline_nov02b` first repeat spend 1.151 s in `recv_observation` and 0.010 s in `send_action`; the second repeat spends 0.692 s receiving and 0.008 s sending; the third spends 0.651 s receiving and 0.009 s sending. Median receive latency drops from ~2.93 ms in first repeat to ~2.05 ms in third repeat. Frame bytes 172,800 per 320×180 RGB frame. The 0.72 s memory-adapter trial spends ~0.0063 s on stats/cache + structured field construction for all 300 ticks, a small but measured incremental cost. This is diagnostic only, not tuned training performance.

**Integrity:** each measured repeat collected exactly 1 oak log with ordinary survival inputs (no admin commands *during the measured phase*), 0 missing frame sequences, all last-action sequence IDs matched, and the structured adapter's inventory report agreed with the packet. No NVIDIA or filesystem sandbox changes. The Gamescope wrapper still exits with status 139 *after* the experiment child writes the complete valid result files; inspect `SAVED` before treating wrapper status as failed experiment.

## Reproduce safely

Use a **new tag each run**; ignored local disposable worlds, JSON metrics and stdout logs must remain intact:

```bash
cd /workspace
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/mcio_throughput_probe.py --run-tag NEWPERF01 --ticks 300 --repeats 3 --visible
```

Optional, non-persistent graphics experiment: `--vsync off --max-fps 260`. This **temporarily** changes only the dedicated MCio `DemoInstance/options.txt` during the child run and restores its original exact bytes in `finally`. It creates an ignored crash-recovery backup `mcio-data/options_throughput_restore_TAG.bak` only while running. If a hard kill/crash interrupts Python before cleanup, **restore that backup before another run** instead of overwriting options; do not delete it blindly. Avoid parallel games while temporarily editing that same options file. The experiment creates new `MDL_Throughput_TAG` only; no existing world is overwritten.

## Current implementation direction

`src/gmaincraft/player_knowledge.py` introduces **opt-in `player_knowledge_v1`**, accessible via `MCioAgentBridge.player_knowledge()`, using directly reported MCio player XYZ/yaw/pitch, health, owned inventory slots and *known* Minecraft progress stats (`minecraft:mined`, `minecraft:picked_up`, `minecraft:used`, `minecraft:crafted`, `minecraft:broken`, `minecraft:dropped`). Unknown counters are absent, **not asserted zero**. The result contains immutable bounded copies; it does not expose the original game packet, world seed, unobserved blocks/entities, target location, or evaluator commands. The existing RGB `AgentView` remains a separate legacy baseline. This observation is now exercised in live no-Torch play but **is not yet connected to a neural training/evaluation policy**.

The underlying MCio packet does **not** provide hunger/armor points, biome/time/weather, raycast hits or a visible block list. Do not fabricate those values. Additional MCio mod instrumentation should be tested later to expose read-only player-known status and eligible blocks, after visible/actionable eligibility research. The filtered world-object budget and representation remain to be calibrated; see `OBSERVATION_POLICY.md`.

**Next performance falsification:** if a future model experiment again takes >=3 s per 300 ticks, run this no-Torch probe on the same host state immediately before/after with the same 320×180 viewport and action sequence; record FPS/VSync state, resource contention, step recv time and first-run warm-up. Compare with an actual identical neural episode in between. Do not conclude the historical regression is permanently fixed until that discrepancy is reproduced and explained.
