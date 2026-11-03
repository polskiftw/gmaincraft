# Real-world MCio speed: render distance and launch warm-up (Nov 2, 2026)

The earlier ~4s/300-tick slowdown is **reproducible on some natural-scene workloads**, but not in the warmed flat-world control. Hardware OpenGL on NVIDIA RTX 4070 SUPER and CUDA inference are functioning; model inference does not explain this difference. The MCio `SYNC` mod captures a rendered frame after each game tick; work needed to render loaded natural chunks affects `recv_observation` wall time. The observed cause is strongly associated with **displayed chunk/render distance plus first-scene warmup**. This is not a source-code-level proof that there are no other causes.

## Paired render-distance experiment: actual natural Minecraft, no neural inference

Real Minecraft Java 1.21.3/Fabric MCio 1.6.1, seed **246813579**, identical confirmed spawn `(16.5,124.0,0.5)`, idle **600 synchronized game ticks**, 320×180 raw RGB every tick, same simulation input and comparable fresh generated source worlds. `natural_world_benchmark.py --modes idle` measured six 100-tick action holds and recorded true game I/O time **excluding process launch**. No admin commands during scoring. Each row is one complete launch; no claims of statistical replication.

| Options | Total 600 game ticks | First 100 ticks (ms/tick) | Last 100 ticks (ms/tick) |
|---|---:|---:|---:|
| Original unmodified MCio options | **4.5349 s** | 15.99 | 5.51 |
| `enableVsync:false`, `maxFps:260` | **4.2651 s** | 15.28 | 4.88 |
| same + `renderDistance:8` | **3.3872 s** | 10.78 | 3.90 |
| same + `renderDistance:6` | **2.5900 s** | 8.76 | 2.96 |
| same + `renderDistance:4` | **1.6945 s** | 6.16 | 1.79 |

**Measured improvement:** using the 4-chunk profile instead of default gives **2.68× higher 600-tick I/O throughput** here (4.5349 / 1.6945). The last 100 ticks are **3.08× faster** (5.51 / 1.79). Disabling VSync alone made little difference. The first 100 ticks are slower for every run, which is consistent with newly loaded chunks/renderer warmup, but no internal Java profiler has isolated the exact code path. This A/B includes different instance restarts and copied source worlds from the **same verified deterministic seed**; repeating each profile on additional seeds and in varied scenes remains a worthwhile check. The lower render distance also limits distant landscape observations and is a legitimate vanilla settings tradeoff, not a free performance upgrade with identical perceptual range.

**V4 actual learner confirmation:** `memfast_nov03c` trained 10 full Minecraft episodes from random weights at renderDistance4, median MCio I/O **1.48 ms/tick**. The subsequent independent `memfrozen_nov03e` scored 12 frozen cases, median **1.43 ms/tick**. All 22 cases acquired **zero logs**; these runs prove speed and end-to-end wiring, **not trained competence**. The old no-Torch flat-world probe also measured ~0.62–0.72 s/300 after warmup without a special render-distance setting because the flat fixture is cheap to render.

## How to compare new runs

```bash
cd /workspace
# Set a fresh tag per run; render distance is temporarily overridden ONLY
# in the isolated DemoInstance, with a byte-exact backup/restore.
.venv/bin/python experiments/natural_world_benchmark.py --run-tag NEWRENDER4 --world-seeds 246813579 --modes idle --max-ticks 600 --prepare
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/natural_world_benchmark.py --run-tag NEWRENDER4 --world-seeds 246813579 --modes idle --max-ticks 600 --game-vsync off --game-fps 260 --render-distance 4 --visible
```

Re-run for `--render-distance 6` and `8` using new prepare/run tags. Do not compare model outcomes across different display profiles without explicitly labeling the renderer/visibility difference. `src/gmaincraft/instance_options.py` rejects concurrent option overrides and preserves a named crash-recovery backup. The ordinary `DemoInstance/options.txt` was restored after all completed tests. The failed `memfast_nov03c` transition to a second Minecraft triggered Gamescope `vulkan_remake_swapchain()` assertion and left the Python child waiting; **SIGTERM to only that orphaned child** executed the options restoration. It is safer to train with `--skip-eval` then call `experiments/memory_frozen_eval.py` in a new viewer session.

**Status:** We have a grounded, high-impact, user-authorized graphics setting fix for natural-world throughput. Whether **all** of the older controlled PPO ~4 s/300 regression came from render distance, world loading and process state is **not proven**; retain the remaining question in the ledger and gather paired profiler traces if it resurfaces.
