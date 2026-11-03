# V4 from-scratch player-memory + RGB policy (November 2, 2026)

**State: implemented and tested in real Minecraft; DOES NOT YET LEARN unrestricted first-log acquisition.** No existing RGB-only V1/V2/V3 checkpoint was modified or reused. The V4 network is initialized from random weights, trains by acting in Minecraft, and receives **the same player-knowledge observations in training and frozen play**.

## Exact observation/action design

- `src/gmaincraft/player_knowledge.py` defines the immutable `player_knowledge_v1` snapshot read directly from the MCio client packet. It contains exact own position XYZ, yaw/pitch, health, owned main/armor/offhand item IDs/counts, and available player-owned accumulated statistics. Unknown statistics are absent rather than fake zeros.
- `MCioAgentBridge.multimodal_view()` combines **that exact snapshot** and an immutable rendered RGB frame; *both* training and normal survival evaluation call it. `src/gmaincraft/memory_factorized.py` exports `encode_player_knowledge()` with a **versioned 20-float input representation**: scaled positions, fractional x/z, yaw sine/cosine, pitch, health, own counts of logs/planks/sticks/occupied slots and known mined/picked-up log stats with availability flags. See `FEATURE_NAMES` for names and ordering. Feature vectors cannot contain hidden block coordinates, target locations or world seeds. Some requested own status fields (hunger, biome, air, dimension) are not available in the current MCio packet and are **not fabricated**.
- RGB is optional **architecturally**, but the present V4 prototype still uses two downscaled 84×84 RGB frames alongside player memory. A 976,430-parameter CNN plus numeric MLP feeds separate 6-way movement, 5-way camera, 2-way attack heads and a value critic. The schema is `factorized_v4_memory_v1_rgb2_6tick`, incompatible with all old checkpoints. Every action is vanilla keyboard/mouse with a six-game-tick hold, including consecutive left-button holds. Controller-provided target geometry affects **scalar training rewards only**, never sensor features or scored actions.
- Training uses `experiments/joint_log_memory.py` with the same `joint_log_factorized.run_episode` reset/score logic, valid persistent cumulative mined-event tracking, and real inventory counts. Frozen held-out evaluation uses `experiments/memory_frozen_eval.py` to avoid spawning a second Minecraft inside the same Gamescope instance. Natural-seed evaluation is `experiments/natural_world_benchmark.py --modes random memory_factorized --memory-model TAG`, with separate verified natural survival source worlds and paired spawn.
- The RGB legacy `AgentView` and all previous checkpoints remain unchanged. New local `.pt` files, JSONL logs and world copies are Git-ignored. Player coordinates do **not** disclose a privileged goal vector, but absolute coordinates should be tested for overfitting to specific arenas/seeds. Additional block/entity observations need a visibility-filtered engine sensor and held-out calibration, not raw chunk maps.

## Real run ledger (all November 2, 2026, local timestamps UTC+0 early November 3)

| Run | World/mode | Minecraft results | MCio I/O |
|---|---|---|---|
| `mempilot_nov03b` | 4 controlled train (one 120-transition CUDA update) | **0/4** real log pickups | ~2.60–4.45 ms/tick during train |
| `mempilot_nov03b` | 3 held-out frozen controlled cases | **0/3** pickups | ~3.17–4.57 ms/tick |
| `memnatural_nov03a` | natural seed 31415927, random and frozen V4, each 180 ticks | **0/1** each | original default graphics, game launch included in wall time |
| `memnatural_nov03b` | natural seed 246813579, random and frozen V4, each 180 ticks | **0/1** each, matched spawn `(16.5,124,0.5)` | random **12.96**, frozen V4 **11.87** ms/tick |
| `memfast_nov03c` | 10 additional V4 train episodes, 2 CUDA updates from **random** weights (no warm start) | **0/10** collected; **0** recorded real oak mine events | median **1.48 ms/tick** on 10 train episodes, tuned renderDistance4 |
| `memfrozen_nov03e` | separate fresh Minecraft, same `memfast_nov03c` frozen checkpoint | **0/3** each restricted stationary, restricted aim, unrestricted near, unrestricted broad; **0/12 total** | median **1.43 ms/tick** across 12 eval cases |

The `memfast_nov03c` **training completed and saved the final checkpoint** but Gamescope then aborted during the second Minecraft launch with `vulkan_remake_swapchain()` assertion. No frozen cases ran in that same process. A local orphaned Python child was terminated via SIGTERM, which restored `DemoInstance/options.txt` from its byte-exact backup; all world saves/checkpoints stayed intact. `memfrozen_nov03e` subsequently completed all 12 cases and wrote final results, with the original graphics settings again restored. Gamescope's usual post-child wrapper exit 139 is separate from child success.

The frozen checkpoint is `checkpoints/joint_memory_memfast_nov03c.pt`, SHA-256 **`c57d5aba99010a704fc332f5f5a7bf1dc598c8de6224a037613531781a30014f`**, 10 train episodes/2 updates, seed triple policy/train/eval `921331/921333/921339`; independent frozen eval seed `921351`. This checkpoint is local/Git-ignored and remains an **unsuccessful prototype**. The invalid historical `v3event01` checkpoint is never used.

## Graphics performance controls and exact commands

V4 training defaults to the measured fast but still vanilla `renderDistance:4`, `maxFps:260`, `enableVsync:false` for the **disposable MCio instance only**. `temporary_game_options()` restores the original byte-exact `options.txt` when the child exits normally; if the shell kills it forcibly, check for `mcio-data/instances/DemoInstance/options.gmaincraft_TAG.backup` and restore before any new override. Never change the user's normal Minecraft installation or weaken filesystem isolation.

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python -m unittest discover -s tests -q
# Run NEW random-initialized V4 training. Recommended: skip eval here because
# some nested Gamescope builds crash on a second game within one process.
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_memory.py --run-tag NEWTRAIN --episodes 10 --eval 3 --max-ticks 240 --update-every 5 --device cuda --visible --skip-eval
# Separate, truly frozen held-out evaluation in a FRESH GameScope process;
# inherits and enforces the TRAINED checkpoint's memory/graphics schema.
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/memory_frozen_eval.py --run-tag NEWEVAL --model-tag NEWTRAIN --eval-seed 421739 --eval 4 --max-ticks 240 --device cuda --visible
# Prepare real Minecraft noise terrain with a NEW source tag before scoring:
.venv/bin/python experiments/natural_world_benchmark.py --run-tag NEWNATURAL --world-seeds 31415927 --modes random memory_factorized --memory-model NEWTRAIN --prepare
# Pass the model's same 4-chunk display profile to BOTH paired controllers:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/natural_world_benchmark.py --run-tag NEWNATURAL --world-seeds 31415927 --modes random memory_factorized --memory-model NEWTRAIN --max-ticks 600 --game-vsync off --game-fps 260 --render-distance 4 --device cuda --visible
```

Never re-use a run tag that created a world/result/checkpoint (including partial/crashed runs). The frozen evaluator rejects wrong observation feature names and action durations. Historical V2 checkpoint/trainer files are left intact.

## Scientifically next

**Stop blindly scaling PPO** after this zero-success smoke. The V4 pipeline proves direct-memory parity and performance, not general tree perception. The next material capability is a read-only, engine-derived **targeted/visible block sensor** (including actual vanilla raycast and player-known own hunger/status), with explicit opt-in/versioned MCio protocol and no X-ray candidates. We inspected original `twoturtles/MCio` source at tag `1.6.1+1.21.3` (kept ignored under `mcio-data/mcio-upstream-inspect`): its `MCioObservationHandler` currently serializes player health/XYZ/yaw/pitch/inventories and stats, **not** current crosshair hit, hunger, or visible block IDs. Research bounded visibility rules (frustum, projected area, occlusion, translucency, lighting, reach) on varied held-out real scenes before adding larger object lists to the neural input.

Once a valid player-visible world-object observation exists, test direct memory+RGB versus unchanged RGB-only and numeric-ablated controls on exactly paired near/broad scenes and newly generated natural seeds. Preserve all prior checkpoints/logs and avoid mistaking restricted success for free gameplay competence. The final objective remains from-scratch Any% on unseen survival seeds, not hardcoded controllers.
