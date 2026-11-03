# MCio/Fabric opt-in memory sensor: own status and actual crosshair target

**Verified live on Minecraft Java 1.21.3, Fabric MCio 1.6.1 + gmaincraft patch, November 2–3 2026.** This is a read-only *player-knowledge* extension, NOT an X-ray map. V5 from-scratch CUDA training and normal frozen Minecraft gameplay both consume it; none of the earlier V1–V4 policy/checkpoint formats were rewritten. Scored gameplay continues to use ordinary vanilla mouse/keyboard input, inventory and world rules.

## Source and reversible installation

- Original MCio upstream: `twoturtles/MCio`, pinned tag `1.6.1+1.21.3`, Git commit `e89bd1ea5187814a9a3b854dedf3573defac6f8a`. Exact source is cloned into ignored `mcio-data/mcio-upstream-inspect` (NOT committed). Uses bundled Minecraft Java 21 and locally cached Gradle/Fabric Loom. An unmodified `compileClientJava` preflight passed before applying the patch.
- **Tracked source patch:** `patches/mcio-1.6.1-player-status.patch`, limited to `MCioNetworkPacket.java` and `MCioObservationHandler.java`. It adds an optional `PlayerStatusOption` to the ordinary MCio observation packet **only when the Minecraft game process has `GMAINCRAFT_STATUS_V1=1`**. The Python side must import `gmaincraft.mcio_status` before receiving the first observation to register the new CBOR subtype.
- **Original mod SHA-256** `21368effcb3b579954deef345c4a8d965ebb667d82bab25723626da406224ab3`; **patched mod SHA-256** `fff86beea0da5c66dc1e78ee8ae450260dace9d5f112120f9c6088388e77ffbb` in this environment. Binary JARs and the original are ignored and never pushed. The original is backed up at `mcio-data/mod_backups/mcio-1.6.1+1.21.3.pristine.jar`. **Only the disposable `DemoInstance/mods` jar is replaced**, no host Gentoo packages, drivers, game installations or filesystem sandbox changes.
- `tools/build-mcio-player-status.sh --build` verifies the pinned upstream commit, exact tracked patch and lack of extra tracked changes, then compiles `remapJar`; `--install` makes/retains the original backup and atomically replaces the single mod jar, refusing unknown divergent jars; `--restore` puts back the backed-up original. Never install or restore while a Minecraft process from this instance is running.

```bash
cd /workspace
tools/build-mcio-player-status.sh --build
tools/build-mcio-player-status.sh --install
# Fully reversible, original kept in /workspace/mcio-data/mod_backups:
# tools/build-mcio-player-status.sh --restore
```

The extension is opt-in **per Minecraft subprocess** through MCio's standard `RunOptions(env_extra={'GMAINCRAFT_STATUS_V1':'1'})`. V5 scripts set this automatically; V4/V2 clients do not. Merely installing the patched jar does not expose new fields to old policies. `PlayerStatusV1.from_packet()` fails if an opt-in run lacks the feature rather than silently inventing player information.

## Policy-eligible game-engine fields

The new `src/gmaincraft/mcio_status.py` validates, copies and freezes a bounded `gmaincraft_status_v1` snapshot containing: actual player hunger, food saturation, armor points, air supply, selected hotbar slot, dimension ID, **plus optionally the targeted block's exact registry ID, BlockPos XYZ, hit face and hit distance**. Missing targets explicitly carry `None` and **no hidden block coordinates**. These are **real values read from the local Minecraft client**, not OCR or server commands.

**Block source of truth:** vanilla `Minecraft.client.hitResult` after Minecraft's normal raycast. A block record is populated ONLY if `HitResult.Type.BLOCK`. We do **not** iterate through world chunks or loaded/unseen blocks. This delivers **one actionable crosshair target**, not the entire collection of blocks perceptible within the screen FOV. An object can be visible but too far away/too far to the side to be targeted. Selecting a rendered target does not give the agent a supernatural ability to mine it: all normal survival interaction requirements remain in force.

`MCioAgentBridge.multimodal_status_view()` returns separate immutable `pixels`, legacy own-state `player_knowledge_v1` and `status` objects. Original `AgentView`, V4 `multimodal_view()`, the private evaluator geometry/reward interface and old checkpoints are unchanged.

## Actual scene controls — both visual rendering and genuine memory reads

`experiments/mcio_target_block_probe.py` creates a **fresh** copy of the DemoWorld, stages an oak trunk at camera-facing z+2 and validates the Minecraft-reported target after real synchronized ticks. Placement, teleport and occlusion block setup are **controller-only lab fixtures** performed outside scored agent gameplay; turning away uses normal `AgentAction` relative mouse movement. Results stored in local ignored `logs/mcio_target_TAG.json`, no world/weights overwritten.

Two independent positive/negative scene runs `targetv1_nov03a` (SDL) and `targetv1_headless_1103` (headless) each **4/4 passed**:

1. Oak trunk directly in line of sight → `minecraft:oak_log`, exact relative position, face and ~1.715-block hit distance.
2. Ordinary mouse turn to yaw 127.5° → `minecraft:bedrock` (not the oak behind).
3. Reset camera to yaw 0°, pitch 29° → oak log restored.
4. Insert opaque stone between player and oak → client reports `minecraft:stone` (NOT hidden oak).

A third real run `reachcal_v1_1103` with extra `--calibrate` produced these **specific** camera-level survival raycast observations (targeted oak's front face is the listed number of blocks away):

| Target front-face distance | Actual oak crosshair hit? |
|---|---|
| 1.5, 2.5, 3.5 blocks | YES at each distance |
| 4.5, 5.5, 6.5, 7.5 blocks | NO at each distance |

**Glass:** an intervening transparent `minecraft:glass` was targeted instead of the oak behind it. This confirms action reach/selection and visibility are different tasks. These measurements **do not establish a universal reach cutoff or general visual detection threshold**; they are seven fixtures in one camera direction/height under one vanilla version. Next calibrate the wider visible candidate set across FOV, distance, occlusion, projected pixel size, leaves, water, darkness and varied natural seeds; do not feed unobserved maps.

Reproduction (unique tag every time):

```bash
cd /workspace
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/mcio_target_block_probe.py --run-tag NEWTARGET --visible --calibrate
```

## A/B performance — the SDL compositor is a second independent bottleneck

The project's earlier natural-world performance investigation found render-distance effects; later testing uncovered a **separate conditional 60-FPS cap** from Gamescope's SDL presentation path. MCio logs `MCIO_UNLIMITED_FPS=true` and disables its own limiter/VSync, yet the present SDL viewer held exactly ~60 game ticks/s, even when its virtual refresh was set to 300 Hz. The same original unmodified MCio jar in headless Gamescope recovered hundreds of ticks/s. No host setting or sandbox access changed.

| Same no-Torch 300-tick flat fixture, 320×180 RGB | Cold measured repeat | Warm second repeat |
|---|---:|---:|
| **Original MCio**, SDL visible window | 4.9523 s | 5.0026 s |
| Original MCio, SDL + nested 300Hz experiment | 4.9759 s | 4.9999 s |
| **Original MCio, headless Gamescope** | **1.1957 s** | **0.7700 s** |
| **Patched MCio, headless**, opt-in status OFF | **1.1472 s** | **0.7140 s** |
| Patched MCio, headless, opt-in status ON and validated on **all 300 ticks** | **1.1559 s** | **0.7226 s** |

No-Torch runs checked sequence parity, frame sequence continuity and actual surviving oak inventory. The patched sensor's measured incremental cost is small (~0.0086 s over 300 warmed ticks in this one pair); do not claim statistical certainty from single-run comparisons. `--visible` still means a GLFW in-game render is active for RGB, **even when the enclosing Gamescope backend has no physical user window**.

The canonical viewer `tools/run-mcio-viewer.sh` now supports optional `GMAINCRAFT_GAMESCOPE_BACKEND=headless` for high-throughput training and `GMAINCRAFT_GAMESCOPE_REFRESH_HZ=300` for the explicit SDL refresh experiment (the latter did not eliminate the cap). Without either variable it keeps the original visible SDL window and mouse-suppression shim so the user can watch the agent. Frame performance is not guaranteed constant across host/GameScope focus or terrain; keep monitoring per episode.

Gamescope can still terminate with wrapper **139 after successful child results**; the actual Python child must complete/save `SAVED`, JSONL, checkpoint first. A separate, earlier Vulkan swapchain assertion during a second Minecraft launch is mitigated by running train and frozen eval in separate viewer sessions.

## V5: truly used by a from-scratch network in BOTH training and gameplay

`src/gmaincraft/status_factorized.py` implements new schema **`factorized_v5_status_v1_player_v1_rgb2_6tick`**, 994,686 random-initialized parameters. Inputs are two RGB84 history frames, a **40-float** bounded own/player-status/crosshair numeric vector and **128 byte tokens** of the exact currently targeted block ID. The trainable character embedding/CNN accepts arbitrary 128-byte block IDs; there is **no manually scripted oak detector or external pretrained weights**. A miss has zero crosshair coordinates/face/ID tokens. The network chooses ordinary 6-game-tick movement/camera/attack; the controller separately gives scalar curriculum shaping without leaking trainer-only target location.

`experiments/joint_log_memory.py --with-status` trains V5 with `GMAINCRAFT_STATUS_V1=1` and saves only **`checkpoints/joint_status_TAG.pt`**; `experiments/memory_frozen_eval.py --with-status` loads the immutable schema and runs frozen eval in a separate Minecraft process; `natural_world_benchmark.py --modes status_factorized --status-model TAG` executes the same policy with normal survival actions and real inventory scoring. V1–V4 code and checkpoint formats are preserved. Use matching 4-chunk/260FPS/no-VSync profile in paired experiments; no hidden administrative commands during scored play.

**Real game proof / no claims of learned success:**

- `statuspilot_1103a`: **4** randomized controlled real CUDA train episodes, one 120-transition PPO update from scratch, **0/4** actual inventory logs; trained V5 checkpoint SHA-256 `77b24ce058fda616eb143432a9ae1c52782e8bd7dfdda0ce0afc353dba59d190` (local ignored file). Last recorded game I/O ~1.61ms/tick.
- `statusfrozen_1103a`: fresh independent copied Minecraft world, frozen exact V5 checkpoint, **8/8** cases actually completed (2 per stationary/aim/near/broad split), **0/8** inventory logs. Most cases ~1.3–1.7ms/tick after warmup.
- `statuscounts_1103b`: new held-out frozen case per phase, **0/4** inventory successes, but every playing policy decision's actually observed hit was recorded. Stationary case saw oak **30/30 decisions** yet **chose no attack** during frozen greedy play. Aim case saw oak **3/30**; near case 1/30 oak, 29/30 bedrock; broad case none. The sensor supplies correct target facts, but this checkpoint has **not** learned sustained vanilla attack, aiming or free discovery.
- `statusnatural_1103a`: genuine noise survival seed **31415927**, confirmed identical starting pose `(-15.5,76.0,16.5)` for random and frozen V5, **0 collected wood in 180 ticks for either**; separate RGB+status frozen policy 30 normal 6-tick decisions, no commands/cheats. MCio game I/O ~6.74ms/tick random vs ~6.99ms/tick V5 including cold chunk loading. Those results establish pipeline parity, **not successful natural-world transfer**.
- **V4 compatibility:** the original `memfast_nov03c` V4 checkpoint loaded and completed four fresh frozen cases in the same patched-mod instance with the extension OFF (`v4compat_1103a`), no model/schema regression; previous V2/V3 checkpoints remain untouched.

Example commands (fresh tags, never overwrite):

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python -m unittest discover -s tests -q
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_memory.py --with-status --run-tag NEWV5TRAIN --episodes 8 --update-every 4 --max-ticks 240 --device cuda --visible --skip-eval
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/memory_frozen_eval.py --with-status --run-tag NEWV5EVAL --model-tag NEWV5TRAIN --eval-seed 817433 --eval 3 --max-ticks 240 --device cuda --visible
.venv/bin/python experiments/natural_world_benchmark.py --run-tag NEWV5NAT --world-seeds 31415927 --modes random status_factorized --status-model NEWV5TRAIN --prepare
GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/natural_world_benchmark.py --run-tag NEWV5NAT --world-seeds 31415927 --modes random status_factorized --status-model NEWV5TRAIN --max-ticks 600 --game-vsync off --game-fps 260 --render-distance 4 --device cuda --visible
```

**Next:** broaden *player-perceivable*, not merely *crosshair-actionable*, block/creature selection after actual calibration and false X-ray tests; add accurate block mining progress and action-duration history where player-knowable; conduct paired RGB-only / numeric / status ablations and sample-efficient learning tests on the SAME held-out scenes before scaling PPO. The first integrated unrestricted find/mine/collect on unseen seeds remains unachieved. Optimize eventually for Minecraft game time to complete Any%, not only training throughput.
