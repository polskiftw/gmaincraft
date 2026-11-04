# Optional zero-RGB MCio memory transport (November 4, 2026)

## What changed

The active natural-survival V9 / replay-Q policy already consumes bounded,
structured Minecraft memory rather than screenshots. Earlier, every MCio
observation nevertheless included an OpenGL RGB frame (320x180) and the
common action bridge unnecessarily converted the final image to NumPy.
Commit 440d857 removed that unused Python conversion/copy.

A separate, opt-in Fabric patch now skips glReadPixels and the large
GPU-to-CPU readback, sends zero image bytes in MCio observations and
keeps the camera frame_sequence and normal game tick synchronization.
The Minecraft renderer still runs; this does NOT mean true renderless
headless simulation. MCio uses the capture callback for synchronized
observations, so the new implementation deliberately invokes the same
callback with an empty buffer rather than switching it off.

## Controls

- Default mode: unchanged RGB capture and wire payload for legacy
  screenshot experiments and independent visibility calibration.
- Setting GMAINCRAFT_MEMORY_ONLY_NO_RGB=1 for the replay-Q collection
  or frozen evaluation shell wrappers enables --memory-only-no-rgb.
- The fresh-seed trainer passes GMAINCRAFT_MEMORY_ONLY_FRAME_V1=1 only
  to its disposable DemoInstance Minecraft child process.
- Python asserts empty image bytes, dimensions 0x0 and a valid advancing
  capture sequence, failing closed on an incompatible JAR or bad packet.
- Historical RGB view rejects empty images instead of pretending to
  provide pixels.
- Transport preference is metadata-versioned ONLY when enabled; older
  checkpoint metadata and player-policy inputs remain unchanged.

## Verification steps

1. Build the pinned source patch with build-mcio-player-status.sh --build.
2. Run unit tests, check shell syntax and original checkpoint metadata.
3. Only AFTER all running Minecraft evaluations have exited, install
   the compiled mod with --install (the preceding JAR is backed up).
4. Run the exact same pretrained frozen Q model on the same new heldout
   seed indices and action RNG, first with RGB, then with the new flag.
5. Verify initial and subsequent actual observation packets, crosshair
   and first-hit surface inputs, normal gameplay and real inventory gains.
6. Compare game_io_ms_per_tick and end-to-end wall time on matched worlds.
   Do not claim speed gains from theoretical packet savings alone.

## Limitations and next research

The mod still renders the entire Minecraft scene, just without reading
the image pixels back or sending them to Python. True renderless
simulation needs a separately proven game-tick / observation protocol.
The geometric sensor still has measured misses through glass/leaves,
small visible objects and missing dropped-item entities. Those require
versioned visibility work, independent of raw RGB transfer. A frozen
policy must not change its sensor contract while being evaluated.


## Verified real Java game behavior

Pinned Fabric JAR SHA256:
42bd0fa0afb680d5d25a1edad3a0d6681750f7cce140986b8fff3d223dd3331f.
Installed in only disposable DemoInstance after older eval closed.
Previous 28fd3209 build backed up as mining-feedback-ondemand.jar.

Frozen Q zero-RGB flag passed one 3600-tick survival seed and then six
eval seeds (21600 real game ticks). Frame sequence advanced with empty
frame payload, fresh surface/mining observations valid and zero optimizer
updates in evaluation. Default flag OFF passed live RGB probe: 10
180x320x3 screenshot decodes, 10 grouped surface samples, zero frame
and action sequence gaps. Outer Gamescope wrapper again reported
signal139 after successful saved report; do not confuse with failure.

SAME frozen Q / SAME six eval seed indices / SAME action RNG:
RGB I/O 2.5756ms/tick, wall168.26s; no-RGB I/O 2.2209ms/tick,
wall158.38s. Measured I/O time reduction 13.77%; overall wall
time reduction 5.87%. This is ONE paired benchmark, not a CI.
Wood acquired differed (RGB8, no-RGB6): stochastic action/game timing
can lead to different trajectories. Sensor/reward policy schema did
not change, but full event-by-event behavioral equivalence is unproven.
