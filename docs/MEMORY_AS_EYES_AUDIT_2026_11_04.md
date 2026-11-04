# Memory as eyes: verified audit (Nov 4, 2026)

## Current behavior

Active Minecraft natural-survival V9 and Q models already consume structured, bounded engine-derived player observations, not screenshots. Inputs include player position, own inventory, vanilla crosshair, mining/crack history and up to 24 camera first-hit block objects. Legacy RGB experiments are separate and must remain reproducible.

Previously the common action bridge decoded and copied the final RGB frame on every call even if the memory-only learner ignored its returned image. The backward-compatible return_view=False option now skips that Python decode/copy in natural-survival training and evaluation; original return_view=True remains default. It does not stop MCio's OpenGL glReadPixels capture, packet serialization or RGB traffic. MCio game-step synchronization currently relies on the capture callback, so disabling it without replacing the handshake could cause hangs.

## Visibility: geometry is not yet human visibility

Current 13x9 and experimental 25x17 finite first-collider ray grids group the sampled surfaces. In validated staged scenes, fine saw 18/23 staged blocks versus 16/23 coarse. Glass and leaves produced misses, some small distant visible objects were skipped and exact registry IDs may exceed what human pixels could recognize. This argues for camera-projection / rendered-face / adaptive ray-based visibility research, *not* blindly exposing loaded-chunk contents. Separate visible, actionable, and previously remembered facts and avoid magically updating past observations. Use frames for sparse independent audit, not a required neural input.

## Next practical gates

1. Benchmark no-decode versus old path on matched real Minecraft runs; unit checks alone do not quantify speedups.
2. Design capture-independent tick synchronization and opt-in RGB-free wire protocol, retaining RGB compatibility, then benchmark CPU, GPU, bytes/tick and whole-run throughput.
3. Improve actual visual occlusion handling for transparent blocks, leaves, fluids, fog, lighting, small and moving objects; use independently captured control images for accuracy.
4. Expose bounded player-visible dropped item entities as a new versioned observation, and evaluate on unseen worlds with consistent train/evaluation inputs.
5. Keep completed Q run frozen and perform fresh heldout seed plus independent action-RNG comparison before claiming generalizable ability.

Completed Q run qwood_v2long1104: 100 fresh training worlds, 120000 six-tick units, 3724 cumulative CUDA Q updates, 283 actually mined wood blocks and 219 recorded wood inventory gains. Valid completed journal/checkpoint; launcher segfault happened after saved run. These are training outcomes, not proof on heldout seeds. Preserve versioned checkpoints and all older artifacts.
