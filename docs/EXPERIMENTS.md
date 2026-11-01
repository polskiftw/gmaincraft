# Experiment record

This is the **verified evidence ledger**, not a roadmap. Add measured results here only after reading actual runtime output and identify evaluation conditions. Current records were reported from the user's Gentoo machine on 2026-11-01. Retain caveats and avoid converting synthetic tests into Minecraft evidence.

| Test / script | Verified output | Interpretation / limitation |
|---|---|---|
| MCio demo | Minecraft 1.21.3 runs, player state and RGB frames received; viewer visible with XWayland and `cbor2<6` | Control connection works, not an RL result |
| `random_agent.py` | Two random-action episodes ran and restarted a clean saved-world copy | Baseline only, no training |
| `benchmark_agent.py` windowed | Two 1,000-step runs, 192.6 and 197.3 active steps/s; 195.0 average | Active simulation only, single machine |
| `benchmark_agent.py --headless` | 163.7 and 207.2 active steps/s; 185.5 average. Whole 2,000-step run 55.505s | Small-sample headless variance; full startup overhead large |
| `persistent_benchmark.py` | Five 1,000-step runs: 208.5, 300.4, 332.6, 314.6, 306.1 steps/s. Combined active 5,000 steps / 17.58s = 284.5 steps/s; whole run 40.50s = 123.5 steps/s | Soft reset ~0.08–0.11s; **not a full world reset** |
| `reward_probe.py` | First oak log +1, second oak log +0, stone pickaxe +1, reset, oak log +1; overall PASS | Controller `/give` tested event detection; agent did not acquire items |
| `navigation_learner.py` | Before: 7/20 targets, avg final distance 5.49 blocks, avg successful time 98 ticks. After 120 training episodes: 20/20, avg final distance 0.70, avg successful time 24 ticks. 12.8s learning time | Real Minecraft movement + online scalar reward updates. Fixed yaw, known numeric goal direction. One held-out set and one run; no statistical confidence estimate |
| `orientation_learner.py` | **PENDING** — script authored and syntax checked; synthetic training logic tested. User plans to supply real run output next | Do not mark as proven until measured in Minecraft |

## Measurement conventions

- `steps/s` is MCio synchronous environment step throughput, not general Java FPS and not neural-network updates/s.
- `active` excludes Minecraft start/stop and world reset; `end-to-end` includes those costs. State which one is used.
- Navigation success requires reaching within the controller's configured radius of a chosen gold block on a flattened test arena. It does not require seeing/identifying gold in the screenshot; the target vector is provided.
- Training and evaluation data should be disjoint by explicit RNG seed, with seed sets and action/reward protocol reported. The 20-target improvement alone does not establish robustness.
- Fast soft reset retains changes to Minecraft world; distinguish from clean save-copy full restarts.

## Known nonfatal logs

- Minecraft's Linux narrator could not load `libflite.so`; game and MCio experiments still worked.
- Some launches timed out looking up Mojang profile properties; the game still loaded.
- These are not currently reasons for system-wide dependency changes.

## Future entry template

### YYYY-MM-DD — experiment name

- Git revision:
- Runtime: Minecraft/MCio/Python; visible or headless; machine:
- Training/world seeds; test seeds; policy starting state:
- Observation contract; available actions; reward function; reset mechanism:
- Episodes/steps/time; baseline and frozen eval:
- Actual raw summary, not inferred: 
- Limitations and failures:
- Checkpoints/logs location (local, ignored by Git):
- Follow-up TODO and corresponding code commit:
