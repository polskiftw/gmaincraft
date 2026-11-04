# V2 replay Q next experimental steps: heldout check, zero-RGB and new data

## Provenance

The original V2 long natural-survival run qwood_v2long1104 finished
100 new noise-generated train worlds on exact TRAIN indices 206..305,
120000 six-tick units / 720000 real Minecraft ticks, 3724 cumulative
optimizer updates. Outcomes: 283 real wood blocks mined, 219 recorded
wood item inventory gains, 49 of 100 worlds with >=1 wood gain, and
176 deaths. These are training counts, NOT generalized skill proof.
Committed immutable frozen network: replay_q_qv2longf1104.pt.

The V2 snapshot retains actual original V1 self-play and both V2 smoke
source lineages, then the exact 100-world continuation source. It does
not import an external pretrained policy or expert-specified route.

## Unseen-world frozen comparison

On independent unseen EVAL stream indices 10..15 with 600 six-tick
units per seed, matched original V2 qv2triplef1104 versus new
qv2longf1104 with two independent Q action RNGs:

- RNG 813777: original 5 wood items / 3 of 6 worlds with pickup /
  1 death. New 8 wood / 2 of 6 / 4 deaths.
- RNG 1639377: original 4 wood / 1 of 6 / 4 deaths.
  New 5 wood / 2 of 6 / 6 deaths.
- Combined 12 paired seed/RNG runs: original 9 wood / 4
  positive runs / 5 deaths; new 13 wood / 4 positive runs / 10 deaths.

The improvement in harvested item count is modest and NOT a significant
demonstration of reliable transfer. The newer policy has more deaths
and no broader success across the sampled world/RNG pairs. Need
additional heldout seed/root, stochastic control and better resource
visibility/reach/survival before claiming success. Offline frozen-on-same
3379 genuine actual observations also shows new stationary-attack
probability on 274 wood-under-crosshair states around 0.331 versus
starting V2 around 0.489; do not misinterpret train reward as policy
skill. This Q model remains local wood-crack practice, not Any%.

Raw old-RGB records in logs/fresh_eval_qv2long_eval10a.jsonl,
qv2base_eval10a.jsonl, qv2long_eval10b.jsonl and
qv2base_eval10b.jsonl. Repeat with tools/run-replay-q-evaluation.sh;
the evaluator refuses to overwrite tags. Summary tool:
tools/analyze-frozen-q-worlds.py.

## Transport performance experiment

See MCIO_STRUCTURED_ONLY_TRANSPORT_2026_11_04.md. The installed
experimental patch is opt-in: default RGB behavior preserved.
The first real single-world frozen zero-RGB run qv2frame_smoke12a
successfully completed 3600 game ticks and validated the actual
zero-byte frame format and frame-sequence advancement throughout.
Its I/O time was 2.418ms/game-tick vs 2.507 for the old RGB
evaluation of the same seed12. Those are different full-run action
sampling histories (old run had played seeds10-11 already), so
wood outcomes are not a deterministic parity test. Do not claim
a universal speedup from this single sample.

A full matched six-world same-start/RNG frozen zero-RGB test is
qv2frame_perf10a; compare against qv2long_eval10a using
tools/compare-memory-transport.py AFTER the new journal completes.

## New bounded collector gate

Only after checking a completed real zero-RGB frozen run with all
packet validation and original RGB compatibility: run a NEW distinct
train experiment starting at TRAIN seed index306, never overwrite the
prior run or use an eval seed as training evidence. The immutable
qv2longf1104 weights and protected true self-play outcomes are the
source. Keep all original optimizer/goal/action observation semantics
and replay hyperparameters, changing ONLY the tested zero-RGB
transport (new versioned metadata). Proposed:
20 fresh worlds x 1200 six-tick units = 24000 units. This is a
bounded engineering/data-collection continuation, NOT a claim that
more identical wood-crack reward teaches crafting or End strategies.

Run with exactly the same environment on resume, same total target.
The shell wrapper holds an advisory flock. Do not remove lock files or
run two Minecraft JVMs against the same DemoInstance. Record train
world true inventory, death counts, I/O throughput, and later frozen
heldout seed/RNG comparisons; preserve the source checkpoint.


## Bounded continuation ACTUALLY LAUNCHED

Live Q replay tag qv2memtrain1104, train index306, source immutable
qv2longf1104. Total 24000 six-tick units, 1200 per world (20 total),
32 CUDA Q updates per completed world, recent32, batch48, nonwood
anchor2, success weight0.6, 20GiB player-only replay cap.
GMAINCRAFT_MEMORY_ONLY_NO_RGB=1 enables verified transport only.
Launcher detached; real Minecraft seed306 3739478096723277114 generated,
first scene started at [0.5,73.0,0.5]. Initial checkpoint shows
cursor306, decisions0, rare-outcome V2 schema and new transport metadata;
exclusive flock independently verified HELD. Inspect committed
episodes before claiming training results. Source weights unchanged.

Safe status: tools/run-replay-q-collection.sh status qv2memtrain1104
Safe stop: tools/run-replay-q-collection.sh stop qv2memtrain1104
