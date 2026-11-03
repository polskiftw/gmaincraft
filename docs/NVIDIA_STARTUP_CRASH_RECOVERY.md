# Nov 3, 2026: actual NVIDIA SIGFPE, stale MCio wait, training lock

## What happened in the real user run

The first five `speedrun01` survival seeds COMPLETED, and the last
atomic checkpoint, `checkpoints/fresh_stream_speedrun01.pt`, contains
`next_episode=5`, **12,000 committed decisions**, **125 neural updates**.
The journal records **100 ordinary blocks mined**, 489 first-hit wood
surfaces, 2 deaths and **0 real wood inventory**. The client was
starting seed episode5 (sixth seed), recorded by the independent
`logs/fresh_stream_speedrun01.inflight.json` marker, when the JVM
crashed before a valid Minecraft observation was returned.

The preserved `mcio-data/instances/DemoInstance/hs_err_pid3518301.log`
identified `SIGFPE` with `FPE_INTDIV` (integer division by zero) in
**`libnvidia-glcore.so.595.104.02`**, on Microsoft OpenJDK 21.0.7,
about 3.6 seconds after Java start. That is native graphics-driver-side
failure, NOT evidence of a Python policy failure or corrupt learned
weights. It does **not** establish Firefox, driver competition, or the
nested Gamescope compositor as the cause: these are possible interaction
hypotheses that require independent reproduction or driver traces.

MCio `ControllerSync.recv_observation()` ignores its `timeout` parameter.
Its native ZMQ receiver polls in a loop indefinitely, so a dead Minecraft
JVM can leave the *Python trainer alive and holding its exclusive flock*
while no future observation is possible. A second trainer launched from
an independent bubblewrap PID namespace therefore printed `Same seed-
stream run already active`, even though `ps` inside the newly connected
shell showed no game process. **Do not unlink the .lock file**: flock is
inode-specific and replacing it while the old process exists could permit
two writers to corrupt the single saved PPO model. Closing the original
shell/process is required. During repair, the old lock later became FREE
without any unsafe lock deletion or second trainer writes.

The abrupt crash also left
`mcio-data/instances/DemoInstance/options.gmaincraft_speedrun01.backup`.
The exact original bytes were verified against the expected temporary
`renderDistance=4`, `maxFps=260`, `enableVsync=false` edits, restored to
the disposable DemoInstance and archived separately to the ignored logs.
The checkpoint, journal, in-flight marker and incomplete sixth world
were deliberately preserved pending safe resume.

## Changes made to prevent indefinite hangs and false recovery

- `src/gmaincraft/mcio_process_watchdog.py` supplies a bounded native
  synchronous-ZMQ receiver. It checks the exact `Launcher.poll()` JVM
  subprocess on every wait and enforces a genuine 30-second observation
  deadline instead of calling MCio's ignored timeout. Connection startup
  is bounded to 50 seconds, and pending stop requests interrupt waits.
  Regular packet delivery still uses the existing MCio socket/CBOR and
  exact ordinary gameplay actions; no game-memory schema or agent
  learned checkpoint changes.
- `experiments/fresh_seed_treadmill.py` catches an unexpectedly exited
  client or timed-out observation, records a distinct error, **discards
  any uncommitted neural/optimizer changes**, restores the last valid
  checkpoint and its PyTorch RNG, quarantines the uncommitted disposable
  world and retries the **same seed** (default at most two retries).
  Exceeding retries safely aborts with the committed model intact rather
  than looping forever or claiming success. A clean `rc=0` Java exit
  before an observation is still an unexpected exit, **not diagnosed as
  an NVIDIA crash without an actual JVM error report**.
- `src/gmaincraft/instance_options.py` restores an interrupted run's
  exact disposable graphics settings on resume **only when** the current
  file precisely matches the expected temporary edit. Unknown user edits
  or another run's backup fail closed and remain untouched.
- `tools/run-fresh-treadmill.sh` checks the *real flock* BEFORE launching
  another Gamescope/Java instance and gives a useful error if another
  namespace still owns it. It does not invent or remove lock ownership.
  The journal+checkpoint verification contract remains unchanged.
- Tests cover dead-JVM observation wakeup, running-but-stalled timeout,
  valid packet receipt, stop interruption, exact settings restoration,
  and refusing to rewrite foreign/modified options.

## Actual live test and caveats

New independent run tag `watchdog_smoke_1103`: the first Minecraft child
unexpectedly exited `rc=0` before the first valid observation. The new
watchdog **detected** it instead of hanging; the trainer discarded that
attempt, quarantined its disposable world, regenerated the same seed and
successfully completed **12 real survival decisions, one CUDA PPO update**
and saved a matching model and JSONL report. Wrapper returned success only
after validating both. This validates the retry and rollback mechanism,
NOT a fix for native `libnvidia-glcore` faults themselves.

As of the verification after the additional driver-workaround test, the
original `speedrun01` flock is FREE and its last verified checkpoint is still the intact
12,000-decision/125-update model. The user can run
`tools/run-fresh-treadmill.sh resume speedrun01 24000` in the ORIGINAL
host shell, with the bridge connected. The runner should quarantine the
interrupted sixth world, retry that same seed and continue from episode5.
If a native NVIDIA SIGFPE repeats, record new `hs_err_pid*.log` and test
alternative isolated compositor/backend or updated driver **separately**;
never claim Firefox caused it from this single crash report.

This is a software watchdog/recovery repair, not a GPU driver patch.
No host Gentoo NVIDIA module, Firefox setting, user game save or older
checkpoint was altered. Model feature encoding and old experiments stay
identical; no need to retrain from random weights.

## Isolated NVIDIA threading workaround (experimental)

Public NVIDIA Linux forum users reported that disabling the NVIDIA
OpenGL threaded optimizer worked around **other** Minecraft crashes in
`libnvidia-glcore` (different driver release and crash context). This
is a **possible workaround**, not a verified root-cause fix for the
user's `SIGFPE/FPE_INTDIV` on 595.104.02. A new per-run Python option
`--nvidia-gl-threading off|default` applies
`__GL_THREADED_OPTIMIZATIONS=0` **only to the spawned Minecraft JVM**,
not to the host Gentoo session, Firefox, Gamescope or the NVIDIA kernel
module. The optional `default` value restores native driver behavior.
It is deliberately not part of the learned observation, model metadata
or training reward, so existing `speedrun01` weights can resume unchanged.

New real Minecraft `nvidia_gl_safe_1103` trained 12 decisions/one CUDA
update, 0 retries, valid saved checkpoint with per-client NVIDIA threaded
optimization off; vanilla noise source generation6.092s, client startup
9.237s, and 9.109ms/tick over just 72 early warmup game ticks. This
short smoke is NOT a benchmark establishing throughput impact or proof
against rare GPU failures. The previous default-driver smoke recovered
from one early `rc=0` exit and completed after retry. On a repeat native
SIGFPE, preserve the next Java `hs_err_pid*.log` and compare carefully.
