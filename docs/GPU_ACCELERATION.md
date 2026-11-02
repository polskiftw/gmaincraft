# GPU rendering and learning on the isolated Gentoo bridge

**Hardware/environment checked 2026-11-02:** NVIDIA RTX 4070 SUPER 12 GiB,
NVIDIA 595.104.02, Linux 6.18.57-gooned, Minecraft Java 1.21.3 with MCio
and nested NVIDIA Gamescope/XWayland, project `.venv` Python 3.14.10.

## Live CUDA update after the Gentoo Shell restart (2026-11-02)

**CUDA is now fully operational INSIDE the restarted isolated shell.** The
user loaded the host `nvidia_uvm` module, created the UVM character devices
and restarted the bridge. The same bridge now sees:

```
/dev/nvidia-uvm       char 240:0
/dev/nvidia-uvm-tools char 240:1
cuInit(0) = 0
PyTorch 2.15.0+cu132: CUDA available
CUDA tensor allocation and reduction: PASS on RTX 4070 SUPER
```

Measured `tools/benchmark-training.py --batch-size 64 --steps 25 --threads 8`:
CUDA **879.69 updates/s, 56,300 images/s**, CPU **128.99 updates/s,
8,255.5 images/s**: approximately **6.82x GPU speedup** for THIS synthetic
CNN update workload. CUDA peak allocated tensor memory was about 50.2 MiB.
This is **not** a Minecraft environment / gameplay speedup measurement.

A REAL CUDA Minecraft training and evaluation run `cuda01` confirmed model
training on `device=cuda`: 12 training episodes, 220 game ticks each, 3
optimizer updates on 95+63+66 real decisions, all on GPU; 4 separate
frozen held-out scenarios, 220 ticks each. No logs acquired in either
stage (0/12 train; 0/4 eval). The model, GPU training code and checkpoint
system ran successfully, but the learned behavior is NOT solved. Evaluations
took just 3 long actions each, highlighting potential overuse of fixed
80-tick mining macros. Results and checkpoint saved only in ignored local
`logs/joint_log_ppo_cuda01.jsonl` and `checkpoints/joint_log_ppo_cuda01.pt`.
Gamescope still exited 139 AFTER the child Python program finished.

**The CUDA failure description below is historical** (before the restart).
The manual UVM nodes may need to be re-created after a future host reboot;
if so, verify with `tools/gpu-preflight.py` before a new training run.

## The GPU rendering question, resolved

The **nested** OpenGL rendering stack used by this Minecraft client was
measured using the real OpenGL `glGetString` API inside Gamescope/XWayland:

```
OPENGL_VENDOR=NVIDIA Corporation
OPENGL_RENDERER=NVIDIA GeForce RTX 4070 SUPER/PCIe/SSE2
OPENGL_VERSION=4.6.0 NVIDIA 595.104.02
```

The same Gamescope wrapper reports selecting the NVIDIA GPU for Vulkan.
**This is hardware-accelerated OpenGL, not Mesa `llvmpipe` CPU rendering.**
The probe checks the identical nested GLX environment, not an injected query
from inside Minecraft's Java renderer. Minecraft still performs game logic,
world generation, simulation ticks and ZMQ control work on CPUs; GPUs do not
replace those workloads. Screenshots are read back to CPU memory by MCio and
sent as Python RGB arrays before going to the learning network, causing some
unavoidable transfer/serialization overhead.

Reproduce without Portage or desktop settings changes:

```bash
cd /workspace
tools/test-opengl-renderer.sh
```

This compiles a tiny C probe into ignored `mcio-data/` and runs it in the
same nested Gamescope environment. The known **Gamescope exit-139 after
child termination** can still cause the wrapper itself to exit nonzero;
inspect the `OPENGL_*` lines instead. Do not mistake that compositor crash
for software rendering or an OpenGL failure.

## CUDA installation and the previously resolved sandbox blocker

The old 4x14 and 5x12 NumPy models used CPU math. New training uses a
PyTorch 2.15.0+cu132 **888,236-parameter CNN actor-critic**, and can put
parameters, batch forward/backward passes, and Adam updates on GPU. The
user's Python 3.14 compatible CUDA 13.2 build was installed into
`/workspace/.venv`, NOT globally through Portage or system Python.

Historical device status **before restarting the bridge**, retained for troubleshooting:

| Check | Status |
|---|---|
| `nvidia-smi` detects card | PASS |
| `libcuda.so.1` is visible | PASS |
| `nvidia_uvm` kernel module loaded | PASS (after host intervention) |
| `/dev/nvidia-uvm` inside current bubblewrap session | **MISSING** |
| `cuInit(0)` in current bridge | **999 (failure)** |
| `torch.cuda.is_available()` in current bridge | **False** |
| PyTorch CPU forward/backprop/update | PASS |

**Do not claim GPU training is running until an actual CUDA allocation passes.**
On the host *outside* the isolated MCP shell, initialize NVIDIA UVM as root:

```bash
modprobe nvidia_uvm
nvidia-modprobe -u
ls -l /dev/nvidia-uvm /dev/nvidia-uvm-tools
```

The first module loading succeeded, but `nvidia-modprobe -u` did not produce
nodes on this Gentoo system. For this driver/kernel, `/proc/devices` exposes
a dynamically assigned character-device major called `nvidia-uvm` (observed
as `240` on this boot). As root on the **host** (not sandbox), the user can
create both nodes using the *observed* major instead of hardcoding it:

```bash
major=$(awk '$2=="nvidia-uvm" {print $1}' /proc/devices)
test -n "$major" && mknod -m 0666 /dev/nvidia-uvm c "$major" 0
test -n "$major" && mknod -m 0666 /dev/nvidia-uvm-tools c "$major" 1
ls -l /dev/nvidia-uvm*
```

These ordinary NVIDIA CUDA device nodes do not grant access to host files.
Their manually created nodes may need recreation after a reboot if Gentoo's
normal device-manager rules do not create them. **Restart only the user-owned
Gentoo Shell bridge** so bubblewrap binds those device files at sandbox
startup. The existing isolated launcher already includes `/dev/nvidia-uvm`
and `/dev/nvidia-uvm-tools` when they exist. It does **not** require widening
filesystem mounts or system services.

Then run **inside the restarted bridge**:

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
.venv/bin/python tools/benchmark-training.py --device cuda --batch-size 64 --steps 20
.venv/bin/python tools/benchmark-training.py --device cpu --batch-size 64 --steps 20
```

`gpu-preflight.py` must report successful `cuInit(0)`, `torch.cuda.is_available()`
and a real CUDA tensor allocation. If CUDA still fails, retain CPU ability
and diagnose UVM/device privileges; do not disable the filesystem sandbox
or guess about GPU execution.

**Measured local CPU training reference:** 15 64-image updates of this CNN
in 0.159 s, ~94.08 parameter updates/s (~6,021 images/s) with 8 CPU threads,
using a synthetic fixed batch. This is *not* Minecraft simulation FPS nor
PPO training throughput including environment collection. A CUDA comparison
has not yet been measured in this session.

## Performance design and limitations

- The joint learner defaults to **320x180 rendered Minecraft RGB** frames;
  two successive frames are bilinearly resized to 84x84 RGB for the CNN.
  Lower resolution cuts framebuffer and CPU readback/transport costs
  compared with the older 640x360 toy experiments, at possible perception
  fidelity cost. Override `--frame-width`/`--frame-height` for comparison.
- Pass `--device cuda` for **strict** GPU training (fails instead of silently
  using CPU). `--device auto` selects CUDA if available, otherwise CPU and
  explicitly reports the selected backend. `--device cpu` is reproducible
  while fixing UVM. GPU mini-batches should improve heavier visual models,
  but one-frame inference may incur transfer overhead; always benchmark.
- The current PPO network is small. A GPU does not magically make Minecraft
  simulations parallel or improve a poorly designed reward function. Speed
  only matters if the learner transfers to previously unseen natural worlds.
- Earlier 284.5 active MCio steps/s is a *different* persistent-world test
  of Minecraft/ZMQ throughput, not GPU training performance and not full
  reset world throughput.

See [COURSE_CORRECTION.md](COURSE_CORRECTION.md) and
[DEVELOPMENT.md](DEVELOPMENT.md) for the research and runtime plan.
