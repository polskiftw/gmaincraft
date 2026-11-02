# Joint visual learning: phase 2 of the course correction

This stage implements a *first* shared visual reinforcement-learning model,
**not** a complete Any% learner. It is designed to expose scaling limits
so we can reject ineffective architecture choices early.

## Implemented and tested

- `src/gmaincraft/vision_ppo.py`: randomly initialized PyTorch convolutional
  actor-critic (888,236 trainable weights). Two 84x84 RGB frame observations
  are stacked, without handcrafted target-color masks or target coordinates.
  Separate policy logits and value head share the CNN trunk.
- One policy selects among 11 vanilla actions: forward, back, left, right,
  look L/R/U/D, jump+forward, wait, and a **fixed 80-game-tick attack hold**.
  The attack hold is an explicit motor macro, **not learned duration**;
  inventory, hotbar/UI/crafting, placing, continuous mouse motion and
  nontrivial memory are not yet learned. All policy actions flow through
  the `AgentAction` / `MCioAgentBridge` contract with no admin commands.
- `src/gmaincraft/ppo_math.py`: action-duration-aware GAE (semi-Markov
  discount per *actual Minecraft game tick*), returns, reward boundaries
  across terminal episodes; PPO clipped surrogate, Adam, normalized
  advantages, gradient clipping and a value loss. Unlike the former linear
  learners, rewards affect whole trajectories, not just one step.
- `src/gmaincraft/joint_curriculum.py` +
  `experiments/joint_log_ppo.py`: vary player start, actual oak-log block
  position and camera yaw/pitch on every training episode. Privileged target
  geometry **only shapes scalar training reward**, never policy input.
  Evaluate frozen model on separately seeded held-out scenarios.
  Success requires actual vanilla mined-log **inventory** acquisition.
  Curriculum still edits a disposable flat bedrock world at episode reset;
  it is not genuine survival-world skill.
- `experiments/natural_world_benchmark.py --modes joint_ppo --joint-model TAG`:
  load a trained model without updating weights and attempt wood acquisition
  on naturally generated survival worlds, with fresh copies / fixed spawn,
  normal game mechanics and no privileged target rewards.
- `tools/gpu-preflight.py`, `tools/benchmark-training.py` and the OpenGL
  renderer probe separate the GPU rendering question, CUDA training access,
  and compute throughput from game-step throughput.

## Real Minecraft smoke evidence (2026-11-02)

Training tag `jointsmoke01`: 2 distinct variable-position planted-log episodes,
140 game ticks each, **0/2** acquired logs. PPO performed one 34-transition
batch update on the actual collected visual trajectories, producing finite
loss and a valid local `checkpoints/joint_log_ppo_jointsmoke01.pt` checkpoint.
Frozen evaluation in a separate world with two new camera/position cases,
140 ticks each: **0/2** logs. All four Minecraft episodes completed cleanly
at the Python level; Gamescope exited with its known post-child 139 error.
Training ran on **CPU**, because CUDA UVM device nodes were missing in the
current sandbox. The original pilot used a 640x360 rendered viewer; newer
joint and natural-world CLI defaults reduce input frames to 320x180. This smoke test validates the training loop, NOT successful
learning or useful policy generalization.

Natural-world transfer tag `jointtransfer01`: generated real Minecraft
seed `161803398`, normal noise terrain, spawn radius 0, and loaded the frozen
`jointsmoke01` checkpoint as a pure pixel-driven player. The 120-tick run
completed, **0 natural logs**, no cheating or Python/game errors. Again,
this is pipeline validation only; two training episodes cannot support
claims about learning quality.

## Warm-start / checkpoint continuity (smoke verified)

Training writes a checkpoint **atomically** after each configured update
batch and includes network weights, optimizer moments and metadata. A new
run can continue from an older checkpoint using `--init-model PREVIOUS_TAG`
and a **NEW** `--run-tag`, `--train-seed` and `--policy-seed`. The script
rejects repeated source training and policy seeds to avoid accidentally
replaying identical training cases. This continues training but is **not**
an exact mid-episode resume: game world, current trajectory and sampler
stream are regenerated as a new tagged experiment. For interrupted runs,
use the latest valid checkpoint; never overwrite the original local files.

Real-world smoke `warmcheck01` used a local `jointsmoke01` checkpoint (made
before optimizer state was added), trained **one** new 100-tick case on CPU,
completed a second PPO update on 23 transitions, and wrote an 11 MB
checkpoint containing the optimizer. Outcome: **0 logs**. Because this pilot
reused the older source RNG seed, it **replayed a scenario**, so this is
only proof that loading/saving works, not independent learning evidence.
The newer code explicitly rejects such repeated seeds going forward.

Example, after CUDA is actually available:

```bash
cd /workspace
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_ppo.py --run-tag joint03 --init-model joint02 --policy-seed 4202 --train-seed 670102 --eval-seed 890102 --episodes 100 --eval 20 --max-ticks 320 --device cuda --visible
```

## Developer commands

Inside isolated Gentoo Shell (choose fresh run tags; never reuse an existing
checkpoint/world tag):

```bash
cd /workspace
.venv/bin/python tools/gpu-preflight.py
# Real-game, variable-position training; requires functioning CUDA device:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/joint_log_ppo.py --run-tag joint02 --episodes 100 --eval 20 --max-ticks 320 --device cuda --visible
# Build a genuine, held-out world and evaluate frozen model:
.venv/bin/python experiments/natural_world_benchmark.py --run-tag transfer02 --world-seeds 246813579 --modes idle random forward_attack joint_ppo --joint-model joint02 --prepare
# Run the actual independent seeded-world evaluation:
tools/run-mcio-viewer.sh .venv/bin/python -u -B experiments/natural_world_benchmark.py --run-tag transfer02 --world-seeds 246813579 --modes idle random forward_attack joint_ppo --joint-model joint02 --max-ticks 2000 --device cuda --visible
```

The examples are **commands, not records of runs**. Local NVIDIA/CUDA
passthrough must be verified first. Each `--prepare` generates a source
world only once. A large multi-mode evaluation launches several independent
Minecraft clients and may exceed a single shell action timeout. Use
fresh tagged runs, preserve ignored checkpoints, record *actual* scores.
The natural benchmark's 320x180 frame resolution is not the original
640x360 used by the early handcrafted RGB toy learners; do not compare
cross-resolution results without stating it.

## Next research decision / gaps

1. **Fix GPU CUDA** on the isolated bridge and compare CPU/CUDA throughput.
2. Train/evaluate across several independent policy/training seeds and
   many more than two episodes on variable real log positions. Check
   genuine inventory outcomes and always-forward/always-attack baselines.
3. Measure whether learned skills transfer to *naturally generated* worlds
   on never-trained world seeds. Avoid using a pilot seed as a final test.
4. If this 2-frame CNN + timed PPO fails, improve memory/representation,
   exploration/reward design and reset throughput **based on evidence**.
5. Add real crafting/UI/environment progression to the same uninterrupted
   episode once first-log acquisition is proven. The final target remains
   from-scratch learned Any% on unseen seeds; that target is unsolved.

Do not mistake good reward curves, synthetic tests, GPU utilization or toy
mined logs for Any% speedrunning ability.
