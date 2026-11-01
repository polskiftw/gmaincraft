"""Minecraft Discovery Lab: camera-orientation generalization experiment.

Uses the working navigation_learner.py and existing MCio installation.
Loads its original saved policy, evaluates that policy on fresh targets with
both fixed and random camera yaw, then trains a NEW model from random weights
using ego-relative goal vectors and randomized start positions/yaw.

The agent receives a direction-to-target vector in its own camera coordinate
frame. It does NOT see the world through pixels or learn camera control yet.
All Minecraft commands are experiment-controller setup; the agent only uses
W/A/S/D during episodes. This is not an Ender Dragon speedrunner.

Only overwrites the disposable MDL_OrientationLab_v1 Minecraft world.
Creates: checkpoints/orientation_policy.npz, logs/orientation_training.csv
Leaves checkpoints/navigation_policy.npz and DemoWorld untouched.
"""

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import navigation_learner as original

ROOT = Path(__file__).resolve().parent.parent
TRIAL = 'MDL_OrientationLab_v1'


@dataclass(frozen=True)
class Scenario:
    x: float
    z: float
    yaw: float
    goal: tuple[float, float]


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def make_scenario(arena, rng: random.Random, *, random_yaw=True) -> Scenario:
    # Target and spawn are randomized independently each episode, within
    # the arena boundary. The target is reachable in empty, flat terrain.
    sx = arena.cx + rng.uniform(-3.0, 3.0)
    sz = arena.cz + rng.uniform(-3.0, 3.0)
    angle = rng.uniform(-math.pi, math.pi)
    radius = rng.uniform(3.0, 6.0)
    target = (sx + radius * math.cos(angle), sz + radius * math.sin(angle))
    yaw = rng.uniform(-180.0, 180.0) if random_yaw else 0.0
    return Scenario(sx, sz, yaw, target)


class OrientationArena(original.Arena):
    def reset_scenario(self, case: Scenario):
        commands = []
        if self.marker is not None:
            mx, mz = self.marker
            commands.append(f'setblock {mx} {self.floor_y} {mz} minecraft:bedrock replace')
        mx, mz = math.floor(case.goal[0]), math.floor(case.goal[1])
        commands.append(f'setblock {mx} {self.floor_y} {mz} minecraft:gold_block replace')
        self.marker = (mx, mz)
        commands.extend([
            f'tp @s {case.x:.5f} {self.player_y:.3f} {case.z:.5f} {case.yaw:.4f} 0',
            'clear @s',
            'effect clear @s',
            'effect give @s minecraft:instant_health 1 5 true',
            'effect give @s minecraft:saturation 1 20 true',
        ])
        self.manager = self.env_util.InputStateManager()
        ob = self.send(commands=commands, clear=True)
        for _ in range(40):
            if (math.hypot(ob.player_pos[0] - case.x, ob.player_pos[2] - case.z) < 0.12
                and angular_difference(ob.player_yaw, case.yaw) < 2.0
                and ob.health > 0):
                break
            ob = self.send()
        else:
            raise RuntimeError('Player teleport/yaw not observed after reset')
        return self.advance(2)


class AbsolutePolicy(original.LinearPolicy):
    """Original learned policy; intentionally ignores camera yaw."""
    def features(self, x, z, goal, yaw):
        return super().features(x, z, goal)


class EgocentricPolicy(original.LinearPolicy):
    """New randomly initialized policy: relative goal direction in camera frame."""
    @staticmethod
    def features(x, z, goal, yaw):
        dx, dz = goal[0] - x, goal[1] - z
        distance = max(math.hypot(dx, dz), 1e-6)
        rad = math.radians(yaw)
        # Minecraft yaw 0 faces +Z, +90 faces -X.
        forward = -math.sin(rad) * dx + math.cos(rad) * dz
        right = -math.cos(rad) * dx - math.sin(rad) * dz
        return np.array([forward / distance, right / distance, 1.0], dtype=np.float64)


def run_trial(arena, model, scenario, rng, *, training, decisions, ticks):
    ob = arena.reset_scenario(scenario)
    total = 0.0
    outcome = 'timeout'
    for idx in range(1, decisions + 1):
        before = math.hypot(scenario.goal[0] - ob.player_pos[0],
                            scenario.goal[1] - ob.player_pos[2])
        features = model.features(ob.player_pos[0], ob.player_pos[2],
                                  scenario.goal, ob.player_yaw)
        action, probs = model.action(features, rng)
        ob = arena.move(action, ticks)
        after = math.hypot(scenario.goal[0] - ob.player_pos[0],
                           scenario.goal[1] - ob.player_pos[2])
        reward = (before - after) - 0.025
        if after <= 1.1:
            reward += 4.0
            outcome = 'success'
        elif ob.health <= 0:
            reward -= 4.0
            outcome = 'death'
        elif math.hypot(ob.player_pos[0] - arena.cx,
                        ob.player_pos[2] - arena.cz) > 12.0:
            reward -= 2.0
            outcome = 'out_of_bounds'
        if training:
            model.learn(features, action, probs, reward)
        total += reward
        if outcome != 'timeout':
            break
    return dict(outcome=outcome, ticks=idx * ticks, decisions=idx,
                reward=total, final_distance=after)


def evaluate(arena, policy, cases, *, decisions, ticks, seed):
    rng = random.Random(seed)
    results = [run_trial(arena, policy, case, rng, training=False,
                         decisions=decisions, ticks=ticks) for case in cases]
    wins = [x for x in results if x['outcome'] == 'success']
    return dict(wins=len(wins), count=len(results),
                mean_final=sum(x['final_distance'] for x in results)/len(results),
                mean_ticks=sum(x['ticks'] for x in wins)/len(wins) if wins else None)


def show(label, result):
    time_str = 'n/a' if result['mean_ticks'] is None else f"{result['mean_ticks']:.1f}"
    print(f"{label:25s} {result['wins']:3d}/{result['count']} wins  "
          f"avg final distance={result['mean_final']:5.2f}  "
          f"avg successful ticks={time_str}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--eval', type=int, default=50, help='number of held-out test cases')
    p.add_argument('--decisions', type=int, default=35)
    p.add_argument('--ticks', type=int, default=6)
    p.add_argument('--visible', action='store_true')
    args = p.parse_args()
    if min(args.episodes, args.eval, args.decisions, args.ticks) < 1:
        p.error('All numeric options must be positive')
    prior_ckpt = ROOT / 'checkpoints' / 'navigation_policy.npz'
    if not prior_ckpt.is_file():
        p.error(f'Missing original trained checkpoint: {prior_ckpt}')

    from mcio_ctrl import controller, instance, network, types, world
    from mcio_ctrl.envs import env_util, mcio_env

    former = AbsolutePolicy(seed=17)
    with np.load(prior_ckpt) as old:
        former.weights = old['weights'].copy()
    if former.weights.shape != (4,3):
        raise ValueError('Unexpected original checkpoint shape')
    fresh = EgocentricPolicy(seed=2718, learning_rate=0.12)

    print(f'Copying DemoWorld into ONLY disposable {TRIAL}...', flush=True)
    world.WorldManager(mcio_dir=original.MCIO_DIR).copy(
        'storage', original.SOURCE, original.INSTANCE, TRIAL, force=True)
    opts = types.RunOptions(mcio_dir=original.MCIO_DIR,
        instance_name=original.INSTANCE, world_name=TRIAL,
        mcio_mode=types.MCioMode.SYNC, hide_window=not args.visible,
        gpu_lib=None if args.visible else 'nvidia',
        width=640 if args.visible else 320,
        height=360 if args.visible else 180)
    output = ROOT / 'logs' / 'orientation_training.csv'
    checkpoint = ROOT / 'checkpoints' / 'orientation_policy.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    launcher = instance.Launcher(opts)
    ctrl = None
    try:
        launcher.launch(wait=False)
        ctrl = controller.ControllerSync(action_port=opts.action_port,
            observation_port=opts.observation_port, connection_timeout=120)
        ctrl.send_action(network.ActionPacket(clear_input=True))
        first = ctrl.recv_observation()
        if first is None:
            raise RuntimeError('No initial Minecraft observation')
        arena = OrientationArena(ctrl, first, mcio_env, env_util, network)
        arena.prepare()
        cases_rng = random.Random(60211)
        cases = [make_scenario(arena, cases_rng) for _ in range(args.eval)]
        fixed_cases = [Scenario(c.x, c.z, 0.0, c.goal) for c in cases]

        print('\nPrevious checkpoint generalization test:', flush=True)
        show('Original (fixed camera)', evaluate(arena, former, fixed_cases,
                    decisions=args.decisions, ticks=args.ticks, seed=771))
        show('Original (random camera)', evaluate(arena, former, cases,
                    decisions=args.decisions, ticks=args.ticks, seed=771))
        print('\nNew model initialized from random weights:', flush=True)
        show('New BEFORE training', evaluate(arena, fresh, cases,
                    decisions=args.decisions, ticks=args.ticks, seed=771))

        train_rng = random.Random(81003)
        start = time.perf_counter()
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode','outcome','ticks','decisions','reward','final_distance'])
            writer.writeheader()
            for ep in range(1, args.episodes + 1):
                case = make_scenario(arena, train_rng)
                result = run_trial(arena, fresh, case, train_rng,
                    training=True, decisions=args.decisions, ticks=args.ticks)
                writer.writerow({'episode': ep, **result})
                f.flush()
                if ep % 25 == 0 or ep == 1 or ep == args.episodes:
                    print(f"  training {ep:4d}/{args.episodes}: "
                          f"{result['outcome']:>13s}, "
                          f"distance={result['final_distance']:.2f}, "
                          f"reward={result['reward']:+.2f}", flush=True)
                if ep % 25 == 0 or ep == args.episodes:
                    np.savez_compressed(checkpoint, weights=fresh.weights,
                        baseline=np.array(fresh.baseline), updates=np.array(fresh.updates))
        elapsed = time.perf_counter() - start
        print('\nFROZEN checkpoint evaluations:', flush=True)
        show('New AFTER training', evaluate(arena, fresh, cases,
                    decisions=args.decisions, ticks=args.ticks, seed=771))
        extra_rng = random.Random(99501)
        extra_cases = [make_scenario(arena, extra_rng) for _ in range(args.eval)]
        show('New extra unseen cases', evaluate(arena, fresh, extra_cases,
                    decisions=args.decisions, ticks=args.ticks, seed=771))
        np.savez_compressed(checkpoint, weights=fresh.weights,
            baseline=np.array(fresh.baseline), updates=np.array(fresh.updates))
        print(f'Learning time: {elapsed:.2f}s')
        print(f'New checkpoint: {checkpoint}')
        print(f'Training log: {output}')
        print('NOTE: Goal direction is still provided numerically; this does NOT learn vision or mouse turning.')
    finally:
        if ctrl is not None:
            try:
                ctrl.send_stop()
            except Exception as e:
                print(f'Minecraft stop warning: {e}', flush=True)
            ctrl.close()
        launcher.close()

if __name__ == '__main__':
    main()
