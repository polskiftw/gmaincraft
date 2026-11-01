"""Minecraft Discovery Lab: first *real* reinforcement-learning smoke test.

Learns W/A/S/D navigation from scratch in a disposable flat Minecraft arena.
Uses a tiny NumPy policy-gradient policy and scalar distance/progress rewards.
There are no scripted move-toward-goal controls. The model learns which key
moves it toward the goal by trial and error. This is deliberately NOT the final
Minecraft-speedrunning model: the test temporarily supplies a target vector,
uses a fixed camera yaw, and does not use visual observations yet.

Requires the existing MCio DemoInstance, DemoWorld and Python virtualenv.
Only overwrites the disposable MCio world MDL_NavigationLab_v1.
Outputs: checkpoints/navigation_policy.npz, logs/navigation_training.csv
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MCIO_DIR = ROOT / "mcio-data"
INSTANCE = "DemoInstance"
SOURCE = "DemoWorld"
TRIAL = "MDL_NavigationLab_v1"
KEYS = ("W", "A", "S", "D")


class LinearPolicy:
    """A trainable single-layer neural policy, initialized randomly."""

    def __init__(self, seed=17, learning_rate=0.14):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.01, size=(4, 3))
        self.baseline = 0.0
        self.learning_rate = learning_rate
        self.updates = 0

    @staticmethod
    def features(x, z, goal):
        dx, dz = goal[0] - x, goal[1] - z
        distance = max(math.hypot(dx, dz), 1e-6)
        # Exposes *direction to goal*, not a hardcoded key mapping.
        return np.array([dx / distance, dz / distance, 1.0], dtype=np.float64)

    def probabilities(self, features):
        logits = self.weights @ features
        logits -= np.max(logits)
        probs = np.exp(logits)
        return probs / probs.sum()

    def action(self, features, rng):
        probs = self.probabilities(features)
        return int(rng.choices(range(4), weights=probs, k=1)[0]), probs

    def learn(self, features, action_index, probs, reward):
        # Online policy gradient with an EMA advantage baseline.
        # Only receives scalar reward; no correct-action labels exist.
        advantage = float(np.clip(reward - self.baseline, -3.0, 3.0))
        direction = -probs.copy()
        direction[action_index] += 1.0
        self.weights += self.learning_rate * advantage * np.outer(direction, features)
        self.baseline = 0.98 * self.baseline + 0.02 * reward
        self.updates += 1


class Arena:
    def __init__(self, ctrl, observation, mcio_env, env_util, network):
        self.ctrl = ctrl
        self.network = network
        self.mcio_env = mcio_env
        self.env_util = env_util
        self.cx = math.floor(observation.player_pos[0]) + 0.5
        self.cz = math.floor(observation.player_pos[2]) + 0.5
        self.floor_y = math.floor(observation.player_pos[1]) - 1
        self.player_y = self.floor_y + 1
        self.marker = None
        self.manager = None
        self.last = observation

    def send(self, *, inputs=None, commands=None, clear=False):
        pkt = self.network.ActionPacket(
            inputs=inputs or [], commands=commands or [], clear_input=clear
        )
        self.ctrl.send_action(pkt)
        ob = self.ctrl.recv_observation()
        if ob is None or ob.version != self.network.MCIO_PROTOCOL_VERSION:
            raise RuntimeError("MCio returned an invalid observation")
        self.last = ob
        return ob

    def advance(self, steps):
        for _ in range(steps):
            self.send()
        return self.last

    def prepare(self):
        # Flatten only a copied, throwaway 31x31 region; never touch DemoWorld.
        x, z = math.floor(self.cx), math.floor(self.cz)
        y = self.floor_y
        commands = [
            "difficulty peaceful",
            "gamerule doMobSpawning false",
            "gamerule doDaylightCycle false",
            "gamerule doWeatherCycle false",
            "time set day",
            "weather clear",
            f"fill {x-15} {y+1} {z-15} {x+15} {y+16} {z+15} minecraft:air replace",
            f"fill {x-15} {y} {z-15} {x+15} {y} {z+15} minecraft:bedrock replace",
            f"tp @s {self.cx:.3f} {self.player_y:.3f} {self.cz:.3f} 0 0",
            "gamemode survival @s",
            "clear @s",
        ]
        self.send(commands=commands, clear=True)
        self.advance(20)
        print("Flat curriculum arena created in disposable world.", flush=True)

    def reset(self, goal):
        commands = []
        if self.marker is not None:
            x, z = self.marker
            commands.append(f"setblock {x} {self.floor_y} {z} minecraft:bedrock replace")
        mx, mz = math.floor(goal[0]), math.floor(goal[1])
        commands.append(f"setblock {mx} {self.floor_y} {mz} minecraft:gold_block replace")
        self.marker = (mx, mz)
        commands += [
            f"tp @s {self.cx:.3f} {self.player_y:.3f} {self.cz:.3f} 0 0",
            "clear @s",
            "effect clear @s",
            "effect give @s minecraft:instant_health 1 5 true",
            "effect give @s minecraft:saturation 1 20 true",
        ]
        self.manager = self.env_util.InputStateManager()
        ob = self.send(commands=commands, clear=True)
        for _ in range(30):
            ob = self.send()
            # Wait for the teleport and reset to actually be observed.
            if (
                math.hypot(ob.player_pos[0] - self.cx, ob.player_pos[2] - self.cz) < 0.1
                and ob.health > 0
                and abs(ob.player_yaw) < 1
            ):
                break
        else:
            raise RuntimeError("Reset teleport was not reflected by MCio observations")
        # Two extra simulation steps settle the post-teleport position.
        return self.advance(2)

    def move(self, key_index, ticks):
        pressed = {key: int(i == key_index) for i, key in enumerate(KEYS)}
        inputs = self.manager.process_action(pressed, self.mcio_env.INPUT_MAP)
        ob = self.send(inputs=inputs)
        for _ in range(ticks - 1):
            ob = self.send()
        return ob


def choose_goal(arena, rng):
    # Near enough for an untrained policy to stumble into, but changing every trial.
    angle = rng.uniform(0, 2 * math.pi)
    radius = rng.uniform(3.2, 5.2)
    return (arena.cx + radius * math.cos(angle), arena.cz + radius * math.sin(angle))


def run_trial(arena, model, goal, rng, *, training, max_decisions, ticks):
    ob = arena.reset(goal)
    total_reward = 0.0
    initial_distance = math.hypot(goal[0] - ob.player_pos[0], goal[1] - ob.player_pos[2])
    outcome = "timeout"
    decisions = 0
    for decisions in range(1, max_decisions + 1):
        before = math.hypot(goal[0] - ob.player_pos[0], goal[1] - ob.player_pos[2])
        features = model.features(ob.player_pos[0], ob.player_pos[2], goal)
        action, probs = model.action(features, rng)
        ob = arena.move(action, ticks)
        after = math.hypot(goal[0] - ob.player_pos[0], goal[1] - ob.player_pos[2])
        # Improving distance is rewarded, wasted game time costs a little.
        reward = (before - after) - 0.025
        if after <= 1.1:
            reward += 4.0
            outcome = "success"
        elif ob.health <= 0:
            reward -= 4.0
            outcome = "death"
        elif math.hypot(ob.player_pos[0] - arena.cx, ob.player_pos[2] - arena.cz) > 12.0:
            reward -= 2.0
            outcome = "out_of_bounds"
        if training:
            model.learn(features, action, probs, reward)
        total_reward += reward
        if outcome != "timeout":
            break
    final_distance = math.hypot(goal[0] - ob.player_pos[0], goal[1] - ob.player_pos[2])
    return {
        "outcome": outcome,
        "decisions": decisions,
        "ticks": decisions * ticks,
        "reward": total_reward,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
    }


def evaluate(arena, policy, goals, *, max_decisions, ticks, seed):
    rng = random.Random(seed)
    results = [
        run_trial(arena, policy, goal, rng, training=False,
                  max_decisions=max_decisions, ticks=ticks)
        for goal in goals
    ]
    wins = [r for r in results if r["outcome"] == "success"]
    return {
        "wins": len(wins),
        "count": len(results),
        "avg_distance": sum(r["final_distance"] for r in results) / len(results),
        "avg_ticks": (sum(r["ticks"] for r in wins) / len(wins)) if wins else None,
    }


def fmt_eval(label, result):
    avg_ticks = "n/a" if result["avg_ticks"] is None else f'{result["avg_ticks"]:.0f}'
    print(f"{label}: {result['wins']}/{result['count']} reached target, "
          f"avg final distance={result['avg_distance']:.2f} blocks, "
          f"avg success time={avg_ticks} game ticks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=120, help="Learning episodes")
    parser.add_argument("--decisions", type=int, default=35, help="Max decisions per episode")
    parser.add_argument("--ticks", type=int, default=6, help="Game ticks per action")
    parser.add_argument("--eval", type=int, default=20, help="Held-out evaluation targets")
    parser.add_argument("--visible", action="store_true", help="Show the fast-forward game window")
    args = parser.parse_args()
    if min(args.episodes, args.decisions, args.ticks, args.eval) < 1:
        parser.error("All counts must be positive")

    from mcio_ctrl import controller, instance, network, types, world
    from mcio_ctrl.envs import env_util, mcio_env

    print(f"Restoring original {SOURCE} into ONLY disposable {TRIAL} ...", flush=True)
    world.WorldManager(mcio_dir=MCIO_DIR).copy("storage", SOURCE, INSTANCE, TRIAL, force=True)
    opts = types.RunOptions(
        mcio_dir=MCIO_DIR,
        instance_name=INSTANCE,
        world_name=TRIAL,
        mcio_mode=types.MCioMode.SYNC,
        hide_window=not args.visible,
        gpu_lib=None if args.visible else "nvidia",
        width=640 if args.visible else 320,
        height=360 if args.visible else 180,
    )
    model = LinearPolicy(seed=17)
    rng = random.Random(1729)
    launcher = instance.Launcher(opts)
    ctrl = None
    logpath = ROOT / "logs" / "navigation_training.csv"
    checkpoint = ROOT / "checkpoints" / "navigation_policy.npz"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    try:
        launcher.launch(wait=False)
        ctrl = controller.ControllerSync(
            action_port=opts.action_port,
            observation_port=opts.observation_port,
            connection_timeout=120,
        )
        ctrl.send_action(network.ActionPacket(clear_input=True))
        first = ctrl.recv_observation()
        if first is None:
            raise RuntimeError("Minecraft did not send first observation")
        arena = Arena(ctrl, first, mcio_env, env_util, network)
        arena.prepare()
        # Validation goals are separate from the randomly sampled training goals.
        evaluation_rng = random.Random(85241)
        eval_goals = [choose_goal(arena, evaluation_rng) for _ in range(args.eval)]
        print("Evaluating randomly initialized policy...", flush=True)
        before = evaluate(arena, model, eval_goals, max_decisions=args.decisions,
                          ticks=args.ticks, seed=991)
        fmt_eval("BEFORE", before)

        print(f"Training for {args.episodes} episodes with real game actions...", flush=True)
        started = time.perf_counter()
        with logpath.open("w", newline="", encoding="utf-8") as output:
            fields = ["episode", "outcome", "decisions", "ticks", "reward", "initial_distance", "final_distance"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for ep in range(1, args.episodes + 1):
                target = choose_goal(arena, rng)
                result = run_trial(arena, model, target, rng, training=True,
                                   max_decisions=args.decisions, ticks=args.ticks)
                writer.writerow({"episode": ep, **result})
                output.flush()
                # Keep a recoverable checkpoint even if a later test fails.
                if ep % 10 == 0 or ep == args.episodes:
                    np.savez_compressed(
                        checkpoint, weights=model.weights,
                        baseline=np.array(model.baseline),
                        updates=np.array(model.updates),
                    )
                if ep % 10 == 0 or ep == 1 or ep == args.episodes:
                    print(f"  episode={ep:3d}/{args.episodes} "
                          f"result={result['outcome']:>13s} "
                          f"final_dist={result['final_distance']:5.2f} "
                          f"reward={result['reward']:+6.2f}", flush=True)
        training_time = time.perf_counter() - started

        print("Evaluating frozen trained policy on the SAME held-out targets...", flush=True)
        after = evaluate(arena, model, eval_goals, max_decisions=args.decisions,
                         ticks=args.ticks, seed=991)
        fmt_eval("AFTER ", after)
        np.savez_compressed(checkpoint, weights=model.weights, baseline=np.array(model.baseline),
                            updates=np.array(model.updates))
        print(f"Learning time: {training_time:.1f} seconds", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Episode log: {logpath}", flush=True)
        print("NOTE: This is a toy navigation curriculum using structured goal vectors; ",
              "NOT Minecraft speedrunning or end-to-end visual RL.", flush=True)
    finally:
        if ctrl is not None:
            try:
                ctrl.send_stop()
            except Exception as exc:
                print(f"Minecraft stop warning: {exc}", flush=True)
            ctrl.close()
        launcher.close()


if __name__ == "__main__":
    main()
