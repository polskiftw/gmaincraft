"""Minecraft Discovery Lab: fast-forward throughput benchmark, no AI training.

Works with mcio_ctrl 1.5.2 and the isolated MCio demo installation.
Launches Minecraft in synchronous mode, samples random inputs, restores a clean
world before each episode, and measures actual environment steps per second.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from mcio_ctrl import types, world
from mcio_ctrl.envs.mcio_env import MCioEnv

ROOT = Path(__file__).resolve().parent.parent
MCIO_DIR = ROOT / "mcio-data"
INSTANCE = "DemoInstance"
TEMPLATE = "DemoWorld"
TRIAL = "BenchmarkWorld"


def make_action(env, rng):
    action = env.get_noop_action()
    direction = rng.choice(("W", "A", "S", "D", None))
    if direction:
        action[direction] = 1
    action["SPACE"] = int(rng.random() < 0.25)
    action["LEFT_BUTTON"] = int(rng.random() < 0.4)
    action["RIGHT_BUTTON"] = int(rng.random() < 0.1)
    action["cursor_delta"] = np.array(
        [rng.randint(-40, 40), rng.randint(-20, 20)], dtype=np.int32
    )
    return action


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCio fast-forward steps")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true", help="Hide Minecraft's window")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.width, args.height) < 1:
        parser.error("Episodes, steps, width and height must be positive")

    worlds = world.WorldManager(mcio_dir=MCIO_DIR)
    opts = types.RunOptions(
        mcio_dir=MCIO_DIR,
        instance_name=INSTANCE,
        world_name=TRIAL,
        mcio_mode=types.MCioMode.SYNC,
        hide_window=args.headless,
        gpu_lib="nvidia" if args.headless else None,
        width=args.width,
        height=args.height,
    )
    env = None  # Create a fresh controller per episode so held keys do not leak.
    rng = random.Random(42)
    results = []
    try:
        for episode in range(1, args.episodes + 1):
            if env is not None:
                env.close()
            # Only overwrite BenchmarkWorld, never the original DemoWorld.
            worlds.copy("storage", TEMPLATE, INSTANCE, TRIAL, force=True)
            env = MCioEnv(opts)  # No mirror GUI; Minecraft still visible unless --headless.
            env.reset()
            print(f"\nEpisode {episode}/{args.episodes}: benchmarking...", flush=True)
            start = time.perf_counter()
            last_report = start
            action = env.get_noop_action()
            steps_done = 0
            reason = "step_limit"
            for step in range(1, args.steps + 1):
                if step % 25 == 1:
                    action = make_action(env, rng)
                else:
                    action["cursor_delta"] = np.zeros(2, dtype=np.int32)
                obs, _, terminated, truncated, _ = env.step(action)
                steps_done = step
                now = time.perf_counter()
                if now - last_report >= 3.0:
                    print(
                        f"  step={step:5d}   speed={step/(now-start):6.1f} steps/s"
                        f"   health={env.health:g}   pos={obs['pos'].round(1)}",
                        flush=True,
                    )
                    last_report = now
                if terminated or truncated:
                    reason = "death" if terminated else "truncated"
                    break
            elapsed = time.perf_counter() - start
            rate = steps_done / elapsed if elapsed else 0.0
            results.append(rate)
            print(
                f"  FINISHED {reason}: {steps_done} steps in {elapsed:.2f}s = "
                f"{rate:.1f} steps/s ({rate/20:.2f}x 20 TPS)",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()
        print("Minecraft closed.", flush=True)
    if results:
        print(f"\nAverage: {sum(results)/len(results):.1f} steps/s (active steps only)")
        print("Note: this is environment throughput, not neural-network training speed.")


if __name__ == "__main__":
    main()
