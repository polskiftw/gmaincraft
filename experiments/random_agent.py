"""Random-action MCio baseline: visible, repeatable episodes, no ML yet."""
from pathlib import Path
import random
import time

import numpy as np
from mcio_ctrl import types, world
from mcio_ctrl.envs.mcio_env import MCioEnv

ROOT = Path(__file__).resolve().parent.parent
MCIO_DIR = ROOT / "mcio-data"
INSTANCE = "DemoInstance"
TEMPLATE = "DemoWorld"
TRIAL = "AgentTestWorld"
EPISODES = 2
MAX_STEPS = 400

worlds = world.WorldManager(mcio_dir=MCIO_DIR)
options = types.RunOptions(
    mcio_dir=MCIO_DIR,
    instance_name=INSTANCE,
    world_name=TRIAL,
    mcio_mode=types.MCioMode.SYNC,
    hide_window=False,
    width=854,
    height=480,
)
env = MCioEnv(options, render_mode="human")
rng = random.Random(42)

try:
    for episode in range(1, EPISODES + 1):
        env.close()  # Ensure Minecraft is stopped before replacing the world.
        worlds.copy("storage", TEMPLATE, INSTANCE, TRIAL, force=True)
        obs, _ = env.reset()
        print(f"\nEpisode {episode}/{EPISODES}: random controls enabled", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if step % 25 == 1:
                action = env.get_noop_action()
                direction = rng.choice(["W", "A", "S", "D", None])
                if direction:
                    action[direction] = 1
                action["SPACE"] = int(rng.random() < 0.25)
                action["LEFT_BUTTON"] = int(rng.random() < 0.4)
                action["RIGHT_BUTTON"] = int(rng.random() < 0.1)
                turn = np.array([rng.randint(-40, 40), rng.randint(-20, 20)], dtype=np.int32)
            else:
                turn = np.zeros(2, dtype=np.int32)

            action["cursor_delta"] = turn
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()

            if step % 50 == 0:
                print(f"  step={step:3d}  pos={obs['pos'].round(1)}  health={env.health:g}", flush=True)

            if terminated or truncated:
                print(f"  Episode ended at step {step} (death or environment termination)", flush=True)
                break
            time.sleep(0.05)  # Watchable speed; remove for later fast-forward benchmarking.

        print(f"  Completed episode {episode}", flush=True)
finally:
    env.close()
    print("Stopped Minecraft and closed MCio.", flush=True)
