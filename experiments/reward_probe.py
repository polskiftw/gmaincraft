"""Minecraft Discovery Lab - reward plumbing test (no machine learning).

Uses your existing MCio 1.21.3 DemoInstance and DemoWorld template.
Creates/overwrites ONLY the disposable world MDL_RewardProbe_v1.
The controller uses /give solely to verify observation and scoring; the
future learning agent will never have access to these commands.
"""

import argparse
from pathlib import Path

from mcio_ctrl import controller, instance, network, types, world

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "mcio-data"
INSTANCE = "DemoInstance"
SOURCE = "DemoWorld"
TRIAL = "MDL_RewardProbe_v1"


class RewardTracker:
    """Award +1 when an item type is seen for the first time THIS episode."""

    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        self.seen = set()
        self.total = 0

    def observe(self, packet):
        slots = (
            packet.inventory_main + packet.inventory_armor + packet.inventory_offhand
        )
        current = {slot.id for slot in slots if slot.id and slot.count > 0}
        new_types = sorted(current - self.seen)
        self.seen.update(current)
        points = len(new_types)
        self.total += points
        return current, new_types, points


def exchange(ctrl, commands=None):
    ctrl.send_action(network.ActionPacket(commands=commands or []))
    packet = ctrl.recv_observation()
    if packet is None or packet.version != network.MCIO_PROTOCOL_VERSION:
        raise RuntimeError("Missing or incompatible MCio observation")
    return packet


def execute_and_check(ctrl, tracker, command, desired, *, max_ticks=80):
    """Send one command, advance the environment, wait for inventory state."""
    rewards = 0
    changes = set()
    for tick in range(max_ticks):
        pkt = exchange(ctrl, [command] if tick == 0 else [])
        current, new, points = tracker.observe(pkt)
        rewards += points
        changes.update(new)
        if desired(current):
            return pkt, tuple(sorted(changes)), rewards, tick + 1
    raise RuntimeError(f"Observation did not reflect command after {max_ticks} ticks: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible", action="store_true", help="Show Minecraft instead of headless")
    args = parser.parse_args()

    print(f"Copying {SOURCE} into disposable {TRIAL} ...", flush=True)
    world.WorldManager(mcio_dir=DATA).copy(
        "storage", SOURCE, INSTANCE, TRIAL, force=True
    )

    opts = types.RunOptions(
        mcio_dir=DATA,
        instance_name=INSTANCE,
        world_name=TRIAL,
        mcio_mode=types.MCioMode.SYNC,
        hide_window=not args.visible,
        gpu_lib=None if args.visible else "nvidia",
        width=320,
        height=180,
    )
    launcher = instance.Launcher(opts)
    ctrl = None
    tracker = RewardTracker()
    try:
        launcher.launch(wait=False)
        ctrl = controller.ControllerSync(
            action_port=opts.action_port,
            observation_port=opts.observation_port,
            connection_timeout=120,
        )
        initial = exchange(ctrl)
        tracker.observe(initial)
        print(f"Connected. Initial health={initial.health}, position={initial.player_pos}", flush=True)
        print("Test 1: controller gives first oak log", flush=True)
        _, new, reward, ticks = execute_and_check(
            ctrl, tracker, "give @s minecraft:oak_log 1",
            lambda inv: "minecraft:oak_log" in inv,
        )
        print(f"  new items={new}; reward=+{reward}; steps={ticks}", flush=True)
        assert new == ("minecraft:oak_log",) and reward == 1, "First acquisition did not score exactly +1"

        print("Test 2: a second oak log should NOT score again", flush=True)
        _, new, reward, ticks = execute_and_check(
            ctrl, tracker, "give @s minecraft:oak_log 1",
            lambda inv: "minecraft:oak_log" in inv,
        )
        # The command may still be queued when the first observation is returned.
        # Advance enough ticks to let it execute, checking for new types each tick.
        last = None
        for _ in range(20):
            last = exchange(ctrl)
            _, more, points = tracker.observe(last)
            new = tuple(sorted(set(new) | set(more)))
            reward += points
        oak_count = sum(s.count for s in last.inventory_main if s.id == "minecraft:oak_log")
        print(f"  oak logs={oak_count}; new items={new}; reward=+{reward}", flush=True)
        assert oak_count >= 2, "The second /give command did not take effect"
        assert reward == 0, "Duplicate item type was rewarded"

        print("Test 3: obtaining a different item type should score", flush=True)
        _, new, reward, ticks = execute_and_check(
            ctrl, tracker, "give @s minecraft:stone_pickaxe 1",
            lambda inv: "minecraft:stone_pickaxe" in inv,
        )
        print(f"  new items={new}; reward=+{reward}; steps={ticks}", flush=True)
        assert "minecraft:stone_pickaxe" in new and reward == 1
        assert tracker.total == 2

        print("Test 4: clear inventory and begin a NEW episode (without JVM restart)", flush=True)
        _, _, _, ticks = execute_and_check(
            ctrl, tracker, "clear @s", lambda inv: len(inv) == 0
        )
        for _ in range(10):
            tracker.observe(exchange(ctrl))
        tracker.reset_episode()  # Scoring reset, not a full Minecraft world reset!
        print(f"  inventory cleared; score reset to {tracker.total}", flush=True)

        print("Test 5: oak log should earn +1 again in the new episode", flush=True)
        _, new, reward, ticks = execute_and_check(
            ctrl, tracker, "give @s minecraft:oak_log 1",
            lambda inv: "minecraft:oak_log" in inv,
        )
        print(f"  new items={new}; reward=+{reward}; steps={ticks}", flush=True)
        assert new == ("minecraft:oak_log",) and reward == 1
        print("\nPASS: observation -> event detection -> numeric reward -> per-episode reset", flush=True)
        print("NOTE: /give is controller-only test equipment; no model training occurred.", flush=True)
    finally:
        if ctrl is not None:
            try:
                ctrl.send_stop()
            except Exception as exc:
                print(f"MCio stop warning: {exc}", flush=True)
            ctrl.close()
        launcher.close()


if __name__ == "__main__":
    main()
