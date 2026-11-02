"""Pure, deterministic variable-tree training scenario definitions.

Sampling privileged world geometry here does NOT expose it to the learning
policy: these dataclasses are used only by the environment controller.
"""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class LogCase:
    player_dx: float
    player_dz: float
    tree_dx: int
    tree_dz: int
    yaw: float
    pitch: float


def sample_cases(seed, n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('At least one case is required')
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        px = rng.uniform(-1.8, 1.8)
        pz = rng.uniform(-1.3, 0.5)
        dx = rng.randint(-4, 4)
        dz = rng.randint(2, 6)
        if 2.3 <= math.hypot(dx - px, dz - pz) <= 7.3:
            cases.append(LogCase(px, pz, dx, dz,
                                 rng.uniform(-85, 85), rng.uniform(-15, 15)))
    return cases
