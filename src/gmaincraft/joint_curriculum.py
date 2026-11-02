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


def sample_near_cases(seed, n):
    """Short transfer-first curriculum: nearby trunks, variable aiming/position.

    Initial center is roughly within camera field of view and vanilla reach.
    There is still no target label or target position passed to the policy.
    Train/eval seeds remain distinct. Compare this EASIER task with the
    broad 2.3-7.3-block yaw-85 scenario distribution separately.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('At least one case required')
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        px = rng.uniform(-.85, .85)
        pz = rng.uniform(-.55, .55)
        dx = rng.randint(-1, 1)
        dz = rng.randint(2, 3)
        horizontal = math.hypot(dx-px, dz-pz)
        if not 1.8 <= horizontal <= 3.7:
            continue
        target_yaw = math.degrees(math.atan2(-(dx-px),dz-pz))
        target_pitch = math.degrees(math.atan2(1.12,horizontal))
        cases.append(LogCase(px,pz,dx,dz,
                             target_yaw+rng.uniform(-24,24),
                             target_pitch+rng.uniform(-9,9)))
    return cases


def sample_stationary_cases(seed,n):
    """A real centered log with small camera jitter, usable attack-only stage.

    Teacher positions the player at a grounded repeatable 2-block range and
    aims roughly at the LOWER actual wood block. No position/target info enters
    the CNN; only the attack decision is permitted during the warm-up stage.
    """
    if not isinstance(n,int) or n<1:
        raise ValueError('At least one stationary case required')
    rng=random.Random(seed)
    return [LogCase(0.,0.,0,2,rng.uniform(-4.5,4.5),
                    29.25+rng.uniform(-4.5,4.5)) for _ in range(n)]


def interleave_phases(plan):
    """Deterministically balance skill phases throughout PPO update batches.

    Replays actual learned experience for every phase across training, rather
    than 'forgetting' the stationary motor skill after an entire later block of
    aiming/movement tasks. Preserve relative case order within each phase.
    """
    if not plan:
        return []
    from collections import OrderedDict
    grouped=OrderedDict()
    for row in plan:
        grouped.setdefault(row[0],[]).append(row)
    counts={k:len(v) for k,v in grouped.items()}
    taken={k:0 for k in grouped}
    output=[]
    total=len(plan)
    for index in range(total):
        available=[k for k in grouped if taken[k]<counts[k]]
        selected=max(available,key=lambda k:(counts[k]*(index+1)/total-taken[k],-list(grouped).index(k)))
        output.append(grouped[selected][taken[selected]])
        taken[selected]+=1
    return output
