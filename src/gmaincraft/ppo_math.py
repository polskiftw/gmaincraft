"""Semi-Markov generalized advantage estimation for timed Minecraft actions.

Actions may last 2 (look), 8 (walk) or 80+ (mining) GAME ticks, so applying
one fixed discount per high-level decision gives physically inconsistent
returns. Discount and GAE trace are adjusted for each actual action duration.
"""
import numpy as np


def timed_gae(rewards, values, durations, terminals, *, last_value=0.0,
              gamma_tick=0.9995, lam_per_five_ticks=0.95):
    """Return (advantages, value targets), each 1-D float32.

    `terminals[i]`: real episode end. Never propagate reward/value across
    terminal worlds. The last transition may bootstrap `last_value` when not
    terminal. A truncated episode is treated as terminal by the caller unless
    a known correct bootstrap value is supplied before resetting that world.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    dt = np.asarray(durations)
    done = np.asarray(terminals, dtype=np.bool_)
    if not (r.ndim == v.ndim == dt.ndim == done.ndim == 1 and
            len(r) == len(v) == len(dt) == len(done) and len(r) > 0):
        raise ValueError('GAE expects equal-length, nonempty 1-D trajectories')
    if (not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)) or
            not np.isfinite(last_value)):
        raise ValueError('GAE requires finite rewards and values')
    if np.any(dt < 1) or not np.all(dt == np.floor(dt)):
        raise ValueError('GAE durations must be positive integer game ticks')
    if not (0 < gamma_tick <= 1 and 0 <= lam_per_five_ticks <= 1):
        raise ValueError('Invalid GAE gamma/lambda')
    advantage = np.zeros(len(r), dtype=np.float64)
    trace = 0.0
    for i in range(len(r) - 1, -1, -1):
        next_value = last_value if i == len(r) - 1 else v[i+1]
        continuation = 0.0 if done[i] else 1.0
        discount = gamma_tick ** int(dt[i])
        trace_discount = lam_per_five_ticks ** (float(dt[i]) / 5.0)
        delta = r[i] + continuation * discount * next_value - v[i]
        trace = delta + continuation * discount * trace_discount * trace
        advantage[i] = trace
    return advantage.astype(np.float32), (advantage + v).astype(np.float32)
