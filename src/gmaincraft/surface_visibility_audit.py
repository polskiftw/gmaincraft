"""Independent *evaluation-only* evidence of changed Minecraft RGB pixels.

Never feed these paired (teacher-constructed) images or the changed-pixel mask
to a playing policy. Differences measure rendered impact, NOT a proof that
people could identify a block's type, and can include lighting/shadow effects.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameChange:
    changed_pixels: int
    max_channel_delta: int
    bbox_xyxy: tuple[int, int, int, int] | None
    changed_fraction: float


def compare_frames(before, after, *, min_channel_delta=40, hud_rows=28, crosshair_radius=5):
    """Count materially changed RGB pixels outside HUD and crosshair.

    A deliberately simple independent validation, *not* an eligibility gate.
    The thresholds are fixed measurement controls for paired experiments.
    """
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('Expected matching HxWx3 uint8 Minecraft screenshots')
    if not 1 <= min_channel_delta <= 255 or hud_rows < 0 or crosshair_radius < 0:
        raise ValueError('Invalid independent image-audit parameters')
    h, w = a.shape[:2]
    delta = np.max(np.abs(a.astype(np.int16)-b.astype(np.int16)), axis=2)
    eligible = np.ones((h,w), dtype=bool)
    if hud_rows:
        eligible[max(h-hud_rows,0):] = False
    if crosshair_radius:
        cy, cx = h//2, w//2
        eligible[max(0,cy-crosshair_radius):min(h,cy+crosshair_radius+1),
                 max(0,cx-crosshair_radius):min(w,cx+crosshair_radius+1)] = False
    changed = eligible & (delta >= min_channel_delta)
    ys, xs = np.nonzero(changed)
    count = len(xs)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)) if count else None
    return FrameChange(count,int(delta[eligible].max()) if eligible.any() else 0,
                       bbox,count/max(1,int(eligible.sum())))
