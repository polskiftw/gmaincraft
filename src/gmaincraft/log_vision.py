"""Cheap, explicitly engineered pixel features for a controlled oak-log arena.

Input is RGB rendered pixels ONLY (not player pose, world coordinates, or a
Minecraft block/raycast query). The color heuristic is a temporary vision
encoder, NOT a learned detector and NOT reliable in general biomes/lighting.
"""
import numpy as np


FEATURE_COUNT = 12


def visual_features(frame: np.ndarray) -> np.ndarray:
    """Summarize likely brown visible pixels in 5 horizontal and 3 vertical bins.

    The final two centroid features retain the direction of visible brown mass,
    but do not reveal hidden-world location. The last feature is bias=1.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB frame')
    height, width = frame.shape[:2]
    if min(height, width) < 72:
        raise ValueError('Frame too small for color-binning prototype')
    # Avoid hotbar/HUD at bottom, and border/window artifacts. Subsample for speed.
    sample = frame[::6, ::6, :].astype(np.int16)
    sample = sample[:max(1, sample.shape[0]-9), :]
    red, green, blue = sample[:,:,0], sample[:,:,1], sample[:,:,2]
    # Oak bark in the test arena has visibly warm red > green > blue pixels.
    # Gray bedrock and blue sky do not satisfy both red/green gaps.
    mask = ((red - green > 15) & (green - blue > 12) &
            (red > 70) & (green > 40) & (blue > 10) & (red < 230))
    ybins = np.array_split(mask, 3, axis=0)
    xbins = np.array_split(mask, 5, axis=1)
    horizontal = np.array([chunk.mean() for chunk in xbins], dtype=np.float64)
    vertical = np.array([chunk.mean() for chunk in ybins], dtype=np.float64)
    ys, xs = np.nonzero(mask)
    if len(xs):
        cx = 2.0 * (xs.mean() / max(1,mask.shape[1]-1) - 0.5)
        cy = 2.0 * (ys.mean() / max(1,mask.shape[0]-1) - 0.5)
    else:
        cx = cy = 0.0
    # 5+3+2+1+1 = 12 features; bounded scale for small policy gradients.
    return np.concatenate((horizontal*2,vertical*2,[cx,cy,float(mask.mean()),1.0]))


class VisualAimPolicy:
    """Tiny from-scratch REINFORCE controller over image-only features.

    Actions are camera left/right/up/down or ATTACK. This is deliberately not
    an algorithmic 'turn toward target' planner or pretrained visual model.
    """
    ACTIONS = ('look_left', 'look_right', 'look_up', 'look_down', 'attack')

    def __init__(self, seed=5501, learning_rate=0.065):
        self.weights = np.random.default_rng(seed).normal(0.0, 0.015,
            size=(len(self.ACTIONS), FEATURE_COUNT))
        self.baseline = 0.0
        self.learning_rate = learning_rate
        self.updates = 0

    def probabilities(self, features):
        if features.shape != (FEATURE_COUNT,):
            raise ValueError('Expected visual RGB feature vector')
        logits = self.weights @ features
        logits -= np.max(logits)
        prob = np.exp(logits)
        return prob / prob.sum()

    def choose(self, features, rng, greedy=False):
        prob = self.probabilities(features)
        if greedy:
            return int(np.argmax(prob)), prob
        return int(rng.choices(range(len(prob)), weights=prob, k=1)[0]), prob

    def learn(self, features, action, prob, reward):
        adv = float(np.clip(reward - self.baseline, -7.0, 7.0))
        grad = -prob.copy()
        grad[action] += 1.0
        self.weights += self.learning_rate * adv * np.outer(grad, features)
        self.baseline = 0.97 * self.baseline + 0.03 * float(reward)
        self.updates += 1
