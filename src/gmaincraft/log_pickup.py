"""RGB-only pickup features and tiny scratch-trained movement policy.

The narrow brown-pixel crop is hand-engineered for a *controlled* bedrock
collection arena. No coordinates, yaw/pitch, raycasts, or Minecraft world state
are passed to the policy. Controller-only geometric scalar rewards are a
training shortcut, not part of the policy's observation.
"""

import numpy as np

PICKUP_ACTIONS = ('W', 'A', 'S', 'D')
PICKUP_FEATURES = 14


def pickup_features(frame: np.ndarray) -> np.ndarray:
    """Track the brown dropped-item sprite within the visible scene (not HUD).

    Seven horizontal, three vertical mask-density bins; 2 centroid components,
    a detection-strength component and a constant bias. A deliberately narrow
    scene crop suppresses much of the hand/HUD, not all backgrounds.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 96:
        raise ValueError('pickup_features expects an HxWx3 RGB image >= 96 pixels')
    h,w=frame.shape[:2]
    roi=frame[int(.30*h):int(.74*h):2,int(.10*w):int(.89*w):2].astype(np.int16)
    r,g,b=roi[:,:,0],roi[:,:,1],roi[:,:,2]
    brown=((r-g>17)&(g-b>12)&(r>70)&(r<205)&(g>42)&(g<140)&(b>10)&(b<102))
    rows=[chunk.mean() for chunk in np.array_split(brown,3,axis=0)]
    cols=[chunk.mean() for chunk in np.array_split(brown,7,axis=1)]
    ys,xs=np.nonzero(brown)
    if len(xs):
        cx=2*(xs.mean()/max(brown.shape[1]-1,1)-.5)
        cy=2*(ys.mean()/max(brown.shape[0]-1,1)-.5)
    else:
        cx=cy=0.0
    return np.array([*(v*35 for v in cols),*(v*35 for v in rows),cx,cy,
                     min(1.0,float(brown.mean()*120)),1.0],dtype=np.float64)


class PickupPolicy:
    """Tiny stochastic softmax REINFORCE agent with four normal movement keys."""
    def __init__(self,seed=4401,learning_rate=.075):
        self.weights=np.random.default_rng(seed).normal(0,.025,size=(4,PICKUP_FEATURES))
        self.baseline=0.0
        self.learning_rate=learning_rate
        self.updates=0

    def probabilities(self,features):
        if features.shape!=(PICKUP_FEATURES,):
            raise ValueError(f'Expected {PICKUP_FEATURES} RGB-derived features')
        logits=self.weights@features
        logits-=np.max(logits)
        prob=np.exp(logits)
        return prob/prob.sum()

    def choose(self,features,rng,*,greedy=False):
        prob=self.probabilities(features)
        return (int(np.argmax(prob)) if greedy else
                int(rng.choices(range(4),weights=prob,k=1)[0])),prob

    def learn(self,features,index,prob,reward):
        advantage=float(np.clip(reward-self.baseline,-5,5))
        score=-prob.copy();score[index]+=1
        self.weights+=self.learning_rate*advantage*np.outer(score,features)
        self.baseline=.96*self.baseline+.04*float(reward)
        self.updates+=1


def fixture_item_command(x: float, y: float, z: float) -> str:
    """Controller-side, zero-initial-drift item fixture, NEVER a policy action.

    All input numbers are generated from safe in-arena positions, not strings.
    The log remains a normal collectable Minecraft item affected by gravity.
    """
    from math import isfinite
    if not all(isfinite(v) for v in (x,y,z)):
        raise ValueError('Nonfinite item fixture coordinates')
    return (f'summon minecraft:item {x:.5f} {y:.4f} {z:.5f} '
            '{Item:{id:"minecraft:oak_log",count:1},PickupDelay:0s,Age:0s,'
            'Motion:[0.0d,0.0d,0.0d]}')
