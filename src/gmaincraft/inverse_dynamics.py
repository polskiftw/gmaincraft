"""Action-effect perception from self-collected, agent-visible Minecraft RGB.

Inputs: two consecutive RGB frames, one at time t and another at t+1.
Targets: selected normal movement, camera and attack factors at time t.
Never supply target/tree coordinates, privileged pose, teacher labels, human
expert demonstrations or pretrained external model weights.

Predictive accuracy on NEW episode scenario/action seeds is the acceptance
metric; a high score does NOT establish Minecraft first-log skill.
"""
from pathlib import Path

import numpy as np

from .rollout_store import read_rollout_batch
from .vision_factorized import HEAD_SIZES,SCHEMA,SCHEMA_12,make_network
from .temporal_pixels import temporal_encode

INVERSE_SCHEMA='self_action_rgb_inverse_v1'


def load_rgb_action_pairs(folder, *, pixel_encoding="raw"):
    paths=sorted(Path(folder).glob('rollout_*.npz'))
    if not paths:
        raise ValueError(f'No recorded rollout batches at {folder}')
    images=[];targets=[];episodes=[]
    for path in paths:
        data=read_rollout_batch(path)
        meta=data['metadata']
        if meta.get('schema') not in (SCHEMA,SCHEMA_12):
            raise ValueError('Incompatible action schema in '+str(path))
        if data['rgb'].shape[0]<2:
            continue
        n=len(data['rgb'])
        # The last frame of each stored PixelHistory is the actual RGB frame
        # BEFORE the corresponding action. The next decision sees RGB AFTER.
        # Never join across terminal events or reset/world boundaries.
        usable=~data['done'][:-1]
        current=data['rgb'][:-1,3:6]
        following=data['rgb'][1:,3:6]
        pair=np.concatenate((current,following),axis=1)
        images.append(pair[usable])
        targets.append(data['action'][:-1][usable].astype(np.int64))
        episodes.extend([path.stem]*int(np.count_nonzero(usable)))
    if not images or sum(len(x) for x in images)<2:
        raise ValueError('Insufficient consecutive actions after skipping episode ends')
    x=np.concatenate(images)
    y=np.concatenate(targets)
    if x.ndim!=4 or x.shape[1:]!=(6,84,84) or x.dtype!=np.uint8 or y.shape!=(len(x),3):
        raise RuntimeError('Invalid RGB/action data tensor')
    for k,count in enumerate(HEAD_SIZES):
        if np.any(y[:,k]<0) or np.any(y[:,k]>=count):
            raise RuntimeError('Invalid action class in recorded data')
    return temporal_encode(x,pixel_encoding),y,episodes


def make_action_predictor():
    # Same initially random CNN trunk/heads as the factorized RL policy;
    # critic output is unused. Weight compatibility is intentional.
    return make_network()


def compute_action_metrics(logits,targets):
    import torch
    with torch.no_grad():
        return [float((head.argmax(dim=-1)==targets[:,i]).float().mean().item())
                for i,head in enumerate(logits)]


def load_encoder_checkpoint(path):
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=INVERSE_SCHEMA or not isinstance(checkpoint.get('visual'),dict):
        raise ValueError('Not a compatible self-observed inverse-dynamics checkpoint')
    return checkpoint['visual'],checkpoint.get('metadata',{})
