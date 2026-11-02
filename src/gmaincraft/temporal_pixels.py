"""Pixel-only temporal image encoding; no privileged Minecraft observations.

The policy normally sees two RGB frames as 6 uint8 channels. Delta mode
replaces the second three channels with signed pixel differences offset by
128. Thus the CNN sees static RGB appearance plus explicit temporal motion,
without losing exact 6-channel shape or touching game/server state.
"""
import numpy as np


MODES=('raw','delta')


def temporal_encode(history,mode='raw'):
    if mode not in MODES:
        raise ValueError('Unknown temporal pixel encoding')
    arr=np.asarray(history)
    if (arr.dtype!=np.uint8 or arr.ndim not in (3,4) or
            arr.shape[-3:]!=(6,84,84)):
        raise ValueError('Expected uint8 (6,84,84) or (N,6,84,84) RGB frames')
    if mode=='raw':return arr
    past=arr[...,0:3,:,:].astype(np.int16)
    current=arr[...,3:6,:,:].astype(np.int16)
    diff=np.clip(current-past+128,0,255).astype(np.uint8)
    return np.concatenate((arr[...,0:3,:,:],diff),axis=-3)
