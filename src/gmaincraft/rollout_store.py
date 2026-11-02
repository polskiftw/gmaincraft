"""Local, versioned, checkpoint-independent Minecraft policy trajectory data.

Store ONLY the agent's pixel history, selected normal action components, and
scalar rewards/values/log probabilities/episode endings plus action masks.
World seed, target coordinates, player pose, controller commands, NBT data,
admin states and private keys MUST NOT be serialized to this dataset.

These are on-policy PPO rollouts: they are reusable for diagnostics and future
representation/offline-learning research, NOT for unlimited stale PPO updates.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROLLOUT_FORMAT=1
ALLOWED_FIELDS=frozenset(('format_version','metadata_json','rgb','action',
                          'reward','value','old_logp','game_ticks','done','active_heads'))


def arrays_from_trajectory(steps, metadata):
    if not steps:
        raise ValueError('Cannot record an empty rollout batch')
    if not isinstance(metadata,dict):
        raise TypeError('Metadata must be a dict')
    allowed_meta={'schema','run_tag','end_episode','episodes','stages','device',
                  'image_shape','action_interval','training_split'}
    if not set(metadata)<=allowed_meta:
        raise ValueError('Metadata must not contain hidden world information')
    rgb=np.stack([t.image for t in steps])
    if rgb.ndim!=4 or rgb.shape[1:]!=(6,84,84) or rgb.dtype!=np.uint8:
        raise ValueError('Rollout RGB must contain two 84x84 full-color frames')
    action=np.asarray([t.action for t in steps],dtype=np.int16)
    if action.shape!=(len(steps),3) or np.any(action[:,0]<0) or np.any(action[:,0]>=6) or np.any(action[:,1]<0) or np.any(action[:,1]>=5) or np.any(action[:,2]<0) or np.any(action[:,2]>=2):
        raise ValueError('Invalid factorized actions')
    masks=np.asarray([t.active_heads for t in steps],dtype=np.bool_)
    if masks.shape!=(len(steps),3) or not masks.any(axis=1).all():
        raise ValueError('Invalid head masks')
    numbers=[np.asarray([getattr(t,k) for t in steps],dtype=np.float32)
             for k in ('reward','value','logp')]
    if not all(np.isfinite(x).all() for x in numbers):
        raise ValueError('Nonfinite trajectory values')
    durations=np.asarray([t.duration for t in steps],dtype=np.int16)
    if np.any(durations<1) or np.any(durations>200):
        raise ValueError('Invalid game tick durations')
    done=np.asarray([t.done for t in steps],dtype=np.bool_)
    return dict(format_version=np.asarray(ROLLOUT_FORMAT,dtype=np.int16),
                metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)),
                rgb=rgb,action=action,reward=numbers[0],value=numbers[1],
                old_logp=numbers[2],game_ticks=durations,done=done,
                active_heads=masks)


def save_rollout_batch(directory, episode, steps, metadata):
    if not isinstance(episode,int) or episode<1:
        raise ValueError('Episode index must be positive')
    folder=Path(directory)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'rollout_{episode:05d}.npz'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite rollout: {path}')
    data=arrays_from_trajectory(steps,metadata)
    fd,temp=tempfile.mkstemp(prefix='rollout.',suffix='.tmp',dir=folder)
    try:
        with os.fdopen(fd,'wb') as file:
            np.savez_compressed(file,**data)
            file.flush();os.fsync(file.fileno())
        if path.exists():
            raise FileExistsError(f'Refusing overwrite: {path}')
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return path


def read_rollout_batch(path):
    with np.load(path,allow_pickle=False) as file:
        if set(file.files)!=ALLOWED_FIELDS:
            raise ValueError('Rollout field mismatch')
        if int(file['format_version'])!=ROLLOUT_FORMAT:
            raise ValueError('Unsupported rollout schema')
        data={k:file[k].copy() for k in file.files}
    data['metadata']=json.loads(str(data.pop('metadata_json')))
    return data
