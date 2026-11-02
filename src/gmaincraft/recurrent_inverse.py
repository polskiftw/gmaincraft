"""Causal, self-supervised RGB/action memory diagnostic.

At decision t, use RGB before+after action_t (diagnostic only) and a GRU
whose previous state contains earlier frames/actions. The previous-action
feature is explicitly action_(t-1), NEVER the current target action_t.
Training labels are the controller's own random normal inputs, not privileged
Minecraft hidden state, demonstrations, or teacher target geometry.

This is an OFFLINE action-effect memory study, not a recurrent deployed
Minecraft agent and not an Any% skill.
"""
from pathlib import Path

import numpy as np

from .rollout_store import read_rollout_batch
from .vision_factorized import HEAD_SIZES,SCHEMA,SCHEMA_12,make_network
from .temporal_pixels import temporal_encode


def previous_action_onehots(labels):
    """Build strictly causal one-hot own action history for a (T,3) sequence."""
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.shape[1]!=3 or len(labels)<1:
        raise ValueError('Expected nonempty (T,3) action indices')
    parts=[]
    for i,classes in enumerate(HEAD_SIZES):
        if np.any(labels[:,i]<0) or np.any(labels[:,i]>=classes):
            raise ValueError('Invalid self-action label')
        block=np.zeros((len(labels),classes),dtype=np.float32)
        if len(labels)>1:
            block[np.arange(1,len(labels)),labels[:-1,i]]=1.
        parts.append(block)
    return np.concatenate(parts,axis=-1)


def episode_action_sequences(folder,*,encoding='raw'):
    paths=sorted(Path(folder).glob('rollout_*.npz'))
    if not paths:raise ValueError('No recorded trajectory data')
    sequences=[]
    for path in paths:
        data=read_rollout_batch(path)
        if data['metadata'].get('schema') not in (SCHEMA,SCHEMA_12):
            raise ValueError('Unsupported recorded action format')
        usable=~data['done'][:-1]
        # A saved file should not cross multiple episodes without careful
        # splitting. Collectors use one episode per rollout file; reject
        # old mixed PPO replay batches rather than splice reset artifacts.
        if np.count_nonzero(data['done']) != 1 or not bool(data['done'][-1]):
            raise ValueError('Expected one terminal episode per rollout file')
        current=data['rgb'][:-1,3:6]
        following=data['rgb'][1:,3:6]
        x=np.concatenate((current,following),axis=1)[usable]
        y=data['action'][:-1][usable].astype(np.int64)
        if len(y)<2:continue
        sequences.append((temporal_encode(x,encoding),previous_action_onehots(y),y))
    if not sequences:raise ValueError('No usable episode sequences')
    return sequences


def make_recurrent_inverse(hidden_size=128):
    import torch
    from torch import nn

    class CausalMemory(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual=make_network().visual
            self.memory=nn.GRU(256+sum(HEAD_SIZES),hidden_size,batch_first=True)
            self.heads=nn.ModuleList([nn.Linear(hidden_size,n) for n in HEAD_SIZES])

        def forward(self,pixel_pairs,previous_action,hidden=None):
            if pixel_pairs.ndim!=5 or pixel_pairs.shape[2:]!=(6,84,84):
                raise ValueError('Expected (batch,seq,6,84,84) uint8 pixels')
            if previous_action.shape[:2]!=pixel_pairs.shape[:2] or previous_action.shape[-1]!=sum(HEAD_SIZES):
                raise ValueError('Mismatched causal action history')
            b,t=pixel_pairs.shape[:2]
            visual=self.visual(pixel_pairs.reshape(b*t,6,84,84).float()/255.).reshape(b,t,-1)
            x=torch.cat((visual,previous_action.float()),dim=-1)
            output,hidden=self.memory(x,hidden)
            return tuple(layer(output) for layer in self.heads),hidden

    return CausalMemory()
