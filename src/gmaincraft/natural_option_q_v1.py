"""Experimental, genuinely TRAINABLE two-option selector from REAL survival replay.

The network's inputs are only the player-known pickup features and selected
own-camera / own-mining-history fields from exact V9; no world seed, fixture,
statistical `minecraft:mined` scorer counters, terrain map or admin commands.
The choices are: frozen wood-mining Q motor vs frozen visible-item pickup Q
motor. Actual Minecraft outcomes, not hand-authored expert switch labels,
are the training signal. This pure module does not operate the game.
"""
import math
import hashlib
import numpy as np
import torch
from .numeric_item_pickup import PickupReading
from .numeric_item_pickup_v2 import FEATURE_NAMES as PICKUP_FEATURES
from .natural_online_policy import (NaturalObservation,TEMPORAL_V9_SELF_FEATURES)

SCHEMA='natural_wood_pickup_option_double_q_v1'
OPTIONS=('keep_mining_Q','select_pickup_Q')
EXTRA_FEATURES=(
    'health_div_20','hunger_div_20','crosshair_is_wood',
    'mining_active','mining_crack_div_10','crack_stage_rose_same_target_div_10',
    'own_log_count_div_16','visible_wood_fraction',
)
FEATURES=tuple('pickup_'+s for s in PICKUP_FEATURES)+tuple('miner_'+s for s in EXTRA_FEATURES)
V9_INDICES=tuple(TEMPORAL_V9_SELF_FEATURES.index(n) for n in EXTRA_FEATURES)


def option_features(pickup:PickupReading,miner:NaturalObservation):
    if not isinstance(pickup,PickupReading) or not isinstance(miner,NaturalObservation):
        raise TypeError('Only own camera-visible item and real V9 mining observations')
    if (pickup.features.shape!=(len(PICKUP_FEATURES),) or
        miner.own.shape!=(len(TEMPORAL_V9_SELF_FEATURES),)):
        raise ValueError('Incompatible player-visible pickup/V9 schema')
    vec=np.concatenate((pickup.features,np.asarray(miner.own)[list(V9_INDICES)]))
    if vec.shape!=(len(FEATURES),) or not np.isfinite(vec).all():
        raise ValueError('Nonfinite option chooser observation')
    vec=np.clip(vec,-8.,8.).astype(np.float32)
    vec.setflags(write=False)
    return vec


def reward_from_real_own_outcome(*,wood_inventory_gain,wood_mined,death,ticks):
    """Scorer/training labels only: own actual inventory, own mined stat,
    own health. No reward term is ever injected into policy observation.
    """
    if (type(wood_inventory_gain) is not int or wood_inventory_gain<0 or
        type(wood_mined) is not int or wood_mined<0 or
        type(death) is not bool or type(ticks) is not int or
        not 1<=ticks<=200):
        raise ValueError('Only actual vanilla player outcome and time allowed')
    return (25.*min(wood_inventory_gain,3) + 3.*min(wood_mined,3)
            - .04*ticks/6 - 8.*int(death))


def validate_real_row(row):
    exact={'schema','before','after','option','ticks','wood_mined',
           'wood_inventory_gain','death','done'}
    if not isinstance(row,dict) or set(row)!=exact or row.get('schema')!=SCHEMA:
        raise ValueError('Real option trace requires exact separate schema')
    before=np.asarray(row.get('before'),dtype=np.float32)
    after=np.asarray(row.get('after'),dtype=np.float32)
    action=row.get('option');ticks=row.get('ticks')
    gain=row.get('wood_inventory_gain');mined=row.get('wood_mined')
    death=row.get('death');done=row.get('done')
    if (before.shape!=(len(FEATURES),) or after.shape!=(len(FEATURES),) or
        not np.isfinite(before).all() or not np.isfinite(after).all() or
        type(action) is not int or not 0<=action<2 or
        type(ticks) is not int or not 1<=ticks<=200 or
        type(done) is not bool):
        raise ValueError('Invalid real player observation/action transition')
    reward=reward_from_real_own_outcome(wood_inventory_gain=gain,
         wood_mined=mined,death=death,ticks=ticks)
    if not math.isfinite(reward):raise ValueError('Invalid real outcome reward')
    # NEVER use hand-authored chosen-option 'correct answer' labels.
    return (before,action,reward,ticks,after,done)


class OptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES),64),torch.nn.ReLU(),
            torch.nn.Linear(64,64),torch.nn.ReLU(),
            torch.nn.Linear(64,len(OPTIONS)))

    def forward(self,x):return self.layers(x)


class NaturalOptionQ:
    def __init__(self,*,seed=20261105,device='cpu',learning_rate=.0004):
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA option model unavailable')
        if device not in ('cpu','cuda'):
            raise ValueError('Unsupported option learner device')
        self.device=torch.device(device)
        torch.manual_seed(seed)
        self.net=OptionNet().to(self.device)
        self.target=OptionNet().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=learning_rate)
        self.rng=np.random.default_rng(seed)
        self.replay=[]
        self.updates=0
        self.actual_wood_outcomes=0

    def act(self,features,*,greedy=False,exploration=.16):
        data=np.asarray(features,dtype=np.float32)
        if data.shape!=(len(FEATURES),) or not np.isfinite(data).all():
            raise ValueError('Invalid current own option observation')
        if not 0<=exploration<=1:
            raise ValueError('Invalid exploration rate')
        if not greedy and self.rng.random()<exploration:
            return int(self.rng.integers(len(OPTIONS)))
        with torch.no_grad():
            return int(self.net(torch.tensor(data.copy(),device=self.device)).argmax().item())

    def train_from_real_rows(self,rows,*,steps=64,batch_size=64,cql=.08):
        if type(steps) is not int or not 1<=steps<=2048 or type(batch_size) is not int or not 8<=batch_size<=512:
            raise ValueError('Invalid bounded optimizer budget')
        if not 0<=cql<=1.:raise ValueError('Invalid conservative replay penalty')
        if not rows:raise ValueError('No real option training rows')
        for raw in rows:
            before,a,reward,ticks,after,done=validate_real_row(raw)
            self.replay.append((before.tolist(),a,float(reward),ticks,after.tolist(),done))
            self.actual_wood_outcomes+=int(raw['wood_inventory_gain'])
        del self.replay[:-200000]
        if len(self.replay)<batch_size:
            return dict(updates=self.updates,real_rows=len(self.replay),
                        actual_wood_outcomes=self.actual_wood_outcomes,loss=None)
        rng=self.rng
        losses=[]
        for _ in range(steps):
            idx=rng.integers(len(self.replay),size=batch_size)
            batch=[self.replay[int(i)] for i in idx]
            old=torch.tensor([r[0] for r in batch],dtype=torch.float32,device=self.device)
            action=torch.tensor([r[1] for r in batch],dtype=torch.long,device=self.device)
            reward=torch.tensor([r[2] for r in batch],dtype=torch.float32,device=self.device)
            after=torch.tensor([r[4] for r in batch],dtype=torch.float32,device=self.device)
            discount=torch.tensor([.99**(r[3]/6)*float(not r[5]) for r in batch],
                                  dtype=torch.float32,device=self.device)
            q=self.net(old)
            with torch.no_grad():
                best=self.net(after).argmax(dim=1,keepdim=True)
                value=self.target(after).gather(1,best).squeeze(1)
                target=reward+discount*value
            selected=q.gather(1,action[:,None]).squeeze(1)
            td=torch.nn.functional.smooth_l1_loss(selected,target)
            conservative=(torch.logsumexp(q,dim=1)-selected).mean()
            loss=td+cql*conservative
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('Nonfinite natural option Q update')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),2.)
            self.optimizer.step()
            self.updates+=1
            losses.append(float(loss.detach().cpu()))
            if self.updates%128==0:self.target.load_state_dict(self.net.state_dict())
        return dict(updates=self.updates,real_rows=len(self.replay),
                    actual_wood_outcomes=self.actual_wood_outcomes,
                    loss=round(sum(losses)/len(losses),5))

    def state(self):
        return dict(schema=SCHEMA,features=FEATURES,
                    model=self.net.state_dict(),target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    replay=self.replay,rng=self.rng.bit_generator.state,
                    updates=self.updates,real_wood_outcomes=self.actual_wood_outcomes)

    def restore(self,state,*,optimizer=True):
        if state.get('schema')!=SCHEMA or tuple(state.get('features',()))!=FEATURES:
            raise ValueError('Different natural option observation/checkpoint schema')
        self.net.load_state_dict(state['model'])
        self.target.load_state_dict(state['target'])
        self.updates=state['updates']
        self.actual_wood_outcomes=state['real_wood_outcomes']
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.replay=state['replay']
            self.rng.bit_generator.state=state['rng']
        self.net.eval()
        return self


def own_observation_fingerprint(miner:NaturalObservation):
    """Evaluator-only hash to detect a true identical starting observation.

    Hash of the exact original V9 OWN and current camera-visible state; never
    an actor input or a training reward. Valid at onset before action choice.
    """
    if not isinstance(miner,NaturalObservation):
        raise TypeError('Only typed current V9 player/camera observation')
    h=hashlib.sha256()
    for x in (miner.own,miner.crosshair_id,miner.surface_ids,
              miner.surface_data,miner.surface_mask):
        a=np.asarray(x)
        h.update(str(a.shape).encode('ascii'))
        h.update(a.dtype.str.encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()[:24]
