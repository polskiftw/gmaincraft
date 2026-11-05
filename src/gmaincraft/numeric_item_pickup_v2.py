"""Versioned item-pickup practice: egocentric item memory and off-policy Q.

Inputs are ONLY camera-filtered ItemEntity sightings, own position/yaw,
own prior actions, elapsed ticks and inventory. The lab's generated case,
item's hidden location and game commands never reach the model. Real
Minecraft transitions (not scripted 'correct' actions) train double Q.
"""
from dataclasses import dataclass
from math import cos, radians, sin, hypot, pi, isfinite
import random
import numpy as np
import torch

from .numeric_item_pickup import (
    ACTIONS, PickupReading, VisiblePickupMemory, motor_action)

SCHEMA = 'numeric_pickup_egocentric_replay_q_v2'
CASE_SCHEMA = 'randomized_world_yaw_egocentric_bearing_v2'
REWARD_SCHEMA = 'own_item_potential_plus_actual_inventory_v2'
FEATURE_NAMES = (
    'visible_now', 'remembered', 'forward_relative_div_6',
    'right_relative_div_6', 'height_relative_div_6',
    'distance_div_6', 'screen_u_centered', 'screen_v_centered',
    'verified_ray_fraction', 'memory_age_div_120', 'own_pitch_div_90',
    'own_previous_forward_displacement_div_3',
    'own_previous_right_displacement_div_3',
    'own_previous_action', 'own_inventory_div_8',
)


@dataclass(frozen=True)
class DirectionalCase:
    dx: float
    dz: float
    yaw: float
    pitch: float = 30.0


def directional_cases_for_seed(seed, count):
    """Training fixture ONLY. No case-derived value enters observations.

    Full 360-degree WORLD yaw means absolute +Z, constant-forward, and
    overfitting the old +/-15-degree yaw can no longer solve the suite.
    The training/evaluation worlds are disjoint by caller-supplied seed.
    """
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        yaw = rng.uniform(-180., 180.)
        radius = rng.uniform(2.5, 4.8)
        draw = rng.random()
        if draw < .40:
            offset = rng.uniform(-20., 20.)
        elif draw < .90:
            offset = rng.choice((-1, 1))*rng.uniform(20., 65.)
        else:
            offset = rng.choice((-1, 1))*rng.uniform(65., 130.)
        # Minecraft yaw=0 looks +Z; yaw=+90 looks -X.
        bearing = radians(yaw + offset)
        dx = -sin(bearing)*radius
        dz = cos(bearing)*radius
        cases.append(DirectionalCase(dx, dz, yaw))
    return cases


def egocentric_offset(dx, dz, yaw_degrees):
    angle = radians(yaw_degrees)
    return (-sin(angle)*dx + cos(angle)*dz,
            -cos(angle)*dx - sin(angle)*dz)


class EgocentricPickupMemory(VisiblePickupMemory):
    """Rotationally equivariant target observation; no hidden navigation oracle."""
    def __init__(self):
        super().__init__()
        self.prev_player_xyz = None

    def observe(self, player, items, *, elapsed_ticks, last_action=0):
        previous_xyz = self.prev_player_xyz
        original = super().observe(player, items,
            elapsed_ticks=elapsed_ticks, last_action=last_action)
        self.prev_player_xyz = player.player_xyz
        dx, dy, dz = (tuple(self.xyz[j]-player.player_xyz[j] for j in range(3))
                      if self.xyz is not None else (0.,0.,0.))
        front, right = egocentric_offset(dx, dz, player.yaw_degrees)
        moved_dx, moved_dz = ((player.player_xyz[0]-previous_xyz[0],
                              player.player_xyz[2]-previous_xyz[2])
                             if previous_xyz is not None else (0.,0.))
        moved_front, moved_right = egocentric_offset(
            moved_dx, moved_dz, player.yaw_degrees)
        # Preserve current sensor projections, not a projection of stale memory.
        v1 = original.features
        held = sum(slot.count for slot in
                   player.main_inventory + player.offhand_inventory)
        features = np.asarray((
            float(original.item_seen), float(self.xyz is not None),
            front/6.,right/6.,dy/6.,
            min(original.distance_to_remembered/6., 8.)
                if original.distance_to_remembered is not None else 0.,
            float(v1[6]),float(v1[7]),float(v1[11]),float(v1[12]),
            player.pitch_degrees/90.,moved_front/3.,moved_right/3.,
            last_action/(len(ACTIONS)-1),min(held/8,8.)),dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite egocentric player observation')
        features = np.clip(features,-8.,8.)
        features.setflags(write=False)
        return PickupReading(features, original.distance_to_remembered,
                             original.item_seen, original.visible_candidate_count)


def progress_reward(before, after, duration, *, inventory_gain, failed):
    """All reward inputs are *own* memory/observation and own inventory.

    Only real inventory gains count as the terminal success. This potential
    difference discourages repetitive left/right oscillations from farming
    positive progress while retaining dense movement feedback.
    """
    if duration not in (6,12,24):
        raise ValueError('Invalid real Minecraft action duration')
    if not isinstance(inventory_gain,int) or inventory_gain < 0:
        raise ValueError('Invalid own inventory delta')
    gamma = .985**(duration/6.)
    potential_before = 2.5*(8.-min(8., before)) if before is not None else 0.
    potential_after = 2.5*(8.-min(8., after)) if after is not None else 0.
    return (gamma*potential_after - potential_before
            - .035*(duration/6.) + 25.*inventory_gain - 3.*int(failed))


class PickupQNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_NAMES),96),torch.nn.ReLU(),
            torch.nn.Linear(96,96),torch.nn.ReLU(),
            torch.nn.Linear(96,len(ACTIONS)))

    def forward(self, x):
        return self.layers(x)


class ReplayPickupQ:
    """Replay-backed double Q, real movement transitions only, strict resume."""
    def __init__(self, *, seed=1, device='cpu', learning_rate=.0006):
        self.device=torch.device('cuda' if device=='auto' and
                 torch.cuda.is_available() else 'cpu' if device=='auto' else device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.manual_seed(seed)
        self.net=PickupQNet().to(self.device)
        self.target=PickupQNet().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=learning_rate,
                                          weight_decay=0.00001)
        self.rng=np.random.default_rng(seed)
        self.training=True
        self.replay=[]
        self.success_replay=[]
        self.episodes_seen=0
        self.updates=0

    def act(self,reading,*,greedy=False,uniform_random=False):
        if not isinstance(reading,PickupReading) or len(reading.features)!=len(FEATURE_NAMES):
            raise TypeError('Q actor requires current egocentric player/item observation')
        if uniform_random:
            return int(self.rng.integers(len(ACTIONS)))
        epsilon=(max(.12,.70-.003*self.episodes_seen)
                 if self.training else .07)
        if not greedy and self.rng.random()<epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        with torch.no_grad():
            data=torch.as_tensor(reading.features.copy(),device=self.device)
            return int(self.net(data).argmax().item())

    def update(self,trajectory):
        if not trajectory:raise ValueError('Cannot train on empty real transitions')
        validated=[]
        for old,action,reward,duration,new,terminal in trajectory:
            if (np.asarray(old).shape!=(len(FEATURE_NAMES),) or
                np.asarray(new).shape!=(len(FEATURE_NAMES),) or
                not np.isfinite(old).all() or not np.isfinite(new).all() or
                type(action) is not int or not 0<=action<len(ACTIONS) or
                duration not in (6,12,24) or not isfinite(reward) or
                type(terminal) is not bool):
                raise ValueError('Invalid recorded real own-state Q transition')
            # Primitive lists remain compatible with torch.load(weights_only=True).
            validated.append((np.asarray(old).tolist(),action,float(reward),
                duration,np.asarray(new).tolist(),terminal))
        self.replay.extend(validated)
        del self.replay[:-8000]
        # Retain entire sequences that truly ended with an inventory gain;
        # never synthesize expert labels or attach teacher actions.
        if any(row[2]>=20. for row in validated):
            self.success_replay.extend(validated)
            del self.success_replay[:-3000]
        self.episodes_seen+=1
        if len(self.replay)<64:
            return dict(updates=self.updates,buffer=len(self.replay),
                        real_success_examples=len(self.success_replay),loss=None)
        losses=[]
        for _ in range(12):
            positive=min(16,len(self.success_replay))
            rest=64-positive
            ids=self.rng.integers(len(self.replay),size=rest)
            batch=[self.replay[int(i)] for i in ids]
            if positive:
                ids=self.rng.integers(len(self.success_replay),size=positive)
                batch.extend(self.success_replay[int(i)] for i in ids)
            before=torch.tensor([row[0] for row in batch],dtype=torch.float32,
                                device=self.device)
            action=torch.tensor([row[1] for row in batch],dtype=torch.long,
                                device=self.device).unsqueeze(1)
            reward=torch.tensor([row[2] for row in batch],dtype=torch.float32,
                                device=self.device)
            discounts=torch.tensor([.985**(row[3]/6.)*(not row[5]) for row in batch],
                                  dtype=torch.float32,device=self.device)
            after=torch.tensor([row[4] for row in batch],dtype=torch.float32,
                               device=self.device)
            pred=self.net(before).gather(1,action).squeeze(1)
            with torch.no_grad():
                best=self.net(after).argmax(dim=1,keepdim=True)
                v=self.target(after).gather(1,best).squeeze(1)
                target=reward+discounts*v
            loss=torch.nn.functional.smooth_l1_loss(pred,target)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('Nonfinite pickup TD error')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),2.)
            self.optimizer.step()
            self.updates+=1
            losses.append(float(loss.detach().cpu()))
            if self.updates%96==0:
                self.target.load_state_dict(self.net.state_dict())
        return dict(updates=self.updates,buffer=len(self.replay),
                    real_success_examples=len(self.success_replay),
                    loss=round(sum(losses)/len(losses),5),
                    epsilon=round(max(.12,.70-.003*self.episodes_seen),4))

    def state(self):
        return dict(schema=SCHEMA,net=self.net.state_dict(),
                    target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
                    updates=self.updates,episodes_seen=self.episodes_seen,
                    rng=self.rng.bit_generator.state,replay=self.replay,
                    success_replay=self.success_replay)

    def restore(self,state,*,optimizer=True):
        if state.get('schema')!=SCHEMA:
            raise ValueError('Not a matching replay pickup Q checkpoint')
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state['target'])
        self.updates=state['updates']
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.episodes_seen=state['episodes_seen']
            self.rng.bit_generator.state=state['rng']
            self.replay=state['replay']
            self.success_replay=state['success_replay']
        self.net.eval()
        return self
