"""From-scratch memory-first PPO actor for continuous NATURAL Minecraft survival.

No fixture position, scripted skill phase or pretrained policy. Its only
world objects are the game's actual bounded, first-collision camera samples
and vanilla crosshair; all other data are the player's own HUD/debug state.
The first-hit grid is PROVISIONAL, not a validated human recognizability
threshold (transparent/dark/small objects still need calibration).
"""
from dataclasses import dataclass
from math import cos, log1p, radians, sin

import numpy as np

from .agent_interface import AgentAction
from .mcio_status import PlayerStatusV1
from .player_knowledge import PlayerKnowledgeV1
from .mcio_surface_objects import SurfaceObjectsV1
from .mcio_mining_feedback import MiningFeedbackV1
from .ppo_math import timed_gae

SCHEMA='natural_memory_online_v1_first_hit_objects_ppo6'
DECISION_TICKS=6
MAX_OBJECTS=24
ID_BYTES=48
MOVE_KEYS=((),('W',),('S',),('A',),('D',),('W','SPACE'))
LOOK_DELTAS=((0.,0.),(-100.,0.),(100.,0.),(-200.,0.),(200.,0.),(0.,-75.),(0.,75.))
HEAD_SIZES=(len(MOVE_KEYS),len(LOOK_DELTAS),2)
SELF_FEATURES=(
 'player_x_div_1024','player_y_div_128','player_z_div_1024',
 'relative_x_from_start_div_64','relative_z_from_start_div_64',
 'yaw_sin','yaw_cos','pitch_div_90','health_div_20',
 'hunger_div_20','saturation_div_20','air_div_300','hotbar_div_8',
 'own_log_count_div_16','own_inventory_count_div_64',
 'crosshair_target_present','crosshair_distance_div_8',
 'crosshair_dx_div_8','crosshair_dy_div_8','crosshair_dz_div_8',
 'mining_feedback_available','mining_active','mining_crack_div_10',
 'last_move_div_5','last_look_div_6','last_attack',
 'local_visits_log_div_8','own_elapsed_ticks_div_72000',
 'visible_object_count_div_24',
)
OBJECT_FEATURES=(
 'relative_dx_div_32','relative_dy_div_32','relative_dz_div_32',
 'screen_u_2_minus_1','screen_v_2_minus_1','distance_div_64',
 'sample_fraction','sky_light_div_15','block_light_div_15',
 'face_count_div_6',
)


def wood_id(name):
    return isinstance(name,str) and name.startswith('minecraft:') and name.endswith(
        ('_log','_wood','_stem','_hyphae'))


def encode_id(name):
    """Literal UTF-8 registry bytes (+1 for 0 padding), never a handcoded oak label."""
    result=np.zeros(ID_BYTES,dtype=np.uint8)
    if name:
        raw=str(name).encode('utf8')[:ID_BYTES]
        result[:len(raw)]=np.frombuffer(raw,dtype=np.uint8)+1
    return result


@dataclass(frozen=True)
class NaturalObservation:
    own: np.ndarray
    crosshair_id: np.ndarray
    surface_ids: np.ndarray
    surface_data: np.ndarray
    surface_mask: np.ndarray


def observe(player,status,surface,mining,*,start_xyz,last_action=(0,0,0),
            visit_count=0,elapsed_ticks=0):
    if not isinstance(player,PlayerKnowledgeV1) or not isinstance(status,PlayerStatusV1) \
            or not isinstance(surface,SurfaceObjectsV1):
        raise TypeError('Expected real typed player/status/first-hit surfaces')
    if mining is not None and not isinstance(mining,MiningFeedbackV1):
        raise TypeError('Mining feedback must be a typed own-player reading or missing')
    if len(start_xyz)!=3 or any(not np.isfinite(v) for v in start_xyz):
        raise ValueError('Invalid remembered starting coordinates')
    if (len(last_action)!=3 or any(type(i) is not int or not 0<=i<HEAD_SIZES[j]
                                   for j,i in enumerate(last_action)) or
        type(visit_count) is not int or visit_count<0 or
        type(elapsed_ticks) is not int or elapsed_ticks<0):
        raise ValueError('Invalid own action/visit/tick history')
    xyz=player.player_xyz
    target=status.crosshair_block
    hit_delta=tuple(target.xyz[j]-xyz[j] for j in range(3)) if target else (0,0,0)
    owned=sum(slot.count for slot in player.main_inventory+player.offhand_inventory)
    owned_logs=sum(slot.count for slot in player.main_inventory+player.offhand_inventory
                   if wood_id(slot.item_id))
    def bound(value,limit=1.):return float(np.clip(value,-limit,limit))
    numbers=(
        bound(xyz[0]/1024,32),bound(xyz[1]/128,4),bound(xyz[2]/1024,32),
        bound((xyz[0]-start_xyz[0])/64,16),bound((xyz[2]-start_xyz[2])/64,16),
        sin(radians(player.yaw_degrees)),cos(radians(player.yaw_degrees)),
        bound(player.pitch_degrees/90),bound(player.health/20,2),
        bound(status.hunger/20,1),bound(status.saturation/20,5),
        bound(status.air_supply/300,4),status.selected_hotbar_slot/8,
        min(owned_logs/16,4),min(owned/64,4),
        float(target is not None),min(target.distance/8,4) if target else 0.,
        *(bound(v/8) for v in hit_delta),
        float(mining is not None),
        float(mining.actively_destroying) if mining else 0.,
        bound(mining.normalized_crack,1) if mining else -1.,
        last_action[0]/(HEAD_SIZES[0]-1),last_action[1]/(HEAD_SIZES[1]-1),
        float(last_action[2]),min(log1p(visit_count)/8,1),
        min(elapsed_ticks/72000,4),min(len(surface.objects)/MAX_OBJECTS,4),
    )
    own=np.asarray(numbers,dtype=np.float32)
    if own.shape!=(len(SELF_FEATURES),) or not np.isfinite(own).all():
        raise ValueError('Invalid own-player numeric features')
    surface_ids=np.zeros((MAX_OBJECTS,ID_BYTES),dtype=np.uint8)
    surface_data=np.zeros((MAX_OBJECTS,len(OBJECT_FEATURES)),dtype=np.float32)
    surface_mask=np.zeros(MAX_OBJECTS,dtype=np.float32)
    # Model budget is top 24 first-hit block surfaces, ranked by projected ray
    # coverage then distance. This is a finite input size, NOT an assertion
    # that other block surfaces are invisible. No hidden world scan.
    ranked=sorted(surface.objects,key=lambda x:(-len(x.ray_indices),x.nearest_distance,x.xyz))[:MAX_OBJECTS]
    for idx,obj in enumerate(ranked):
        dx,dy,dz=(obj.xyz[j]-xyz[j] for j in range(3))
        u,v=obj.screen_center_uv
        features=(bound(dx/32,4),bound(dy/32,4),bound(dz/32,4),
                  2*u-1,2*v-1,min(obj.nearest_distance/64,4),
                  obj.sample_fraction,obj.sky_light/15,obj.block_light/15,
                  len(obj.visible_faces)/6)
        surface_data[idx]=features
        surface_ids[idx]=encode_id(obj.block_id)
        surface_mask[idx]=1
    arrays=(own,encode_id(target.block_id if target else None),surface_ids,
            surface_data,surface_mask)
    for array in arrays:
        if not np.isfinite(array).all():raise ValueError('Nonfinite player observation')
        array.setflags(write=False)
    return NaturalObservation(*arrays)


def action_from_heads(indices,remaining_ticks=DECISION_TICKS):
    if (not isinstance(indices,(tuple,list)) or len(indices)!=3 or
            any(type(x) is not int or not 0<=x<HEAD_SIZES[j]
                for j,x in enumerate(indices)) or
            type(remaining_ticks) is not int or remaining_ticks<1):
        raise ValueError('Invalid factorized ordinary Minecraft action')
    move,look,attack=indices
    return AgentAction(keys=MOVE_KEYS[move]+(('LEFT_BUTTON',) if attack else ()),
                       look_dx=LOOK_DELTAS[look][0],look_dy=LOOK_DELTAS[look][1],
                       ticks=min(DECISION_TICKS,remaining_ticks))


@dataclass(frozen=True)
class NaturalTransition:
    observation: NaturalObservation
    action: tuple[int,int,int]
    reward: float
    value: float
    logp: float
    duration: int
    done: bool


class NaturalPPO:
    """Small attention-pooled byte-ID/block-memory actor-critic; no RGB encoder."""
    def __init__(self,*,seed=730011,device='auto',learning_rate=.00035,
                 epochs=3,minibatch=64):
        import torch
        from torch import nn
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda') or device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested policy accelerator unavailable')
        if not 1<=epochs<=8 or not 8<=minibatch<=512:
            raise ValueError('Invalid PPO hyperparameters')
        self.torch=torch
        self.device=torch.device(device)
        torch.manual_seed(seed)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.chars=nn.Embedding(257,16,padding_idx=0)
                self.object=nn.Sequential(nn.Linear(16+len(OBJECT_FEATURES),64),nn.Tanh(),
                                          nn.Linear(64,64),nn.Tanh())
                self.attention=nn.Linear(64,1)
                self.fuse=nn.Sequential(nn.Linear(len(SELF_FEATURES)+16+128,128),nn.Tanh(),
                                         nn.Linear(128,128),nn.Tanh())
                self.heads=nn.ModuleList([nn.Linear(128,n) for n in HEAD_SIZES])
                self.value=nn.Linear(128,1)

            def mean_bytes(self,tokens):
                raw=self.chars(tokens.long())
                mask=(tokens!=0).unsqueeze(-1)
                return (raw*mask).sum(dim=-2)/mask.sum(dim=-2).clamp(min=1)

            def forward(self,own,crosshair,ids,data,mask):
                objects=self.object(torch.cat((self.mean_bytes(ids),data.float()),dim=-1))
                active=mask>0
                attention=self.attention(objects).squeeze(-1).masked_fill(~active,-1e4)
                weights=attention.softmax(dim=-1)*mask
                weights=weights/weights.sum(dim=-1,keepdim=True).clamp(min=1)
                pooled=(objects*weights.unsqueeze(-1)).sum(dim=1)
                maximum=objects.masked_fill(~active.unsqueeze(-1),-1e4).max(dim=1).values
                maximum=torch.where(active.any(dim=1,keepdim=True),maximum,
                                          torch.zeros_like(maximum))
                x=self.fuse(torch.cat((own.float(),self.mean_bytes(crosshair),
                                       pooled,maximum),dim=-1))
                return tuple(head(x) for head in self.heads),self.value(x).squeeze(-1)


        self.net=Network().to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=learning_rate,eps=1e-5)
        self.epochs=epochs
        self.minibatch=minibatch
        self.updates=0
        self.seed=seed

    def _batch(self,observations):
        torch=self.torch
        tensors=[]
        for name in ('own','crosshair_id','surface_ids','surface_data','surface_mask'):
            values=np.stack([getattr(o,name) for o in observations])
            tensors.append(torch.as_tensor(np.array(values,copy=True),device=self.device))
        return tuple(tensors)

    def act(self,observation,*,greedy=False):
        if not isinstance(observation,NaturalObservation):raise TypeError('Invalid typed natural observation')
        self.net.eval()
        with self.torch.no_grad():
            logits,value=self.net(*self._batch([observation]))
            distributions=[self.torch.distributions.Categorical(logits=v) for v in logits]
            choices=[d.probs.argmax(-1) if greedy else d.sample() for d in distributions]
            logp=sum(d.log_prob(choice) for d,choice in zip(distributions,choices))
        return tuple(int(x.item()) for x in choices),float(logp.item()),float(value.item())

    def update(self,transitions,*,bootstrap_value=0.):
        if not transitions:raise ValueError('No real experience')
        torch=self.torch
        rewards=[t.reward for t in transitions]
        advantages,returns=timed_gae(rewards,[t.value for t in transitions],
                   [t.duration for t in transitions],[t.done for t in transitions],
                   last_value=bootstrap_value)
        if len(transitions)>1:
            advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        obs=self._batch([t.observation for t in transitions])
        actions=torch.tensor([t.action for t in transitions],device=self.device,dtype=torch.long)
        old_logp=torch.tensor([t.logp for t in transitions],device=self.device)
        adv=torch.tensor(advantages,device=self.device)
        ret=torch.tensor(returns,device=self.device)
        self.net.train()
        records=[]
        for _ in range(self.epochs):
            for ix in torch.randperm(len(transitions),device=self.device).split(self.minibatch):
                logits,values=self.net(*(x[ix] for x in obs))
                distributions=[torch.distributions.Categorical(logits=x) for x in logits]
                logp=sum(d.log_prob(actions[ix,k]) for k,d in enumerate(distributions))
                entropy=sum(d.entropy() for d in distributions).mean()
                ratio=(logp-old_logp[ix]).exp()
                objective=torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
                value_loss=(values-ret[ix]).square().mean()
                loss=-objective+.30*value_loss-.014*entropy
                if not bool(torch.isfinite(loss).item()):raise RuntimeError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),.7)
                self.optimizer.step()
                records.append((float(loss.detach()),float(value_loss.detach()),float(entropy.detach())))
        self.updates+=1
        return dict(update=self.updates,transitions=len(transitions),
                    loss=round(float(np.mean([x[0] for x in records])),4),
                    value_loss=round(float(np.mean([x[1] for x in records])),4),
                    entropy=round(float(np.mean([x[2] for x in records])),4),
                    reward_mean=round(float(np.mean(rewards)),4))

    def state(self):
        torch=self.torch
        return dict(schema=SCHEMA,feature_order=SELF_FEATURES,object_order=OBJECT_FEATURES,
                    action_sizes=HEAD_SIZES,net=self.net.state_dict(),
                    optimizer=self.optimizer.state_dict(),updates=self.updates,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
                    seed=self.seed)

    def restore(self,state,*,optimizer=True):
        if state['schema']!=SCHEMA or tuple(state['feature_order'])!=SELF_FEATURES or \
                tuple(state['object_order'])!=OBJECT_FEATURES or tuple(state['action_sizes'])!=HEAD_SIZES:
            raise ValueError('Incompatible natural-memory PPO schema or action layout')
        self.net.load_state_dict(state['net'])
        self.updates=int(state['updates'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.torch.set_rng_state(state['torch_rng'].cpu())
            if self.device.type=='cuda' and state['cuda_rng']:
                self.torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda_rng']])
        return self
