"""From-scratch, ordinary-control pickup practice on player-visible item eyes.

This is a NEW dedicated practice skill, not a silent modification of the
legacy V9/Q trained wood actor or an evaluator-chained Minecraft speedrun.
Its only target evidence is a typed *current camera-visible* dropped
ItemEntity sample and stale player-owned memory of prior visible samples.
The scorer watches actual player-owned inventory acquisition. The actor
chooses Minecraft key/mouse controls; never tell it to move to a position
or pick up a resource with an admin command.
"""
from dataclasses import dataclass
from math import cos,hypot,radians,sin
import numpy as np
import torch

from .agent_interface import AgentAction
from .mcio_visible_items import VisibleItemsV1
from .player_knowledge import PlayerKnowledgeV1

SCHEMA='numeric_item_pickup_actorcritic_v1'
ACTIONS=(
    ('forward6',('W',),0.,0.,6),
    ('forward24',('W',),0.,0.,24),
    ('strafe_left12',('A',),0.,0.,12),
    ('strafe_right12',('D',),0.,0.,12),
    ('forward_left12',('W','A'),0.,0.,12),
    ('forward_right12',('W','D'),0.,0.,12),
    ('back12',('S',),0.,0.,12),
    ('look_left6',(),-60.,0.,6),
    ('look_right6',(),60.,0.,6),
    ('look_down6',(),0.,45.,6),
    ('look_up6',(),0.,-45.,6),
    ('idle6',(),0.,0.,6),
)
FEATURE_NAMES=(
    'currently_visible','has_player_memory','item_dx_div_8',
    'item_dy_div_8','item_dz_div_8','item_distance_div_8',
    'current_screen_u_centered','current_screen_v_centered',
    'yaw_sin','yaw_cos','pitch_div_90',
    'current_visible_ray_fraction','memory_age_div_120',
    'last_motor_div_11','own_held_item_div_8',
)


@dataclass(frozen=True)
class PickupReading:
    features:np.ndarray
    distance_to_remembered:float|None
    item_seen:bool
    visible_candidate_count:int


class VisiblePickupMemory:
    """Own remembered sightings are timestamped and NEVER magically refreshed."""
    def __init__(self):
        self.xyz=None
        self.last_seen_tick=None
        self.last_item_id=None

    def observe(self,player,items,*,elapsed_ticks,last_action=0):
        if not isinstance(player,PlayerKnowledgeV1):
            raise TypeError('Pickup agent needs actual typed player-only state')
        if not isinstance(items,VisibleItemsV1):
            raise TypeError('Pickup agent only sees filtered current-camera ItemEntity data')
        if (type(elapsed_ticks) is not int or elapsed_ticks<0 or
            type(last_action) is not int or not 0<=last_action<len(ACTIONS)):
            raise ValueError('Bad own action/time history')
        # Every candidate was already proved camera-visible by the sensor.
        # Choose a stable *observation focus*, NOT a movement/aim action:
        # track a remembered previously visible item when possible, else
        # use the existing sensor's bounded nearest-first observation order.
        # The unrestricted future multi-object actor will receive the full
        # set through a separately versioned learned-attention branch.
        seen=bool(items.items)
        u=v=0.
        ray_fraction=0.
        if seen:
            if self.xyz is not None and self.last_item_id is not None:
                matches=[x for x in items.items if x.item_id==self.last_item_id]
                if matches:
                    item=min(matches,key=lambda x:sum(
                        (x.xyz[j]-self.xyz[j])**2 for j in range(3)))
                else:
                    item=items.items[0]
            else:
                item=items.items[0]
            self.xyz=item.xyz
            self.last_item_id=item.item_id
            self.last_seen_tick=elapsed_ticks
            u,v=(2*item.screen_uv[0]-1,2*item.screen_uv[1]-1)
            ray_fraction=item.visible_samples/5
        if self.last_seen_tick is not None and elapsed_ticks-self.last_seen_tick>600:
            # Stale remembered coordinates eventually expire; do not
            # pretend items remain forever after leaving the camera.
            self.xyz=None
            self.last_seen_tick=None
            self.last_item_id=None
        known=self.xyz is not None
        dx,dy,dz=(tuple(self.xyz[i]-player.player_xyz[i] for i in range(3))
                  if known else (0.,0.,0.))
        dist=(float(np.sqrt(dx*dx+dy*dy+dz*dz)) if known else None)
        age=(elapsed_ticks-self.last_seen_tick if known else 0)
        held=sum(slot.count for slot in player.main_inventory+player.offhand_inventory)
        numbers=(float(seen),float(known),dx/8,dy/8,dz/8,
                 min(dist/8,8) if known else 0.,u,v,
                 sin(radians(player.yaw_degrees)),
                 cos(radians(player.yaw_degrees)),player.pitch_degrees/90,
                 ray_fraction,min(age/120,5),last_action/(len(ACTIONS)-1),
                 min(held/8,8))
        vec=np.asarray(numbers,dtype=np.float32)
        if vec.shape!=(len(FEATURE_NAMES),) or not np.isfinite(vec).all():
            raise ValueError('Nonfinite pickup observation')
        vec=np.clip(vec,-8.,8.)
        vec.setflags(write=False)
        return PickupReading(vec,dist,seen,len(items.items))


def motor_action(index):
    if type(index) is not int or not 0<=index<len(ACTIONS):
        raise ValueError('Unknown primitive pickup motor')
    _,keys,dx,dy,ticks=ACTIONS[index]
    return AgentAction(keys=keys,look_dx=dx,look_dy=dy,ticks=ticks)


class PickupNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_NAMES),64),torch.nn.Tanh(),
            torch.nn.Linear(64,64),torch.nn.Tanh())
        self.actor=torch.nn.Linear(64,len(ACTIONS))
        self.value=torch.nn.Linear(64,1)

    def forward(self,x):
        hidden=self.base(x)
        return self.actor(hidden),self.value(hidden).squeeze(-1)


class NumericPickupActor:
    def __init__(self,*,seed=511103,device='cpu',learning_rate=.0009):
        if not .000001<=learning_rate<=.01:
            raise ValueError('Invalid learning rate')
        self.device=torch.device(
            'cuda' if device=='auto' and torch.cuda.is_available() else
            'cpu' if device=='auto' else device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.manual_seed(seed)
        self.net=PickupNet().to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        self.updates=0

    def act(self,observation,*,greedy=False,uniform_random=False):
        if not isinstance(observation,PickupReading):
            raise TypeError('Invalid player-visible pickup observation')
        x=torch.as_tensor(observation.features.copy(),device=self.device).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            logits,_=self.net(x)
            if uniform_random:
                # Reuse torch's exact checkpointed RNG, not a hidden oracle.
                action=int(torch.randint(len(ACTIONS),(1,),device=self.device).item())
            elif greedy:
                action=int(logits.argmax(dim=-1).item())
            else:
                action=int(torch.distributions.Categorical(logits=logits).sample().item())
        return action

    def update(self,transitions):
        """Episodic, duration-aware REINFORCE with learned own-state baseline."""
        if not transitions:
            raise ValueError('Cannot update from empty real action sequence')
        feats=np.stack([f for f,a,r,d in transitions])
        if feats.shape[1]!=len(FEATURE_NAMES):
            raise ValueError('Wrong stored features')
        actions=[a for f,a,r,d in transitions]
        if any(type(a) is not int or not 0<=a<len(ACTIONS) for a in actions):
            raise ValueError('Invalid episode action')
        values=[]
        total=0.
        for _,_,reward,duration in reversed(transitions):
            if duration not in (6,12,24):
                raise ValueError('Actual action duration unknown')
            total=float(reward)+(.99**(duration/6))*total
            values.append(total)
        returns=torch.tensor(values[::-1],dtype=torch.float32,device=self.device)
        data=torch.tensor(feats.copy(),dtype=torch.float32,device=self.device)
        acts=torch.tensor(actions,dtype=torch.long,device=self.device)
        self.net.train()
        logits,baseline=self.net(data)
        choice=torch.distributions.Categorical(logits=logits)
        advantage=(returns-baseline).detach()
        scaled=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-5) if len(transitions)>1 else advantage
        actor_loss=-(choice.log_prob(acts)*scaled).mean()
        critic_loss=torch.nn.functional.smooth_l1_loss(baseline,returns)
        entropy=choice.entropy().mean()
        loss=actor_loss+0.4*critic_loss-0.015*entropy
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError('Invalid pickup loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
        self.optimizer.step()
        self.updates+=1
        return dict(loss=round(float(loss.detach().cpu()),5),
                    actor=round(float(actor_loss.detach().cpu()),5),
                    value=round(float(critic_loss.detach().cpu()),5),
                    entropy=round(float(entropy.detach().cpu()),5),
                    updates=self.updates)

    def state(self):
        return dict(schema=SCHEMA,net=self.net.state_dict(),
                    optimizer=self.optimizer.state_dict(),updates=self.updates,
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])

    def restore(self,state,*,optimizer=True):
        if state.get('schema')!=SCHEMA:
            raise ValueError('Incompatible numeric item skill')
        self.net.load_state_dict(state['net'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.updates=state['updates']
            torch.set_rng_state(state['cpu_rng'])
            if self.device.type=='cuda' and state['cuda_rng']:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            self.updates=state['updates']
        self.net.eval()
        return self
