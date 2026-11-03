"""From-scratch *attack motor skill* on own visible Minecraft mining state.

Deliberately NARROW curriculum, NOT a general V6 agent or a speedrun.
The policy receives vanilla player crosshair plus visible crack animation
and its own previous action/time. No fixture target XYZ, world generator,
admin commands, game stats, hidden blocks or privileged rewards are inputs.
Same versioned feature encoder is used for train/frozen real Minecraft.
"""
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .agent_interface import AgentAction
from .mcio_status import PlayerStatusV1
from .mcio_mining_feedback import MiningFeedbackV1

SCHEMA='mining_attack_motor_v1_own_crack_stage_interval6'
FEATURE_NAMES=('crosshair_hits_block','crosshair_distance_per_6',
               'currently_breaking','crack_stage_per_10',
               'previous_attack','elapsed_ticks_per_120')
FEATURE_SIZE=len(FEATURE_NAMES)
ACTION_TICKS=6


def encode_attack_observation(status, feedback, *, previous_attack, elapsed_ticks):
    if not isinstance(status,PlayerStatusV1) or not isinstance(feedback,MiningFeedbackV1):
        raise TypeError('Only the actual player status and local crack feedback are permitted')
    if type(previous_attack) is not bool or type(elapsed_ticks) is not int or not 0<=elapsed_ticks<=72000:
        raise ValueError('Invalid locally known attack history/tick count')
    block=status.crosshair_block
    # Important: exact block identity is NOT a privileged target oracle here;
    # this motor-skill probe only uses whether vanilla client has a target.
    state=np.asarray((float(block is not None),
          min(block.distance/6.,1.) if block else 0.,
          float(feedback.actively_destroying),
          max(feedback.normalized_crack,0.),
          float(previous_attack),min(elapsed_ticks/120.,1.)),dtype=np.float32)
    if state.shape!=(FEATURE_SIZE,) or not np.isfinite(state).all():
        raise ValueError('Non-finite or malformed own-player attack input')
    state.setflags(write=False)
    return state


def action_for_attack(choice, remaining_ticks=ACTION_TICKS):
    if type(choice) is not int or choice not in (0,1):
        raise ValueError('Attack choice must be a discrete integer 0 or 1')
    if type(remaining_ticks) is not int or remaining_ticks<1:
        raise ValueError('remaining_ticks must be positive')
    return AgentAction(keys=('LEFT_BUTTON',) if choice else (),
                       ticks=min(ACTION_TICKS,remaining_ticks))


def step_reward(previous,next_view,*,mined_new_block):
    """Training-only reward from OWN visible crack change + real mined event.

    No controller distance, placed-log position or teacher angular bearing.
    Zero-credit if broken successfully but crack stage disappears: only the
    independently observed Minecraft mined stat awards terminal success.
    """
    if not isinstance(previous,MiningFeedbackV1) or not isinstance(next_view,MiningFeedbackV1):
        raise TypeError('Require genuine typed client break feedback')
    if type(mined_new_block) is not bool: raise ValueError('Boolean mined event required')
    was=previous.crack_stage if previous.actively_destroying else -1
    now=next_view.crack_stage if next_view.actively_destroying else -1
    progress=max(0,now-max(-1,was)) if now>=0 else 0
    abandoned=(was>=0 and now<0 and not mined_new_block)
    return (.35*progress - (.40*(was+1) if abandoned else 0.)
            + (12. if mined_new_block else 0.) - .02)


@dataclass(frozen=True)
class AttackStep:
    features: np.ndarray
    action: int
    reward: float


def discounted_returns(steps,gamma=.94):
    if not steps:raise ValueError('No trajectory')
    if not 0<gamma<=1:raise ValueError('Invalid discount')
    out=np.empty(len(steps),dtype=np.float32)
    running=0.
    for i in range(len(steps)-1,-1,-1):
        running=float(steps[i].reward)+gamma*running
        if not isfinite(running): raise ValueError('Non-finite return')
        out[i]=running
    return out


class MiningAttackLearner:
    """Small on-policy actor-critic, initialized strictly from random seed."""
    def __init__(self,*,seed=46131,device='auto',learning_rate=.004):
        import torch
        from torch import nn
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
            raise RuntimeError('Unavailable attack learner device')
        self.torch=torch
        self.device=torch.device(device)
        torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(FEATURE_SIZE,32),nn.Tanh(),
                               nn.Linear(32,32),nn.Tanh()).to(self.device)
        self.actor=nn.Linear(32,2).to(self.device)
        self.critic=nn.Linear(32,1).to(self.device)
        self.optim=torch.optim.Adam([*self.net.parameters(),*self.actor.parameters(),
                                     *self.critic.parameters()],lr=learning_rate)
        self.updates=0

    def forward(self,values):
        x=self.net(values)
        return self.actor(x),self.critic(x).squeeze(-1)

    def act(self,features,*,greedy=False):
        if not isinstance(features,np.ndarray) or features.shape!=(FEATURE_SIZE,) or not np.isfinite(features).all():
            raise ValueError('Bad player memory feature shape')
        torch=self.torch
        with torch.no_grad():
            state=torch.as_tensor(np.array(features,copy=True),dtype=torch.float32,device=self.device)
            logits,_=self.forward(state[None,:])
            d=torch.distributions.Categorical(logits=logits)
            index=(logits.argmax(dim=-1) if greedy else d.sample())
            return int(index.item()),float(d.probs[0,1].item())

    def update(self,steps):
        if not steps:raise ValueError('No real Minecraft steps for on-policy update')
        torch=self.torch
        X=np.stack([step.features for step in steps]).astype(np.float32)
        y=np.asarray([step.action for step in steps],dtype=np.int64)
        R=discounted_returns(steps)
        feats=torch.tensor(X,device=self.device)
        choices=torch.tensor(y,device=self.device)
        returns=torch.tensor(R,device=self.device)
        logits,values=self.forward(feats)
        distr=torch.distributions.Categorical(logits=logits)
        advantage=returns-values.detach()
        actor_loss=-(distr.log_prob(choices)*advantage).mean()
        value_loss=(values-returns).square().mean()
        loss=actor_loss+.20*value_loss-.012*distr.entropy().mean()
        if not bool(torch.isfinite(loss).item()):raise RuntimeError('Nonfinite attack learning loss')
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*self.net.parameters(),*self.actor.parameters(),
                                       *self.critic.parameters()],1.)
        self.optim.step()
        self.updates+=1
        return dict(loss=round(float(loss.detach()),4),actor_loss=round(float(actor_loss.detach()),4),
                    critic_loss=round(float(value_loss.detach()),4),
                    transitions=len(steps),updates=self.updates)

    def save(self,path,metadata):
        import os
        from pathlib import Path
        path=Path(path)
        if path.exists():raise FileExistsError('Refusing to replace prior checkpoint')
        data=dict(schema=SCHEMA,features=FEATURE_NAMES,
                  state=self.net.state_dict(),actor=self.actor.state_dict(),
                  critic=self.critic.state_dict(),updates=self.updates,
                  metadata=metadata)
        staging=path.with_suffix('.tmp')
        if staging.exists():raise FileExistsError('Existing staging file')
        self.torch.save(data,staging)
        os.replace(staging,path)

    def load(self,path):
        data=self.torch.load(path,map_location=self.device,weights_only=True)
        if data['schema']!=SCHEMA or tuple(data['features'])!=FEATURE_NAMES:
            raise ValueError('Attack policy checkpoint and observation schema mismatch')
        self.net.load_state_dict(data['state'])
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.updates=int(data['updates'])
        return data['metadata']
