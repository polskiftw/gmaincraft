"""Experimental conservative off-policy skill learning from PLAYER-ONLY MCio replay.

The outcome goal is either an item first gained in actual owned inventory or
visible vanilla crack progress already recorded by the V9 history sensor. No
teacher actions, world seeds, coordinates of hidden targets or evaluator stats
enter these transitions. This is NOT a proven skill or a replacement for PPO.
"""
import json
from copy import deepcopy

import numpy as np

from .experience_archive import FIELDS,ObservedItemGoal
from .natural_online_policy import (NaturalObservation,encode_id,OBJECT_FEATURES,
                                    MOTOR_V7_HEAD_SIZES,TEMPORAL_V9_SCHEMA,
                                    TEMPORAL_V9_SELF_FEATURES)

SCHEMA='player_known_conservative_goal_q_v1'
CRACK_GOAL='effect:visible_crack_progress'
CRACK_RISE_INDEX=TEMPORAL_V9_SELF_FEATURES.index('crack_stage_rose_same_target_div_10')
CONTINUITY_INDEX=TEMPORAL_V9_SELF_FEATURES.index('same_vanilla_crosshair_target_as_previous')


def goal_transitions(meta,data,goal):
    """One-step REAL action consequences, truncated at true goal/death/end.

    The goal terminal and priority are TRAINING labels, never an actor sensor.
    The stored PPO shaped reward is deliberately ignored. In particular a
    'wood seen' event is never mistaken for actually obtaining that item.
    """
    if goal!=CRACK_GOAL:
        if not isinstance(goal,str) or not goal.startswith('minecraft:'):
            raise ValueError('Inventory goals must be vanilla items')
        ObservedItemGoal(goal)
    if (meta['observation_schema']!=TEMPORAL_V9_SCHEMA or
        tuple(meta['action_sizes'])!=MOTOR_V7_HEAD_SIZES):
        raise ValueError('Only original V9 player-history, 20-motor replay is supported')
    results=[]
    for i in range(meta['count']):
        if goal==CRACK_GOAL:
            own=data['after_own'][i]
            achieved=bool(own[CONTINUITY_INDEX]>.5 and own[CRACK_RISE_INDEX]>0)
        else:
            before=dict(json.loads(str(data['inventory_before'][i])))
            after=dict(json.loads(str(data['inventory_after'][i])))
            achieved=after.get(goal,0)>before.get(goal,0)
        done=bool(data['done'][i]) or achieved or i==meta['count']-1
        row={prefix+field:data[prefix+field][i] for prefix in ('before_','after_') for field in FIELDS}
        row.update(action=int(data['action'][i][0]),duration=int(data['duration'][i]),
                   achieved=achieved,done=done,priority=20. if achieved else 1.)
        results.append(row)
    # Increase sampling of real trajectories immediately preceding observed
    # positive effects, without fabricating positive results for other actions.
    for i,row in enumerate(results):
        if not row['achieved']:continue
        for back in range(1,13):
            j=i-back
            if j<0 or results[j]['done']:break
            results[j]['priority']=max(results[j]['priority'],4.)
    return results


class ReplayQSkill:
    """Goal-conditioned double-DQN with conservative logged-action penalty.

    Uses the exact archived V9 camera/own-state budget and ordinary 20-way
    motor. Double-DQN and logsumexp CQL reduce (but cannot eliminate) severe
    offline extrapolation errors. Long actions use game-tick discounting.
    """
    def __init__(self,*,seed=20261104,device='auto',lr=.0002):
        import torch
        from torch import nn
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda') or device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested Q-learner accelerator unavailable')
        self.torch=torch
        self.device=torch.device(device)
        torch.manual_seed(seed)
        size=len(TEMPORAL_V9_SELF_FEATURES)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.chars=nn.Embedding(257,16,padding_idx=0)
                self.obj=nn.Sequential(nn.Linear(16+len(OBJECT_FEATURES),64),nn.Tanh(),
                                       nn.Linear(64,48),nn.Tanh())
                self.fuse=nn.Sequential(nn.Linear(size+16+16+48,128),nn.Tanh(),
                                        nn.Linear(128,128),nn.Tanh())
                self.q=nn.Linear(128,MOTOR_V7_HEAD_SIZES[0])

            def tokens(self,ids):
                x=self.chars(ids.long())
                present=(ids!=0).unsqueeze(-1)
                return (x*present).sum(dim=-2)/present.sum(dim=-2).clamp(min=1)

            def forward(self,own,crosshair,ids,data,mask,goal):
                obj=self.obj(torch.cat((self.tokens(ids),data.float()),dim=-1))
                visible=mask.float().unsqueeze(-1)
                pooled=(obj*visible).sum(dim=1)/visible.sum(dim=1).clamp(min=1)
                x=self.fuse(torch.cat((own.float(),self.tokens(crosshair),
                          self.tokens(goal),pooled),dim=-1))
                return self.q(x)

        self.net=Net().to(self.device)
        self.target=deepcopy(self.net).to(self.device).eval()
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=lr,eps=1e-5)
        self.updates=0

    def _batch(self,examples,prefix,goal):
        out=[]
        for field in FIELDS:
            vals=np.stack([ex[prefix+field] for ex in examples])
            out.append(self.torch.as_tensor(np.array(vals,copy=True),device=self.device))
        out.append(self.torch.as_tensor(np.stack([encode_id(goal)]*len(examples)),device=self.device))
        return out

    def update(self,batch,goal,*,cql_weight=.1):
        if not batch or not 0<=cql_weight<=10:raise ValueError('Invalid Q batch / conservative penalty')
        torch=self.torch
        b=self._batch(batch,'before_',goal)
        a=self._batch(batch,'after_',goal)
        actions=torch.tensor([r['action'] for r in batch],device=self.device,dtype=torch.long)
        gain=torch.tensor([float(r['achieved']) for r in batch],device=self.device)
        duration=torch.tensor([r['duration'] for r in batch],device=self.device,dtype=torch.float32)
        terminal=torch.tensor([r['done'] for r in batch],device=self.device,dtype=torch.float32)
        self.net.train()
        qs=self.net(*b)
        chosen=qs.gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            best=self.net(*a).argmax(-1)
            next_q=self.target(*a).gather(1,best[:,None]).squeeze(1)
            expected=gain-.001*(duration/6)+torch.pow(torch.full_like(duration,.997),duration/6)*(1-terminal)*next_q
        bellman=torch.nn.functional.smooth_l1_loss(chosen,expected)
        conservative=(torch.logsumexp(qs,dim=-1)-chosen).mean()
        loss=bellman+cql_weight*conservative
        if not bool(torch.isfinite(loss).item()):raise RuntimeError('Nonfinite Q objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
        self.optimizer.step()
        self.updates+=1
        if self.updates%100==0:self.target.load_state_dict(self.net.state_dict())
        return dict(loss=float(loss.detach()),bellman=float(bellman.detach()),
                    conservative=float(conservative.detach()),updates=self.updates)

    def act(self,observation,goal,*,greedy=False,temperature=.2):
        if (not isinstance(observation,NaturalObservation) or
            observation.own.shape!=(len(TEMPORAL_V9_SELF_FEATURES),)):
            raise ValueError('Q skill requires genuine V9 bounded player observation')
        if goal!=CRACK_GOAL:
            if not isinstance(goal,str) or not goal.startswith('minecraft:'):
                raise ValueError('Inventory goals must be vanilla items')
            ObservedItemGoal(goal)
        if not .01<=temperature<=10:raise ValueError('Invalid action sampling temperature')
        example={'before_'+key:getattr(observation,key) for key in FIELDS}
        self.net.eval()
        with self.torch.no_grad():
            q=self.net(*self._batch([example],'before_',goal))
            action=(q.argmax(-1) if greedy else
                    self.torch.distributions.Categorical(logits=q/temperature).sample())
        return (int(action.item()),)

    def state(self):
        return dict(schema=SCHEMA,own_size=len(TEMPORAL_V9_SELF_FEATURES),
                    action_sizes=MOTOR_V7_HEAD_SIZES,net=self.net.state_dict(),
                    target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
                    updates=self.updates,torch_rng=self.torch.get_rng_state(),
                    cuda_rng=self.torch.cuda.get_rng_state_all()
                    if self.device.type=='cuda' else [])

    def restore(self,state,*,optimizer=True):
        if (state['schema']!=SCHEMA or state['own_size']!=len(TEMPORAL_V9_SELF_FEATURES)
                or tuple(state['action_sizes'])!=MOTOR_V7_HEAD_SIZES):
            raise ValueError('Incompatible replay Q checkpoint schema')
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state['target'])
        self.updates=int(state['updates'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.torch.set_rng_state(state['torch_rng'].cpu())
            if self.device.type=='cuda' and state['cuda_rng']:
                self.torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        return self
