"""From-scratch goal-conditioned policy trained only from self-achieved items.

Hindsight labels say 'this normal sequence happened before I actually
acquired item X', not 'perform a human recipe'. No pretrained model,
external demonstrations, scripted handoffs or privileged block locations.
Positive samples alone will NOT establish a capable, robust option; held-out
real Minecraft evaluation is required before calling anything a skill.
"""
import numpy as np

from .experience_archive import FIELDS
from .natural_online_policy import ID_BYTES,MAX_OBJECTS,OBJECT_FEATURES,encode_id

SCHEMA='self_observed_goal_behavior_cloning_v1'


class HindsightSkillModel:
    def __init__(self,*,own_size,action_sizes,seed=20261103,device='auto',lr=.0003):
        import torch
        from torch import nn
        if (type(own_size) is not int or not 1<=own_size<=512 or
            not isinstance(action_sizes,tuple) or not action_sizes or
            any(type(x) is not int or not 1<=x<=256 for x in action_sizes) or
            not 0<lr<=.01):
            raise ValueError('Invalid learned-goal observation/action configuration')
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cuda','cpu') or device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested learner accelerator unavailable')
        self.device=torch.device(device)
        self.torch=torch
        self.own_size=own_size
        self.action_sizes=action_sizes
        torch.manual_seed(seed)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.chars=nn.Embedding(257,16,padding_idx=0)
                self.object=nn.Sequential(nn.Linear(16+len(OBJECT_FEATURES),64),
                                          nn.Tanh(),nn.Linear(64,48),nn.Tanh())
                self.shared=nn.Sequential(nn.Linear(own_size+16+16+16+48,128),
                                         nn.Tanh(),nn.Linear(128,128),nn.Tanh())
                self.heads=nn.ModuleList(nn.Linear(128,n) for n in action_sizes)

            def token(self,values):
                embedded=self.chars(values.long())
                mask=(values!=0).unsqueeze(-1)
                return (embedded*mask).sum(dim=-2)/mask.sum(dim=-2).clamp(min=1)

            def forward(self,own,crosshair,ids,data,mask,goal):
                objects=self.object(self.torch_concat(self.token(ids),data.float()))
                weights=mask.float().unsqueeze(-1)
                pooled=(objects*weights).sum(dim=1)/weights.sum(dim=1).clamp(min=1)
                x=self.shared(torch.cat((own.float(),self.token(crosshair),
                        self.token(goal),self.token(ids).mean(dim=1),pooled),dim=-1))
                return [head(x) for head in self.heads]

            @staticmethod
            def torch_concat(a,b):
                return torch.cat((a,b),dim=-1)

        self.net=Network().to(self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.updates=0

    def state(self):
        return dict(schema=SCHEMA,own_size=self.own_size,action_sizes=self.action_sizes,
                    model=self.net.state_dict(),optimizer=self.optimizer.state_dict(),
                    updates=self.updates,torch_rng=self.torch.get_rng_state(),
                    cuda_rng=self.torch.cuda.get_rng_state_all()
                    if self.device.type=='cuda' else [])

    def restore(self,state,*,optimizer=True):
        if state['schema']!=SCHEMA or state['own_size']!=self.own_size or                 tuple(state['action_sizes'])!=self.action_sizes:
            raise ValueError('Incompatible learned-goal skill checkpoint')
        self.net.load_state_dict(state['model'])
        self.updates=int(state['updates'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.torch.set_rng_state(state['torch_rng'].cpu())
            if self.device.type=='cuda' and state['cuda_rng']:
                self.torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])

    def batch(self,examples):
        torch=self.torch
        out=[]
        for field in FIELDS:
            vals=np.stack([ex['before_'+field] for ex in examples])
            out.append(torch.as_tensor(np.array(vals,copy=True),device=self.device))
        out.append(torch.as_tensor(np.stack([encode_id(ex['goal_item_id']) for ex in examples]),
                                   device=self.device))
        return out

    def train_examples(self,examples,*,batch_size=64,seed=1):
        if not examples:raise ValueError('No genuine self-achieved goals found')
        if type(batch_size) is not int or not 1<=batch_size<=1024:
            raise ValueError('Invalid learned-goal batch')
        torch=self.torch
        rng=np.random.default_rng(seed)
        order=rng.permutation(len(examples))
        self.net.train()
        losses=[]
        for indexes in np.array_split(order,max(1,int(np.ceil(len(order)/batch_size)))):
            batch=[examples[int(i)] for i in indexes]
            logits=self.net(*self.batch(batch))
            choices=torch.tensor([ex['action'] for ex in batch],device=self.device,dtype=torch.long)
            weights=torch.tensor([1./(1.+ex['ticks_to_achievement']/180.)
                                  for ex in batch],device=self.device)
            loss=sum((torch.nn.functional.cross_entropy(head,choices[:,j],reduction='none')
                      *weights).sum()/weights.sum().clamp(min=1e-6)
                     for j,head in enumerate(logits))
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('Nonfinite skill-learning loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
            self.optimizer.step()
            self.updates+=1
            losses.append(float(loss.detach()))
        return dict(updates=self.updates,examples=len(examples),
                    mean_weighted_loss=round(float(np.mean(losses)),6))

    def act(self,observation,goal_item_id,*,greedy=True):
        from .natural_online_policy import NaturalObservation
        if not isinstance(observation,NaturalObservation):
            raise TypeError('Invalid player-known observation')
        if observation.own.shape!=(self.own_size,):
            raise ValueError('Incompatible player observation size')
        if not isinstance(goal_item_id,str) or ':' not in goal_item_id:
            raise ValueError('Goal must be previously player-observable item ID')
        example={'before_'+k:getattr(observation,k) for k in FIELDS}
        example['goal_item_id']=goal_item_id
        self.net.eval()
        with self.torch.no_grad():
            logits=self.net(*self.batch([example]))
            distributions=[self.torch.distributions.Categorical(logits=x) for x in logits]
            choices=[d.probs.argmax(-1) if greedy else d.sample() for d in distributions]
        return tuple(int(c.item()) for c in choices)
