"""V2 self-play ONLY rare-outcome learning from the actual visible target.

The practice outcome 'visible wood crack progress' uses two already-existing
V9 player-visible features: crosshair_is_wood and a positive known crack rise.
No unseen target location, teacher action, seed, or privileged mining stat.
The anchor policy is an older policy learned by THIS AGENT on Minecraft,
not a human model. It regularizes non-target behavior to avoid catastrophic
forgetting. This is a practice policy, not a solved Minecraft skill.
"""
from copy import deepcopy

import numpy as np

from .experience_archive import FIELDS
from .natural_online_policy import NaturalObservation
from .natural_online_policy import TEMPORAL_V9_SELF_FEATURES
from .replay_q_skill import ReplayQSkill,SCHEMA as LEGACY_SCHEMA,CRACK_GOAL,goal_transitions

SCHEMA='player_known_rare_outcome_conservative_q_v2'
WOOD_CRACK_GOAL='effect:visible_wood_crack_progress_v2'
WOOD_INDEX=TEMPORAL_V9_SELF_FEATURES.index('crosshair_is_wood')


def wood_crack_transitions(meta,data):
    """Label only genuinely observed same-block crack increments on wood.

    All inputs are prior policy-recorded own sensations; do not relabel
    unrelated crack progress. Training-only labels never enter the actor.
    """
    records=goal_transitions(meta,data,CRACK_GOAL)
    for i,row in enumerate(records):
        row['achieved']=bool(row['achieved'] and row['after_own'][WOOD_INDEX]>.5)
        row['done']=bool(data['done'][i]) or row['achieved'] or i==len(records)-1
        row['priority']=20. if row['achieved'] else 1.
    for i,row in enumerate(records):
        if not row['achieved']:continue
        for back in range(1,13):
            j=i-back
            if j<0 or records[j]['done']:break
            records[j]['priority']=max(records[j]['priority'],4.)
    return records


class RareOutcomeQSkill(ReplayQSkill):
    """V2 network: preserve original non-target behavior, boost own successes.

    A frozen BASELINE learned from real self-play regularizes the policy in
    states where wood isn't in the crosshair. On real positive target states,
    advantage-weighted supervised action likelihood trains FROM OWN ACTIONS,
    alongside conservative double-Q and game-tick cost. Every gradient is
    driven by the actual sampled vanilla action and subsequent observation.
    """
    def __init__(self,*,seed=20261104,device='auto',lr=.00008):
        super().__init__(seed=seed,device=device,lr=lr)
        self.anchor=deepcopy(self.net).to(self.device).eval()
        self.anchor.requires_grad_(False)

    def initialize_from_selfplay(self,checkpoint):
        if checkpoint.get('schema')!=LEGACY_SCHEMA:
            raise ValueError('V2 anchor requires an original player-trained Q model')
        meta=checkpoint.get('metadata',{})
        if meta.get('goal')!=CRACK_GOAL:
            raise ValueError('V2 needs only the original genuine visible crack goal')
        source=checkpoint['model']
        if source.get('schema')!=LEGACY_SCHEMA:
            raise ValueError('V2 source state contract mismatch')
        self.net.load_state_dict(source['net'])
        self.target.load_state_dict(source['target'])
        self.anchor.load_state_dict(source['net'])
        self.updates=0  # Separate new optimizer and learning history.
        return self

    def update(self,batch,goal,*,cql_weight=.03,success_weight=.6,anchor_weight=.2):
        if (not batch or goal!=WOOD_CRACK_GOAL or not 0<=cql_weight<=1 or
            not 0<success_weight<=10 or not 0<=anchor_weight<=10):
            raise ValueError('Invalid versioned rare-outcome optimizer configuration')
        torch=self.torch
        b=self._batch(batch,'before_',goal)
        a=self._batch(batch,'after_',goal)
        actions=torch.tensor([r['action'] for r in batch],device=self.device,dtype=torch.long)
        achieved=torch.tensor([float(r['achieved']) for r in batch],device=self.device)
        durations=torch.tensor([r['duration'] for r in batch],device=self.device,dtype=torch.float32)
        terminal=torch.tensor([r['done'] for r in batch],device=self.device,dtype=torch.float32)
        self.net.train()
        qs=self.net(*b)
        chosen=qs.gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            best=self.net(*a).argmax(-1)
            future=self.target(*a).gather(1,best[:,None]).squeeze(1)
            expected=achieved-.001*(durations/6)+(.997**(durations/6))*(1-terminal)*future
        td=torch.nn.functional.smooth_l1_loss(chosen,expected)
        cql=(torch.logsumexp(qs,dim=-1)-chosen).mean()
        positive=achieved>.5
        # Only genuine success actions qualify for imitation. Never create
        # 'attack on wood' commands as labels, and do not treat an unknown
        # crack stage as zero or infer any unobserved block damage.
        if bool(positive.any().item()):
            success=torch.nn.functional.cross_entropy(qs[positive]/.4,actions[positive])
        else:success=qs.sum()*0
        outside=b[0][:,WOOD_INDEX]<.5
        if bool(outside.any().item()):
            with torch.no_grad():ref=self.anchor(*b)[outside]/.4
            current=torch.log_softmax(qs[outside]/.4,dim=-1)
            preserve=torch.nn.functional.kl_div(current,torch.softmax(ref,dim=-1),reduction='batchmean')
        else:preserve=qs.sum()*0
        loss=.5*td+cql_weight*cql+success_weight*success+anchor_weight*preserve
        if not bool(torch.isfinite(loss).item()):raise RuntimeError('Nonfinite rare-outcome loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),.8)
        self.optimizer.step()
        self.updates+=1
        if self.updates%100==0:self.target.load_state_dict(self.net.state_dict())
        return dict(updates=self.updates,loss=float(loss.detach()),td=float(td.detach()),
                    positive_own_actions=int(positive.sum().item()),
                    positive_imitation=float(success.detach()),anchor_kl=float(preserve.detach()))

    def act(self,observation,goal,*,greedy=False,temperature=.2):
        if goal!=WOOD_CRACK_GOAL:raise ValueError('Wrong v2 training/practice goal')
        if (not isinstance(observation,NaturalObservation) or
            observation.own.shape!=(len(TEMPORAL_V9_SELF_FEATURES),) or
            not .01<=temperature<=10):
            raise ValueError('V2 actor needs exact V9 own observation and temperature')
        example={'before_'+field:getattr(observation,field) for field in FIELDS}
        self.net.eval()
        with self.torch.no_grad():
            q=self.net(*self._batch([example],'before_',goal))
            choice=(q.argmax(-1) if greedy else
                self.torch.distributions.Categorical(logits=q/temperature).sample())
        return (int(choice.item()),)


    def state(self):
        data=super().state()
        data.update(schema=SCHEMA,anchor=self.anchor.state_dict(),
                    goal=WOOD_CRACK_GOAL,learning='own_success_imitation+conservative_q+nonwood_anchor')
        return data

    def restore(self,state,*,optimizer=True):
        if (state.get('schema')!=SCHEMA or state.get('goal')!=WOOD_CRACK_GOAL or
            state.get('learning')!='own_success_imitation+conservative_q+nonwood_anchor'):
            raise ValueError('Incompatible V2 Q model schema and behavior contract')
        baseline=dict(state,schema=LEGACY_SCHEMA)
        super().restore(baseline,optimizer=optimizer)
        self.anchor.load_state_dict(state['anchor'])
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        return self
