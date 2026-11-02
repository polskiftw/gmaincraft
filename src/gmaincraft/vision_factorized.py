"""Factorized movement/look/attack policy with short Minecraft action intervals.

V2 preserves all V1 weights, scripts and checkpoints. Independent categorical
heads control each motor channel simultaneously; joint log probabilities and
entropies train with entire-trajectory PPO. Actions contain NO hidden state,
controller commands or teacher labels. The 6-tick action interval avoids the
old 80-tick no-observation mining macro. Consecutive attack selections keep
LEFT_BUTTON held through InputStateManager, allowing real block breaking.
"""
from dataclasses import dataclass
import random

import numpy as np

from .agent_interface import AgentAction
from .ppo_math import timed_gae
from .vision_ppo import PPOTrainer, IMAGE_SIZE, HISTORY

ACTION_TICKS = 6
MOVE_KEYS = ((), ('W',), ('S',), ('A',), ('D',), ('W', 'SPACE'))
LOOK_DELTAS = ((0.,0.), (-60.,0.), (60.,0.), (0.,-60.), (0.,60.))
HEAD_SIZES = (len(MOVE_KEYS), len(LOOK_DELTAS), 2)
SCHEMA = 'factorized_v2_move6_look5_attack2_interval6'


def action_for_factors(factors, remaining_ticks=None):
    if (not isinstance(factors, (tuple, list)) or len(factors) != 3 or
            any(not isinstance(x, int) or isinstance(x, bool) for x in factors)):
        raise ValueError('Expected (movement, camera, attack) integer triple')
    m, l, attack = factors
    if not (0 <= m < HEAD_SIZES[0] and 0 <= l < HEAD_SIZES[1] and attack in (0,1)):
        raise ValueError('Invalid factorized action indices')
    if remaining_ticks is not None and (not isinstance(remaining_ticks,int) or remaining_ticks < 1):
        raise ValueError('Remaining game ticks must be positive')
    ticks = min(ACTION_TICKS, remaining_ticks) if remaining_ticks is not None else ACTION_TICKS
    return AgentAction(keys=MOVE_KEYS[m] + (('LEFT_BUTTON',) if attack else ()),
                       look_dx=LOOK_DELTAS[l][0], look_dy=LOOK_DELTAS[l][1],
                       ticks=ticks)


@dataclass
class FactorTransition:
    image: np.ndarray
    action: tuple[int, int, int]
    reward: float
    value: float
    logp: float
    duration: int
    done: bool
    active_heads: tuple[bool, bool, bool] = (True, True, True)


def validate_mask(mask):
    if (not isinstance(mask, (tuple,list)) or len(mask)!=3 or
            any(type(x) is not bool for x in mask) or not any(mask)):
        raise ValueError('Head mask must be a three-bool tuple with at least one active head')
    return tuple(mask)


def make_network():
    import torch
    from torch import nn

    class FactorizedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(6,32,kernel_size=8,stride=4),nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2),nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),nn.ReLU(),
                nn.Flatten(),nn.Linear(64*7*7,256),nn.ReLU())
            self.movement = nn.Linear(256,HEAD_SIZES[0])
            self.camera = nn.Linear(256,HEAD_SIZES[1])
            self.attack = nn.Linear(256,HEAD_SIZES[2])
            self.critic = nn.Linear(256,1)

        def forward(self, rgb):
            if rgb.ndim != 4 or rgb.shape[1:] != (6,IMAGE_SIZE,IMAGE_SIZE):
                raise ValueError('Invalid RGB frame-history shape')
            x = self.visual(rgb.float()/255.)
            return (self.movement(x),self.camera(x),self.attack(x)),self.critic(x).squeeze(-1)
    return FactorizedCNN()


class FactorizedPPOTrainer(PPOTrainer):
    """Reuses reliable atomic checkpoint logic of V1; NOT weight compatible."""
    def __init__(self, *, device='auto', seed=4101, learning_rate=.00025,
                 clip_ratio=.2, vf_coef=.5, entropy_coef=.015,
                 batch_size=64, update_epochs=3):
        import torch
        self.torch = torch
        if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda'):
            raise ValueError('Device must be cpu, cuda or auto')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; run tools/gpu-preflight.py')
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.net = make_network().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(),lr=learning_rate,eps=1e-5)
        self.clip_ratio,self.vf_coef,self.entropy_coef = clip_ratio,vf_coef,entropy_coef
        self.batch_size,self.update_epochs = batch_size,update_epochs
        self.rng = random.Random(seed+1)
        self.updates = 0
        if device == 'cuda': torch.backends.cudnn.benchmark = True

    def act(self, image, *, greedy=False, active_heads=(True,True,True)):
        active_heads=validate_mask(active_heads)
        torch=self.torch
        if image.shape != (HISTORY*3,IMAGE_SIZE,IMAGE_SIZE):
            raise ValueError('Expected 2xRGB 84x84 image')
        self.net.eval()
        with torch.no_grad():
            logits,value=self.net(self._images(image[np.newaxis,...]))
            distributions=[torch.distributions.Categorical(logits=x) for x in logits]
            choices=[(x.probs.argmax(-1) if greedy else x.sample()) if active else
                     torch.zeros(x.logits.shape[0],device=x.logits.device,dtype=torch.long)
                     for x,active in zip(distributions,active_heads)]
            joint_logp=sum(d.log_prob(choice) for d,choice,active in
                           zip(distributions,choices,active_heads) if active)
        return tuple(int(x.item()) for x in choices),float(joint_logp.item()),float(value.item())

    def update(self, transitions, *, bootstrap_value=0.):
        torch=self.torch
        if not transitions: raise ValueError('No transitions to update')
        images=np.stack([t.image for t in transitions])
        actions=torch.tensor([t.action for t in transitions],device=self.device,dtype=torch.long)
        oldlog=torch.tensor([t.logp for t in transitions],device=self.device,dtype=torch.float32)
        head_masks=torch.tensor([validate_mask(t.active_heads) for t in transitions],
                                device=self.device,dtype=torch.float32)
        advantages,returns=timed_gae(
            [t.reward for t in transitions],[t.value for t in transitions],
            [t.duration for t in transitions],[t.done for t in transitions],
            last_value=bootstrap_value)
        advantages=torch.as_tensor(advantages,device=self.device)
        returns=torch.as_tensor(returns,device=self.device)
        if len(transitions)>1:
            advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        self.net.train()
        metrics=[]
        for _ in range(self.update_epochs):
            for select in torch.randperm(len(transitions),device=self.device).split(self.batch_size):
                logits,values=self.net(self._images(images[select.cpu().numpy()]))
                distributions=[torch.distributions.Categorical(logits=x) for x in logits]
                logp=sum(d.log_prob(actions[select,i])*head_masks[select,i]
                         for i,d in enumerate(distributions))
                entropy=sum(d.entropy()*head_masks[select,i]
                            for i,d in enumerate(distributions)).mean()
                ratio=torch.exp(logp-oldlog[select])
                objective=torch.minimum(ratio*advantages[select],
                    torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*advantages[select])
                value_loss=(values-returns[select]).square().mean()
                loss=-objective.mean()+self.vf_coef*value_loss-self.entropy_coef*entropy
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite PPO loss')
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),.5)
                self.optim.step()
                metrics.append((float(loss.detach()),float(value_loss.detach()),float(entropy.detach())))
        self.updates+=1
        return dict(loss=float(np.mean([r[0] for r in metrics])),
                    value_loss=float(np.mean([r[1] for r in metrics])),
                    joint_entropy=float(np.mean([r[2] for r in metrics])),
                    transitions=len(transitions),device=str(self.device),updates=self.updates)
