"""Pure natural-seed evaluation plans and frozen diagnostic policies.

No hidden-world information is used by these policies. LegacyProbe is an
honestly labelled scripted handoff of the OLD frozen arena-trained subskills,
not a trained general Minecraft agent or a claim of autonomous speedrunning.
"""
from dataclasses import dataclass
import random

import numpy as np

from .agent_interface import AgentAction
from .log_vision import VisualAimPolicy, visual_features
from .log_pickup import PickupPolicy, PICKUP_ACTIONS, pickup_features

MODES = ('idle', 'random', 'forward_attack', 'legacy_probe', 'joint_ppo', 'joint_factorized', 'memory_factorized', 'status_factorized')


@dataclass(frozen=True)
class NaturalRun:
    seed: int
    source: str
    world: str
    mode: str


def plan_runs(tag, seeds, modes):
    if not (tag and len(tag) <= 24 and tag[0].isalnum() and
            all(c.isascii() and (c.isalnum() or c in '-_') for c in tag)):
        raise ValueError('Invalid tag (1-24 ASCII letters/digits/_/-)')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be a nonempty unique list')
    if not modes or len(modes) != len(set(modes)) or any(m not in MODES for m in modes):
        raise ValueError('Modes must be a nonempty unique subset of supported modes')
    if any(not isinstance(s, int) or isinstance(s, bool) or not -(2**63) <= s < 2**63
           for s in seeds):
        raise ValueError('Minecraft seeds must be signed 64-bit integers')
    return [NaturalRun(seed, f'MDL_NaturalSource_{tag}_{i:02d}',
                       f'MDL_NaturalEval_{tag}_{i:02d}_{mode}', mode)
            for i, seed in enumerate(seeds, 1) for mode in modes]


class IdleAgent:
    def act(self, view, remaining):
        return AgentAction(ticks=min(100, remaining))


class ForwardAttackAgent:
    def act(self, view, remaining):
        return AgentAction(keys=('W', 'LEFT_BUTTON'), ticks=min(25, remaining))


class RandomAgent:
    """A random ordinary-key baseline, not a sophisticated explorer."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, view, remaining):
        if self.rng.random() < .22:
            return AgentAction(keys=('LEFT_BUTTON',), ticks=min(55, remaining))
        direction = self.rng.choice(('W', 'A', 'S', 'D'))
        return AgentAction(keys=(direction,),
                           look_dx=self.rng.choice((-90, 0, 90)),
                           look_dy=self.rng.choice((-45, 0, 45)),
                           ticks=min(22, remaining))


class LegacyProbe:
    """Repeated frozen aim -> 100-tick attack -> frozen pickup.

    These were *separately* learned in a placed-trunk arena. The handoff and
    100-tick hold are SCRIPTED. No checkpoint updates, exploration or planner.
    """
    def __init__(self, aim_weights, pickup_weights):
        self.aim = VisualAimPolicy(seed=0)
        self.pickup = PickupPolicy(seed=0)
        if aim_weights.shape != self.aim.weights.shape or not np.isfinite(aim_weights).all():
            raise ValueError('Invalid aim checkpoint')
        if pickup_weights.shape != self.pickup.weights.shape or not np.isfinite(pickup_weights).all():
            raise ValueError('Invalid pickup checkpoint')
        self.aim.weights = aim_weights.copy()
        self.pickup.weights = pickup_weights.copy()
        self.phase = 'aim'
        self.aim_decisions = 0
        self.pickup_decisions = 0

    def act(self, view, remaining):
        if self.phase == 'aim':
            action, _ = self.aim.choose(visual_features(view.pixels), random.Random(0), greedy=True)
            self.aim_decisions += 1
            if action == 4:
                self.phase = 'pickup'
                self.pickup_decisions = 0
                return AgentAction(keys=('LEFT_BUTTON',), ticks=min(100, remaining))
            if self.aim_decisions >= 18:
                # Original trial would time out without an attack. A live-world
                # diagnostic may restart its look cycle; it does not gain help.
                self.aim_decisions = 0
            dx = (-45 if action == 0 else 45 if action == 1 else 0)
            dy = (-45 if action == 2 else 45 if action == 3 else 0)
            return AgentAction(look_dx=dx, look_dy=dy, ticks=min(2, remaining))
        action, _ = self.pickup.choose(pickup_features(view.pixels), random.Random(0), greedy=True)
        self.pickup_decisions += 1
        if self.pickup_decisions >= 22:
            self.phase = 'aim'
            self.aim_decisions = 0
        return AgentAction(keys=(PICKUP_ACTIONS[action],), ticks=min(5, remaining))


class JointPPOAgent:
    """Frozen joint pixel-only PPO checkpoint for natural-world transfer.

    The screenshot sequence is the ONLY input. The pre-trained checkpoint is
    evaluated without training or added controller assistance.
    """
    def __init__(self, checkpoint, device='auto'):
        from .vision_ppo import PPOTrainer, PixelHistory, action_for_index
        self.trainer = PPOTrainer(device=device, seed=0)
        self.metadata = self.trainer.load_weights(checkpoint)
        self.history = PixelHistory()
        self.started = False
        self.action_for_index = action_for_index

    def act(self, view, remaining):
        image = self.history.append(view) if self.started else self.history.reset(view)
        self.started = True
        action, _, _ = self.trainer.act(image, greedy=True)
        return self.action_for_index(action, remaining)


class FactorizedPPOAgent:
    """Pure pixel-only frozen V2 actor in naturally generated survival worlds.

    A separate CNN checkpoint from the old 11-action macro policy. It uses
    no privileged target position, trainer mask or reward in evaluation.
    """
    def __init__(self, checkpoint, device='auto'):
        from .vision_factorized import FactorizedPPOTrainer,action_schema,action_for_factors
        from .vision_ppo import PixelHistory
        self.trainer=FactorizedPPOTrainer(device=device,seed=0)
        metadata=self.trainer.load_weights(checkpoint)
        self.decision_ticks=int(metadata.get('decision_ticks',6))
        if metadata.get('schema')!=action_schema(self.decision_ticks):
            raise ValueError('Invalid factorized policy schema')
        self.history=PixelHistory()
        self.started=False
        self.action_for_factors=action_for_factors

    def act(self, view, remaining):
        frame=(self.history.append(view) if self.started else self.history.reset(view))
        self.started=True
        choices,_,_=self.trainer.act(frame,greedy=True)
        return self.action_for_factors(choices,remaining,decision_ticks=self.decision_ticks)


class MemoryFactorizedAgent:
    """Frozen V4 from-scratch actor with exactly the TRAINED memory schema.

    No trainer target geometry, administrative resets or hidden world fields.
    The same view and feature pipeline powers both train and gameplay.
    """
    def __init__(self, checkpoint, device='auto'):
        import torch
        from .memory_factorized import (MemoryFactorizedPPOTrainer, SCHEMA,
                                        FEATURE_NAMES, encode_player_knowledge)
        from .vision_factorized import action_for_factors
        from .vision_ppo import PixelHistory
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        metadata=cp.get('metadata',{})
        if metadata.get('schema')!=SCHEMA or tuple(metadata.get('feature_names',()))!=FEATURE_NAMES:
            raise ValueError('Incompatible memory observation or network checkpoint schema')
        self.trainer=MemoryFactorizedPPOTrainer(device=device,seed=0)
        self.trainer.load_weights(checkpoint)
        self.history=PixelHistory()
        self.started=False
        self.encode_player_knowledge=encode_player_knowledge
        self.action_for_factors=action_for_factors
        self.decision_ticks=6

    def act(self, view, remaining):
        from .agent_interface import MultimodalView
        if not isinstance(view,MultimodalView):
            raise ValueError('Memory policy requires bounded combined observation')
        frame=self.history.append(view) if self.started else self.history.reset(view)
        self.started=True
        values=self.encode_player_knowledge(view.player)
        choices,_,_=self.trainer.act(frame,values,greedy=True)
        return self.action_for_factors(choices,remaining,decision_ticks=self.decision_ticks)


class StatusFactorizedAgent:
    """Frozen V5: same opt-in Minecraft crosshair+status, owned memory, RGB.

    The only world object is what the player's actual vanilla crosshair hit;
    trainer geometry, hidden seed, nearby unobserved blocks and any controller
    commands are NEVER policy observations.
    """
    def __init__(self,checkpoint,device='auto'):
        import torch
        from .status_factorized import (StatusFactorizedPPOTrainer,SCHEMA,
            FEATURE_NAMES,BLOCK_TOKEN_LENGTH,BLOCK_TOKENIZER,encode_status_knowledge)
        from .vision_factorized import action_for_factors
        from .vision_ppo import PixelHistory
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        metadata=cp.get('metadata',{})
        if metadata.get('schema')!=SCHEMA or tuple(metadata.get('feature_names',()))!=FEATURE_NAMES or                 metadata.get('status_schema')!='gmaincraft_status_v1' or                 metadata.get('block_tokenizer')!=BLOCK_TOKENIZER or                 int(metadata.get('block_token_length',-1))!=BLOCK_TOKEN_LENGTH:
            raise ValueError('Incompatible V5 status/feature/tokenizer checkpoint schema')
        self.trainer=StatusFactorizedPPOTrainer(device=device,seed=0)
        self.trainer.load_weights(checkpoint)
        self.history=PixelHistory()
        self.started=False
        self.encode_status_knowledge=encode_status_knowledge
        self.action_for_factors=action_for_factors
        self.decision_ticks=6

    def act(self,view,remaining):
        from .agent_interface import MultimodalStatusView
        if not isinstance(view,MultimodalStatusView):
            raise ValueError('V5 gameplay requires its own opt-in status+RGB observation')
        image=self.history.append(view) if self.started else self.history.reset(view)
        self.started=True
        numeric,tokens=self.encode_status_knowledge(view.player,view.status)
        choices,_,_=self.trainer.act(image,numeric,tokens,greedy=True)
        return self.action_for_factors(choices,remaining,decision_ticks=self.decision_ticks)


def make_agent(mode, action_seed, *, aim_weights=None, pickup_weights=None,
               joint_checkpoint=None, factorized_checkpoint=None,
               memory_checkpoint=None, status_checkpoint=None, device='auto'):
    if mode == 'idle':
        return IdleAgent()
    if mode == 'forward_attack':
        return ForwardAttackAgent()
    if mode == 'random':
        return RandomAgent(action_seed)
    if mode == 'legacy_probe':
        if aim_weights is None or pickup_weights is None:
            raise ValueError('Legacy probe requires two frozen checkpoints')
        return LegacyProbe(aim_weights, pickup_weights)
    if mode == 'joint_ppo':
        if joint_checkpoint is None:
            raise ValueError('Joint PPO mode requires a trained local checkpoint')
        return JointPPOAgent(joint_checkpoint, device=device)
    if mode == 'status_factorized':
        if status_checkpoint is None:
            raise ValueError('Status factorized mode requires a trained V5 checkpoint')
        return StatusFactorizedAgent(status_checkpoint,device=device)
    if mode == 'memory_factorized':
        if memory_checkpoint is None:
            raise ValueError('Memory factorized mode requires a trained local checkpoint')
        return MemoryFactorizedAgent(memory_checkpoint,device=device)
    if mode == 'joint_factorized':
        if factorized_checkpoint is None:
            raise ValueError('Factorized mode requires a frozen checkpoint')
        return FactorizedPPOAgent(factorized_checkpoint, device=device)
    raise ValueError(f'Unknown mode {mode}')
