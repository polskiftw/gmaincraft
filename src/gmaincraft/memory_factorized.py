"""Opt-in player-knowledge + self-rendered RGB factorized PPO (V4).

Read-only, player-knowable MCio data; no world seed, target vector, blocks,
privileged geometry or admin commands. Both training and frozen gameplay use
``MCioAgentBridge.multimodal_view()`` and the exact same feature encoder.
Old RGB-only V2/V3 checkpoint formats and policy code are untouched.
"""
from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from .player_knowledge import PlayerKnowledgeV1, SCHEMA as KNOWLEDGE_SCHEMA
from .vision_factorized import FactorizedPPOTrainer, validate_mask, HEAD_SIZES
from .vision_ppo import IMAGE_SIZE, HISTORY
from .ppo_math import timed_gae


FEATURE_NAMES = (
    'x_blocks_per_1024', 'y_blocks_per_128', 'z_blocks_per_1024',
    'x_fraction', 'z_fraction', 'yaw_sin', 'yaw_cos', 'pitch_per_90',
    'health_per_20', 'owned_logs_per_32', 'owned_planks_per_64',
    'owned_sticks_per_64', 'owned_total_items_per_128',
    'nonempty_main_slots_per_36', 'occupied_armor_slots_per_4',
    'occupied_offhand_slots', 'mined_logs_per_32', 'picked_up_logs_per_32',
    'mined_stats_available', 'picked_up_stats_available',
)
FEATURE_SIZE = len(FEATURE_NAMES)
SCHEMA = 'factorized_v4_memory_v1_rgb2_6tick'


def _wood_family(item_id):
    return item_id.startswith('minecraft:') and item_id.endswith(
        ('_log', '_wood', '_stem', '_hyphae'))


def encode_player_knowledge(player):
    """Stable explicit bounded featurization of known player state.

    Full original XYZ/owned slots/stats remain in ``PlayerKnowledgeV1``.
    The network sees deliberately scaled feature values without target hints.
    Missing statistics are marked *unknown*, not invented as zero events.
    """
    if not isinstance(player, PlayerKnowledgeV1) or player.schema != KNOWLEDGE_SCHEMA:
        raise ValueError('Expected player_knowledge_v1, no other policy schema')
    x,y,z = player.player_xyz
    counts = {key: 0 for key in ('log', 'plank', 'stick')}
    total = 0
    for slot in player.main_inventory + player.offhand_inventory:
        total += slot.count
        if _wood_family(slot.item_id): counts['log'] += slot.count
        elif slot.item_id.endswith('_planks'): counts['plank'] += slot.count
        elif slot.item_id == 'minecraft:stick': counts['stick'] += slot.count
    mined = picked = 0
    have_mined = have_picked = False
    for category, item_id, value in player.progress_stats:
        if category == 'minecraft:mined':
            have_mined = True
            if _wood_family(item_id): mined += value
        elif category == 'minecraft:picked_up':
            have_picked = True
            if _wood_family(item_id): picked += value
    bounded = lambda value, bound: float(np.clip(value, -bound, bound))
    result = np.asarray((
        bounded(x / 1024, 8), bounded(y / 128, 4), bounded(z / 1024, 8),
        x % 1, z % 1, sin(radians(player.yaw_degrees)),
        cos(radians(player.yaw_degrees)), bounded(player.pitch_degrees / 90, 1),
        bounded(player.health / 20, 2), min(counts['log'] / 32, 8),
        min(counts['plank'] / 64, 8), min(counts['stick'] / 64, 8),
        min(total / 128, 8), len(player.main_inventory) / 36,
        len(player.armor_inventory) / 4, len(player.offhand_inventory),
        min(mined / 32, 8), min(picked / 32, 8),
        float(have_mined), float(have_picked)
    ), dtype=np.float32)
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise ValueError('Invalid finite player feature vector')
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MemoryFactorTransition:
    image: np.ndarray
    features: np.ndarray
    action: tuple[int,int,int]
    reward: float
    value: float
    logp: float
    duration: int
    done: bool
    active_heads: tuple[bool,bool,bool] = (True,True,True)


def make_network():
    import torch
    from torch import nn
    from .vision_factorized import make_network as old_network

    class MemoryFactorizedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Preserve comparable raw RGB encoder as an optional perception
            # channel; structured player-owned state need not be OCR'd.
            self.visual = old_network().visual
            self.proprio = nn.Sequential(nn.Linear(FEATURE_SIZE,64),nn.ReLU(),
                                        nn.Linear(64,64),nn.ReLU())
            self.fuse = nn.Sequential(nn.Linear(256+64,256),nn.ReLU())
            self.movement = nn.Linear(256,HEAD_SIZES[0])
            self.camera = nn.Linear(256,HEAD_SIZES[1])
            self.attack = nn.Linear(256,HEAD_SIZES[2])
            self.critic = nn.Linear(256,1)

        def forward(self, image, features):
            if image.ndim != 4 or image.shape[1:] != (HISTORY*3,IMAGE_SIZE,IMAGE_SIZE):
                raise ValueError('Expected 2xRGB history of 84x84 pixels')
            if features.ndim != 2 or features.shape != (image.shape[0],FEATURE_SIZE):
                raise ValueError('Expected one player feature vector per image')
            x=self.fuse(torch.cat((self.visual(image.float()/255.),
                                   self.proprio(features.float())),dim=1))
            return (self.movement(x),self.camera(x),self.attack(x)),self.critic(x).squeeze(-1)

    return MemoryFactorizedCNN()


class MemoryFactorizedPPOTrainer(FactorizedPPOTrainer):
    observation_schema = SCHEMA
    """Separate random-weight checkpoint family; same PPO masks/timing as V2.

    Never loads an RGB-only checkpoint as compatible, and never silently
    drops the structured vector during frozen evaluation.
    """
    def __init__(self, *, device='auto', seed=4101, learning_rate=.00025,
                 clip_ratio=.2, vf_coef=.5, entropy_coef=.015,
                 batch_size=64, update_epochs=3):
        super().__init__(device=device, seed=seed, learning_rate=learning_rate,
                         clip_ratio=clip_ratio, vf_coef=vf_coef,
                         entropy_coef=entropy_coef, batch_size=batch_size,
                         update_epochs=update_epochs)
        # Seed again: the legacy network created by super() is discarded.
        self.torch.manual_seed(seed)
        self.net=make_network().to(self.device)
        self.optim=self.torch.optim.Adam(self.net.parameters(),lr=learning_rate,eps=1e-5)

    def _features(self, data):
        return self.torch.as_tensor(data,device=self.device,dtype=self.torch.float32)

    def act(self, image, features, *, greedy=False, active_heads=(True,True,True)):
        mask=validate_mask(active_heads)
        if image.shape != (6,IMAGE_SIZE,IMAGE_SIZE) or features.shape != (FEATURE_SIZE,):
            raise ValueError('Invalid memory+RGB policy observation')
        self.net.eval()
        with self.torch.no_grad():
            logits,value=self.net(self._images(image[np.newaxis,...]),
                                  self._features(features[np.newaxis,...]))
            distributions=[self.torch.distributions.Categorical(logits=x) for x in logits]
            choices=[(d.probs.argmax(-1) if greedy else d.sample()) if enabled else
                     self.torch.zeros(d.logits.shape[0],device=self.device,dtype=self.torch.long)
                     for d,enabled in zip(distributions,mask)]
            logp=sum(d.log_prob(choice) for d,choice,enabled in
                     zip(distributions,choices,mask) if enabled)
        return tuple(int(c.item()) for c in choices),float(logp.item()),float(value.item())

    def update(self, transitions, *, bootstrap_value=0.):
        torch=self.torch
        if not transitions: raise ValueError('No transitions to update')
        images=np.stack([t.image for t in transitions])
        features=np.stack([t.features for t in transitions])
        actions=torch.tensor([t.action for t in transitions],device=self.device,dtype=torch.long)
        oldlog=torch.tensor([t.logp for t in transitions],device=self.device,dtype=torch.float32)
        masks=torch.tensor([validate_mask(t.active_heads) for t in transitions],
                            device=self.device,dtype=torch.float32)
        adv,ret=timed_gae([t.reward for t in transitions],
                          [t.value for t in transitions],
                          [t.duration for t in transitions],
                          [t.done for t in transitions],last_value=bootstrap_value)
        adv=torch.as_tensor(adv,device=self.device)
        ret=torch.as_tensor(ret,device=self.device)
        if len(transitions)>1:
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        metrics=[]
        self.net.train()
        for _ in range(self.update_epochs):
            for selection in torch.randperm(len(transitions),device=self.device).split(self.batch_size):
                logits,values=self.net(self._images(images[selection.cpu().numpy()]),
                                       self._features(features[selection.cpu().numpy()]))
                ds=[torch.distributions.Categorical(logits=l) for l in logits]
                logp=sum(d.log_prob(actions[selection,i])*masks[selection,i]
                         for i,d in enumerate(ds))
                entropy=sum(d.entropy()*masks[selection,i] for i,d in enumerate(ds)).mean()
                ratio=torch.exp(logp-oldlog[selection])
                objective=torch.minimum(ratio*adv[selection],
                    torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*adv[selection])
                vloss=(values-ret[selection]).square().mean()
                loss=-objective.mean()+self.vf_coef*vloss-self.entropy_coef*entropy
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite PPO loss')
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),.5)
                self.optim.step()
                metrics.append((float(loss.detach()),float(vloss.detach()),float(entropy.detach())))
        self.updates+=1
        return dict(loss=float(np.mean([r[0] for r in metrics])),
                    value_loss=float(np.mean([r[1] for r in metrics])),
                    joint_entropy=float(np.mean([r[2] for r in metrics])),
                    transitions=len(transitions),device=str(self.device),updates=self.updates)
