"""From-random-weights RGB convolutional actor-critic + trajectory PPO.

PyTorch GPU is an OPTIONAL local dependency (requirements-gpu.txt). The old
NumPy toy models still run without it. At training time this module sees only
policy pixel stacks/actions/rewards, never privileged coordinates or NBT.

This prototype uses short RGB frame history, not recurrent long-term memory.
Its temporal credit assignment is GAE over entire trajectories, unlike old
one-action REINFORCE. The initial action list includes an 80-tick mine macro:
that is a transparent motor primitive, NOT autonomously learned hold duration.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import random

import numpy as np

from .agent_interface import AgentAction
from .ppo_math import timed_gae

IMAGE_SIZE = 84
HISTORY = 2
ACTION_NAMES = ('forward', 'back', 'strafe_left', 'strafe_right',
                'look_left', 'look_right', 'look_up', 'look_down',
                'mine_hold_80', 'jump_forward', 'wait')
ACTION_COUNT = len(ACTION_NAMES)


def action_for_index(index, remaining_ticks=None):
    table = (
        AgentAction(keys=('W',), ticks=8),
        AgentAction(keys=('S',), ticks=8),
        AgentAction(keys=('A',), ticks=8),
        AgentAction(keys=('D',), ticks=8),
        AgentAction(look_dx=-75, ticks=2),
        AgentAction(look_dx=75, ticks=2),
        AgentAction(look_dy=-75, ticks=2),
        AgentAction(look_dy=75, ticks=2),
        AgentAction(keys=('LEFT_BUTTON',), ticks=80),
        AgentAction(keys=('W', 'SPACE'), ticks=8),
        AgentAction(ticks=5),
    )
    if not isinstance(index, int) or index < 0 or index >= len(table):
        raise ValueError('Invalid joint-action index')
    action = table[index]
    if remaining_ticks is None or remaining_ticks >= action.ticks:
        return action
    if remaining_ticks < 1:
        raise ValueError('No ticks left in episode')
    return AgentAction(action.keys, action.look_dx, action.look_dy, remaining_ticks)


class PixelHistory:
    """Two resized full-RGB frames, uint8 and no target-color heuristic."""
    def __init__(self):
        self.previous = deque(maxlen=HISTORY)

    @staticmethod
    def _resize(frame):
        from PIL import Image
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected raw HxWx3 uint8 RGB frame')
        image = Image.fromarray(frame, mode='RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.uint8).transpose(2, 0, 1).copy()

    def reset(self, view):
        f = self._resize(view.pixels)
        self.previous.clear()
        for _ in range(HISTORY):
            self.previous.append(f.copy())
        return self.state()

    def append(self, view):
        self.previous.append(self._resize(view.pixels))
        return self.state()

    def state(self):
        if len(self.previous) != HISTORY:
            raise RuntimeError('PixelHistory requires reset before policy inference')
        return np.concatenate(tuple(self.previous), axis=0)


@dataclass
class Transition:
    image: np.ndarray
    action: int
    reward: float
    value: float
    logp: float
    duration: int
    done: bool


def make_network():
    """Avoid importing optional torch in the base NumPy-only CI workflow."""
    import torch
    from torch import nn

    class CNNActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(HISTORY*3, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64*7*7, 256), nn.ReLU())
            self.actor = nn.Linear(256, ACTION_COUNT)
            self.critic = nn.Linear(256, 1)

        def forward(self, rgb):
            if rgb.ndim != 4 or rgb.shape[1:] != (HISTORY*3, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError('Invalid RGB history shape')
            x = self.visual(rgb.float() / 255.0)
            return self.actor(x), self.critic(x).squeeze(-1)

    return CNNActorCritic()


class PPOTrainer:
    def __init__(self, *, device='auto', seed=4101, learning_rate=0.00025,
                 clip_ratio=.2, vf_coef=.5, entropy_coef=.01, batch_size=64,
                 update_epochs=3):
        import torch
        self.torch = torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be auto, cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; check nvidia_uvm and /dev/nvidia-uvm')
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.net = make_network().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-5)
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.rng = random.Random(seed+1)
        self.updates = 0
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True  # fixed 84x84 network inputs

    def _images(self, np_images):
        return self.torch.from_numpy(np_images).to(device=self.device,
                                                   dtype=self.torch.uint8)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.net.parameters())

    def act(self, image, *, greedy=False):
        """Exactly one selected action plus frozen logprob/value for PPO."""
        import torch
        if image.shape != (HISTORY*3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError('Expected CHW uint8 two-RGB-frame history')
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(self._images(image[np.newaxis, ...]))
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def evaluate_value(self, image):
        return self.act(image, greedy=True)[2]

    def update(self, transitions, *, bootstrap_value=0.0):
        import torch
        if not transitions:
            raise ValueError('No transitions to update')
        images = np.stack([t.image for t in transitions])
        a = torch.tensor([t.action for t in transitions], device=self.device, dtype=torch.long)
        oldlog = torch.tensor([t.logp for t in transitions], device=self.device, dtype=torch.float32)
        advantages, returns = timed_gae(
            [t.reward for t in transitions], [t.value for t in transitions],
            [t.duration for t in transitions], [t.done for t in transitions],
            last_value=bootstrap_value)
        advantages = torch.as_tensor(advantages, device=self.device)
        returns = torch.as_tensor(returns, device=self.device)
        # Stable PPO scaling, retain zero-variance advantages unchanged.
        if len(transitions)>1:
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        n = len(transitions)
        self.net.train()
        metrics = []
        for epoch in range(self.update_epochs):
            order = torch.randperm(n, device=self.device)
            for select in order.split(self.batch_size):
                logits, values = self.net(self._images(images[select.cpu().numpy()]))
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a[select])
                ratio = torch.exp(logp-oldlog[select])
                objective = torch.minimum(ratio*advantages[select],
                    torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*advantages[select])
                value_loss = (values-returns[select]).square().mean()
                entropy = dist.entropy().mean()
                loss = -objective.mean() + self.vf_coef*value_loss - self.entropy_coef*entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('PPO produced nonfinite loss')
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=.5)
                self.optim.step()
                metrics.append((float(loss.detach()), float(value_loss.detach()),
                                float(entropy.detach())))
        self.updates += 1
        return dict(loss=float(np.mean([x[0] for x in metrics])),
                    value_loss=float(np.mean([x[1] for x in metrics])),
                    entropy=float(np.mean([x[2] for x in metrics])),
                    transitions=n, device=str(self.device), updates=self.updates)

    def save(self, path, metadata):
        """Atomically replace LOCAL checkpoint after a full PPO update.

        Keep optimizer moments as well as weights for an eventual resumable
        trainer; a partial/failed disk write never replaces a valid file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(model={k:v.detach().cpu() for k,v in self.net.state_dict().items()},
                       optimizer=self.optim.state_dict(),
                       updates=self.updates, metadata=dict(metadata))
        fd, temp_path = tempfile.mkstemp(prefix=target.stem+'.', suffix='.tmp',dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as handle:
                self.torch.save(payload,handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path,target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_weights(self, path, *, restore_optimizer=False):
        checkpoint = self.torch.load(path, map_location='cpu', weights_only=True)
        self.net.load_state_dict(checkpoint['model'])
        self.updates = int(checkpoint.get('updates', 0))
        if restore_optimizer and 'optimizer' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer'])
            # The optimizer was saved from an arbitrary device; move its
            # momentum tensors to the active training device on warm resume.
            for state in self.optim.state.values():
                for key, value in state.items():
                    if self.torch.is_tensor(value):
                        state[key] = value.to(self.device)
        return checkpoint.get('metadata', {})
