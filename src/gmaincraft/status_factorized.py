"""V5 from-scratch actor: own Minecraft state + actual crosshair target + RGB.

The opt-in patched MCio/Fabric engine (not screen OCR) provides the block
currently hit by Minecraft's vanilla crosshair, plus player hunger/armor/air.
This exposes NO unloaded/occluded block list and no privileged goal. The same
snapshot and encoder must be used during real training and frozen gameplay.

Block IDs are encoded from their FULL UTF-8 bytes with a randomly initialized
character convolution (no hand-coded tree vocabulary or pretrained models).
The V5 checkpoint schema is intentionally incompatible with V4 and earlier.
"""
from dataclasses import dataclass
import numpy as np

from .mcio_status import PlayerStatusV1,SCHEMA as STATUS_SCHEMA
from .memory_factorized import (MemoryFactorizedPPOTrainer,FEATURE_NAMES as V4_FEATURES,
                                encode_player_knowledge)
from .vision_factorized import HEAD_SIZES,validate_mask
from .vision_ppo import HISTORY,IMAGE_SIZE
from .ppo_math import timed_gae

STATUS_FEATURE_NAMES=(
    'hunger_per_20','saturation_per_20','armor_per_20','air_per_300',
    'hotbar_slot_per_8',
    'dimension_overworld','dimension_nether','dimension_end','dimension_other',
    'crosshair_has_block','crosshair_rel_x_per_8','crosshair_rel_y_per_8',
    'crosshair_rel_z_per_8','crosshair_distance_per_8',
    'crosshair_face_up','crosshair_face_down','crosshair_face_north',
    'crosshair_face_south','crosshair_face_east','crosshair_face_west',
)
FEATURE_NAMES=V4_FEATURES+STATUS_FEATURE_NAMES
FEATURE_SIZE=len(FEATURE_NAMES)
BLOCK_TOKEN_LENGTH=128
BLOCK_TOKENIZER='utf8_bytes_plus_one_v1'
SCHEMA='factorized_v5_status_v1_player_v1_rgb2_6tick'
FACE_NAMES=('up','down','north','south','east','west')
DIM_NAMES=('minecraft:overworld','minecraft:the_nether','minecraft:the_end')


def encode_status_knowledge(player,status):
    """Return bounded numeric and full-id byte tokens, independent of RGB.

    Only the actual vanilla client-side hit is accepted from
    ``PlayerStatusV1``. For a MISS, target coordinates, face and tokens are
    strictly zero. World seed, unobserved chunks and trainer geometry are
    never queried. A player-visible target can be distant from a goal tree.
    """
    if not isinstance(status,PlayerStatusV1) or status.schema!=STATUS_SCHEMA:
        raise ValueError('Expected gmaincraft_status_v1 from patched MCio')
    own=encode_player_knowledge(player)
    target=status.crosshair_block
    dim=status.dimension
    face=target.face if target is not None else None
    relative=tuple(float(np.clip((v-p)/8,-1,1)) for v,p in
                   zip(target.xyz,player.player_xyz)) if target else (0.,0.,0.)
    tail=(
        float(np.clip(status.hunger/20,0,1)),
        float(np.clip(status.saturation/20,0,5)),
        float(np.clip(status.armor_points/20,0,10)),
        float(np.clip(status.air_supply/300,-4,20)),
        status.selected_hotbar_slot/8.,
        *(float(dim==name) for name in DIM_NAMES),
        float(dim not in DIM_NAMES),
        float(target is not None),
        *relative,
        float(np.clip(target.distance/8,0,16)) if target else 0.,
        *(float(face==name) for name in FACE_NAMES),
    )
    features=np.concatenate((own,np.asarray(tail,dtype=np.float32)))
    tokens=np.zeros(BLOCK_TOKEN_LENGTH,dtype=np.int16)
    if target:
        block_id_bytes=target.block_id.encode('utf-8')
        if len(block_id_bytes)>BLOCK_TOKEN_LENGTH:
            raise ValueError('Block identifier exceeds V5 tokenizer budget')
        tokens[:len(block_id_bytes)]=np.frombuffer(block_id_bytes,dtype=np.uint8).astype(np.int16)+1
    if features.shape!=(FEATURE_SIZE,) or not np.isfinite(features).all():
        raise ValueError('Invalid V5 player+target feature shape/values')
    features.setflags(write=False)
    tokens.setflags(write=False)
    return features,tokens


@dataclass(frozen=True)
class StatusFactorTransition:
    image: np.ndarray
    features: np.ndarray
    block_tokens: np.ndarray
    action: tuple[int,int,int]
    reward: float
    value: float
    logp: float
    duration: int
    done: bool
    active_heads: tuple[bool,bool,bool]=(True,True,True)


def make_network():
    import torch
    from torch import nn
    from .vision_factorized import make_network as old_network

    class StatusFactorizedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual=old_network().visual
            self.proprio=nn.Sequential(nn.Linear(FEATURE_SIZE,64),nn.ReLU(),
                                      nn.Linear(64,64),nn.ReLU())
            self.block_chars=nn.Embedding(257,16,padding_idx=0)
            self.block_conv=nn.Sequential(nn.Conv1d(16,32,kernel_size=3,padding=1),nn.ReLU(),
                                          nn.Conv1d(32,32,kernel_size=3,padding=1),nn.ReLU())
            self.fuse=nn.Sequential(nn.Linear(256+64+32,256),nn.ReLU())
            self.movement=nn.Linear(256,HEAD_SIZES[0])
            self.camera=nn.Linear(256,HEAD_SIZES[1])
            self.attack=nn.Linear(256,HEAD_SIZES[2])
            self.critic=nn.Linear(256,1)

        def forward(self,image,features,block_tokens):
            if image.ndim!=4 or tuple(image.shape[1:])!=(HISTORY*3,IMAGE_SIZE,IMAGE_SIZE):
                raise ValueError('V5 expects two RGB history frames, 84x84')
            if features.ndim!=2 or tuple(features.shape)!=(image.shape[0],FEATURE_SIZE):
                raise ValueError('Wrong V5 player+status numeric vector')
            if block_tokens.ndim!=2 or tuple(block_tokens.shape)!=(image.shape[0],BLOCK_TOKEN_LENGTH):
                raise ValueError('Wrong V5 full block-ID token sequence')
            encoded=self.block_chars(block_tokens.long()).transpose(1,2)
            encoded=self.block_conv(encoded)
            mask=block_tokens.ne(0).unsqueeze(1)
            block=(encoded*mask).sum(-1)/mask.sum(-1).clamp(min=1)
            x=self.fuse(torch.cat((self.visual(image.float()/255.),
                                   self.proprio(features.float()),block),dim=1))
            return (self.movement(x),self.camera(x),self.attack(x)),self.critic(x).squeeze(-1)

    return StatusFactorizedCNN()


class StatusFactorizedPPOTrainer(MemoryFactorizedPPOTrainer):
    """Independent CUDA PPO for V5, never silently compatible with V4."""
    observation_schema=SCHEMA

    def __init__(self,*,device='auto',seed=4101,learning_rate=.00025,
                 clip_ratio=.2,vf_coef=.5,entropy_coef=.015,
                 batch_size=64,update_epochs=3):
        super().__init__(device=device,seed=seed,learning_rate=learning_rate,
                         clip_ratio=clip_ratio,vf_coef=vf_coef,
                         entropy_coef=entropy_coef,batch_size=batch_size,
                         update_epochs=update_epochs)
        self.torch.manual_seed(seed)
        self.net=make_network().to(self.device)
        self.optim=self.torch.optim.Adam(self.net.parameters(),lr=learning_rate,eps=1e-5)

    def _features(self,values):
        # Policy snapshots are immutable; PyTorch rejects read-only NumPy
        # storage even for inference. Use an explicit defensive copy.
        return self.torch.as_tensor(np.array(values,copy=True,dtype=np.float32),
                                    device=self.device,dtype=self.torch.float32)

    def _tokens(self,values):
        return self.torch.as_tensor(np.array(values,copy=True,dtype=np.int64),
                                    device=self.device,dtype=self.torch.long)

    def act(self,image,features,block_tokens,*,greedy=False,active_heads=(True,True,True)):
        mask=validate_mask(active_heads)
        if image.shape!=(6,IMAGE_SIZE,IMAGE_SIZE) or features.shape!=(FEATURE_SIZE,) or \
                block_tokens.shape!=(BLOCK_TOKEN_LENGTH,):
            raise ValueError('V5 requires memory+status+block ID+RGB policy input')
        self.net.eval()
        with self.torch.no_grad():
            logits,value=self.net(self._images(image[np.newaxis,...]),
                                  self._features(features[np.newaxis,...]),
                                  self._tokens(block_tokens[np.newaxis,...]))
            ds=[self.torch.distributions.Categorical(logits=x) for x in logits]
            choices=[(d.probs.argmax(-1) if greedy else d.sample()) if enabled else
                     self.torch.zeros(d.logits.shape[0],device=self.device,dtype=self.torch.long)
                     for d,enabled in zip(ds,mask)]
            logp=sum(d.log_prob(choice) for d,choice,enabled in
                     zip(ds,choices,mask) if enabled)
        return tuple(int(c.item()) for c in choices),float(logp.item()),float(value.item())

    def update(self,transitions,*,bootstrap_value=0.):
        torch=self.torch
        if not transitions:raise ValueError('No V5 transitions to update')
        images=np.stack([t.image for t in transitions])
        features=np.stack([t.features for t in transitions])
        tokens=np.stack([t.block_tokens for t in transitions])
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
                ids=selection.cpu().numpy()
                logits,values=self.net(self._images(images[ids]),
                                       self._features(features[ids]),self._tokens(tokens[ids]))
                ds=[torch.distributions.Categorical(logits=l) for l in logits]
                logp=sum(d.log_prob(actions[selection,i])*masks[selection,i]
                         for i,d in enumerate(ds))
                entropy=sum(d.entropy()*masks[selection,i] for i,d in enumerate(ds)).mean()
                ratio=torch.exp(logp-oldlog[selection])
                objective=torch.minimum(ratio*adv[selection],
                    torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*adv[selection])
                vloss=(values-ret[selection]).square().mean()
                loss=-objective.mean()+self.vf_coef*vloss-self.entropy_coef*entropy
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite V5 PPO loss')
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
