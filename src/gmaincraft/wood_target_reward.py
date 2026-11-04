"""Versioned V5 reward from newly targeted, directly observed vanilla wood.

This is a TRAINING reward ledger, not a policy input or a target selector.
Only the player's live vanilla crosshair hit may earn a first-contact bonus.
No block search, privileged positions, scripted phases or repeatable farming.
"""
from .mcio_status import CrosshairBlock
from .natural_online_policy import wood_id


class WoodTargetFirstContact:
    """A fixed per-world memory cap; every unique crosshair-hit wood pays once."""
    def __init__(self, *, max_blocks=4096, bonus=.9):
        if type(max_blocks) is not int or max_blocks < 1 or type(bonus) not in (int,float) or not 0 < bonus <= 10:
            raise ValueError('Invalid bounded wood-target reward settings')
        self.seen = set()
        self.max_blocks = max_blocks
        self.bonus = float(bonus)

    def observe(self, crosshair):
        if crosshair is None:return 0.
        if not isinstance(crosshair,CrosshairBlock):
            raise TypeError('Wood targeting requires the actual typed vanilla crosshair block')
        if not wood_id(crosshair.block_id):return 0.
        key=(tuple(crosshair.xyz),crosshair.block_id)
        if key in self.seen or len(self.seen)>=self.max_blocks:return 0.
        self.seen.add(key)
        return self.bonus


class WoodEngagement:
    """V6: separately pay first visible wood contact and first vanilla attack.

    Each block has at most one contact AND one attack reward per Minecraft
    world. Input is only the live crosshair and the actual vanilla attack key;
    this ledger never decides actions or provides a target to the model.
    """
    def __init__(self, *, max_blocks=4096):
        if type(max_blocks) is not int or max_blocks < 1:
            raise ValueError('Invalid bounded wood engagement memory cap')
        self.targets = set()
        self.attacked = set()
        self.max_blocks = max_blocks

    def observe(self, crosshair, *, attack):
        if type(attack) is not bool:raise TypeError('Attack state must be bool')
        if crosshair is None:return 0.,0.
        if not isinstance(crosshair,CrosshairBlock):
            raise TypeError('Wood engagement requires actual vanilla crosshair block')
        if not wood_id(crosshair.block_id):return 0.,0.
        key=(tuple(crosshair.xyz),crosshair.block_id)
        contact=0.
        if key not in self.targets:
            if len(self.targets)>=self.max_blocks:return 0.,0.
            self.targets.add(key)
            contact=.6
        first_attack=0.
        if attack and key not in self.attacked:
            self.attacked.add(key)
            first_attack=1.8
        return contact,first_attack


class WoodSustainedAttack:
    """V8: bounded reward for the player's own sustained wood-targeted attack.

    Only a genuine vanilla visible crosshair and a stationary ordinary
    attack command qualify. One block can earn at most 12 six-tick samples,
    even across repeated targeting or animation reset. No world scans,
    path, aim correction, hidden material, or policy action intervention.
    """
    def __init__(self, *, max_blocks=4096, max_samples=12, bonus=.75):
        if (type(max_blocks) is not int or max_blocks<1 or type(max_samples) is not int
            or not 1<=max_samples<=50 or type(bonus) not in (int,float)
            or not 0<float(bonus)<=10):
            raise ValueError('Invalid bounded wood persistence reward')
        self.counts={}
        self.max_blocks=max_blocks
        self.max_samples=max_samples
        self.bonus=float(bonus)

    def observe(self,crosshair,*,stationary_attack):
        if type(stationary_attack) is not bool:
            raise TypeError('Stationary attack flag must be bool')
        if crosshair is None:return 0.
        if not isinstance(crosshair,CrosshairBlock):
            raise TypeError('Sustained wood attack requires live typed vanilla crosshair')
        if not stationary_attack or not wood_id(crosshair.block_id):return 0.
        key=(tuple(crosshair.xyz),crosshair.block_id)
        if key not in self.counts and len(self.counts)>=self.max_blocks:return 0.
        n=self.counts.get(key,0)
        if n>=self.max_samples:return 0.
        self.counts[key]=n+1
        return self.bonus
