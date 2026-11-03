"""Nonrepeating, mixed-parity Minecraft seeds with provably disjoint holdouts.

A keyed SplitMix64 finalizer (two invertible odd multiplications, XOR-right
shifts and modular addition) maps interleaved training and evaluation
COUNTERS to mixed-parity, signed 64-bit seeds without collisions.
Train inputs 0,2,4,... and eval inputs 1,3,5,... cannot collide, but both
outputs include both even and odd seeds. Up to 2**63 unique seeds each.
The model NEVER observes the generator key, counter, or current seed.
"""
from dataclasses import dataclass
from hashlib import sha256

MASK = (1 << 64) - 1
CAPACITY = 1 << 63
SCHEMA = 'minecraft_seed_stream_v2_full64_interleaved_train_eval'


@dataclass(frozen=True)
class SeedSchedule:
    root: int
    namespace: str = 'train'

    def __post_init__(self):
        if type(self.root) is not int or not -(1<<63) <= self.root < (1<<63):
            raise ValueError('seed root must be a signed 64-bit integer')
        if self.namespace not in ('train','eval'):
            raise ValueError('Seed namespace must be train or eval')

    def seed_at(self, index: int) -> int:
        if type(index) is not int or index < 0 or index >= CAPACITY:
            raise ValueError('Seed cursor out of bounds')
        digest=sha256(('gmaincraft:'+SCHEMA+':'+str(self.root)).encode('ascii')).digest()
        offset=int.from_bytes(digest[:8],'big') & MASK
        counter=(index<<1) | int(self.namespace=='eval')
        bits=(counter+offset) & MASK
        bits=((bits ^ (bits>>30)) * 0xbf58476d1ce4e5b9) & MASK
        bits=((bits ^ (bits>>27)) * 0x94d049bb133111eb) & MASK
        bits^=bits>>31
        return bits if bits < 1<<63 else bits-(1<<64)

    def validate_seed(self,index,seed):
        return type(seed) is int and seed==self.seed_at(index)
