"""Opt-in MCio Fabric player status and *actual* vanilla crosshair block hit.

Requires the project-patched MCio 1.6.1 mod and process environment
GMAINCRAFT_STATUS_V1=1. Both are scoped to the disposable DemoInstance.

Import this module BEFORE receiving the first MCio observation to register
PlayerStatusOption with mcio_ctrl.cbor. No raw chunk/world state is exposed.
"""
from dataclasses import dataclass
from math import isfinite

from mcio_ctrl.cbor import MCioType

SCHEMA = 'gmaincraft_status_v1'
ENV_NAME = 'GMAINCRAFT_STATUS_V1'


@MCioType
@dataclass
class PlayerStatusOption:
    """Exact new Java record shape; CBOR decorator registers `.PlayerStatusOption`."""
    schema: str
    hunger: int
    saturation: float
    armor_points: int
    air_supply: int
    selected_hotbar_slot: int
    dimension: str
    targeted_block_id: str | None
    targeted_block_pos: list[int] | None
    targeted_block_face: str | None
    targeted_hit_distance: float


@dataclass(frozen=True)
class CrosshairBlock:
    block_id: str
    xyz: tuple[int, int, int]
    face: str
    distance: float


@dataclass(frozen=True)
class PlayerStatusV1:
    schema: str
    hunger: int
    saturation: float
    armor_points: int
    air_supply: int
    selected_hotbar_slot: int
    dimension: str
    crosshair_block: CrosshairBlock | None

    @classmethod
    def from_packet(cls, packet):
        opt=packet.get_option(PlayerStatusOption)
        if opt is None:
            raise RuntimeError('MCio did not emit PlayerStatusOption; install patched mod and set GMAINCRAFT_STATUS_V1=1')
        if opt.schema!=SCHEMA:
            raise ValueError('Unknown MCio status schema')
        hunger=int(opt.hunger)
        saturation=float(opt.saturation)
        armor=int(opt.armor_points)
        air=int(opt.air_supply)
        selected=int(opt.selected_hotbar_slot)
        if not (0<=hunger<=20 and 0<=saturation<=100 and
                0<=armor<=1000 and -1000<=air<=100000 and 0<=selected<=8 and
                isfinite(saturation)):
            raise ValueError('Invalid direct player status values')
        dimension=str(opt.dimension)
        if not (':' in dimension and len(dimension)<=128):
            raise ValueError('Invalid current dimension ID')
        block=None
        target=opt.targeted_block_id
        if target is not None:
            if not isinstance(target,str) or ':' not in target or len(target)>128:
                raise ValueError('Invalid targeted block identifier')
            pos=opt.targeted_block_pos
            if not isinstance(pos,(tuple,list)) or len(pos)!=3 or any(
                    type(v) is not int or abs(v)>30_000_000 for v in pos):
                raise ValueError('Invalid targeted block position')
            face=opt.targeted_block_face
            if face not in ('up','down','north','south','east','west'):
                raise ValueError('Invalid targeted block face')
            distance=float(opt.targeted_hit_distance)
            if not isfinite(distance) or not 0<=distance<=128:
                raise ValueError('Invalid targeted block distance')
            block=CrosshairBlock(target,tuple(pos),face,distance)
        elif (opt.targeted_block_pos is not None or opt.targeted_block_face is not None or
              float(opt.targeted_hit_distance)!=-1):
            raise ValueError('Non-block hit must not expose target coordinates')
        return cls(SCHEMA,hunger,saturation,armor,air,selected,dimension,block)
