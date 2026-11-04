"""Experimental opt-in *perceivable* dropped-item entity sensor (NOT policy input).

Minecraft's client filters candidate ItemEntity objects by camera frustum,
at least one unobstructed VISUAL ray to its tiny bounding box and minimum
projected size. This is research instrumentation, not proof of actual pixel
recognition: transparency, sprites and lighting need staged calibration.
Never expose all loaded entities, their count, hidden boxes or stacks.
Imported before the first patched MCio observation is decoded.
"""
from dataclasses import dataclass
from math import isfinite

from mcio_ctrl.cbor import MCioType

SCHEMA='gmaincraft_visible_items_v1'
ENV_NAME='GMAINCRAFT_VISIBLE_ITEMS_V1'
ON_DEMAND_ENV='GMAINCRAFT_VISIBLE_ITEMS_ON_DEMAND_V1'
MAX_ITEMS=16
MAX_SAMPLES=5


@MCioType
@dataclass
class VisibleItemsRequest:
    fresh:bool=True


@MCioType
@dataclass
class VisibleItemHit:
    item_id:str
    position:list[float]
    distance:float
    screen_u:float
    screen_v:float
    visible_samples:int


@MCioType
@dataclass
class VisibleItemsOption:
    schema:str
    max_distance:float
    items:list[VisibleItemHit]


@dataclass(frozen=True)
class VisibleDroppedItemV1:
    item_id:str
    xyz:tuple[float,float,float]
    distance:float
    screen_uv:tuple[float,float]
    visible_samples:int


@dataclass(frozen=True)
class VisibleItemsV1:
    schema:str
    max_distance:float
    items:tuple[VisibleDroppedItemV1,...]

    @classmethod
    def from_packet(cls,packet):
        option=packet.get_option(VisibleItemsOption)
        if option is None:
            raise RuntimeError('Missing same-action visible-items option; enable patched client sensor and explicitly request it')
        if not isinstance(option,VisibleItemsOption) or option.schema!=SCHEMA:
            raise ValueError('Invalid visible-item observation schema')
        distance=float(option.max_distance)
        if not isfinite(distance) or not 1<=distance<=24:
            raise ValueError('Invalid visible-item sensor reach')
        records=option.items
        if not isinstance(records,(list,tuple)) or len(records)>MAX_ITEMS:
            raise ValueError('Invalid number of player-visible dropped items')
        parsed=[]
        last_distance=-1.
        for hit in records:
            if not isinstance(hit,VisibleItemHit):
                raise ValueError('Malformed visible item record')
            name=hit.item_id
            if (not isinstance(name,str) or len(name)>128 or
                not name.startswith('minecraft:') or
                any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_/:.-' for c in name)):
                raise ValueError('Invalid item registry ID')
            coords=hit.position
            if (not isinstance(coords,(tuple,list)) or len(coords)!=3 or
                any(not isinstance(x,(float,int)) or isinstance(x,bool) or
                    not isfinite(x) or abs(x)>30000000 for x in coords)):
                raise ValueError('Invalid item position')
            d=float(hit.distance)
            if not isfinite(d) or d<=0 or d>distance+.001 or d<last_distance:
                raise ValueError('Invalid item range or out-of-order nearest-first records')
            last_distance=d
            u,v=float(hit.screen_u),float(hit.screen_v)
            # Object center may be just outside view while one corner is in.
            if not all(isfinite(x) and -0.15<=x<=1.15 for x in (u,v)):
                raise ValueError('Malformed projected position')
            if type(hit.visible_samples) is not int or not 1<=hit.visible_samples<=MAX_SAMPLES:
                raise ValueError('Malformed visible sample evidence')
            parsed.append(VisibleDroppedItemV1(
                name,tuple(float(x) for x in coords),d,(u,v),hit.visible_samples))
        return cls(SCHEMA,distance,tuple(parsed))
