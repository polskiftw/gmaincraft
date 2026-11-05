"""Explicit versioned tensor adapter for future memory-based item pickup.

This module is intentionally NOT used by the existing V9/Q wood policy,
which must retain exact old observation shape and checkpoint semantics.
It uses ONLY the already visibility-filtered same-camera dropped items
and the player's own XYZ; it never scans loaded entities/chunks.
"""
from dataclasses import dataclass

import numpy as np

from .mcio_visible_items import VisibleItemsV1
from .natural_online_policy import ID_BYTES,encode_id

SCHEMA='visible_item_relative_encoder_v1'
MAX_ITEMS=8
FEATURES=('relative_dx_div_24','relative_dy_div_24','relative_dz_div_24',
          'screen_u_2_minus_1','screen_v_2_minus_1','distance_div_24',
          'verified_visible_sample_fraction')


@dataclass(frozen=True)
class EncodedVisibleItems:
    schema:str
    ids:np.ndarray
    data:np.ndarray
    mask:np.ndarray


def encode_visible_items(visible,player_xyz):
    """Bounded perspective-limited item objects for a FUTURE skill branch."""
    if not isinstance(visible,VisibleItemsV1):
        raise TypeError('Only typed camera-verified items are eligible')
    if (not isinstance(player_xyz,(tuple,list)) or len(player_xyz)!=3 or
            not all(isinstance(v,(int,float)) and np.isfinite(v) for v in player_xyz)):
        raise ValueError('Expected finite own-player XYZ')
    ids=np.zeros((MAX_ITEMS,ID_BYTES),dtype=np.uint8)
    data=np.zeros((MAX_ITEMS,len(FEATURES)),dtype=np.float32)
    mask=np.zeros(MAX_ITEMS,dtype=np.float32)
    for i,item in enumerate(visible.items[:MAX_ITEMS]):
        u,v=item.screen_uv
        delta=[(item.xyz[j]-player_xyz[j])/24 for j in range(3)]
        values=[*delta,2*u-1,2*v-1,item.distance/24,item.visible_samples/5]
        if not all(np.isfinite(a) for a in values):
            raise ValueError('Nonfinite visible item features')
        data[i]=np.clip(values,-2.,2.)
        ids[i]=encode_id(item.item_id)
        mask[i]=1
    for arr in ids,data,mask:
        arr.setflags(write=False)
    return EncodedVisibleItems(SCHEMA,ids,data,mask)
