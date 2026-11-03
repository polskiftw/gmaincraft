"""Experimental read-only first-hit camera rays from patched MCio/Fabric.

This v0 grid is a research instrument, NOT a validated visibility selector,
object catalog, or learner input. Import before the first CBOR observation.
The game computes first COLLIDER hits on 13x9 camera rays, never a chunk dump.
"""
from dataclasses import dataclass
from math import isfinite

from mcio_ctrl.cbor import MCioType

SCHEMA = 'gmaincraft_surface_grid_v0'
ENV_NAME = 'GMAINCRAFT_SURFACE_GRID_V0'


@MCioType
@dataclass
class SurfaceRayHit:
    ray_index: int
    block_id: str
    block_pos: list[int]
    face: str
    distance: float
    sky_light: int
    block_light: int


@MCioType
@dataclass
class SurfaceGridOption:
    schema: str
    columns: int
    rows: int
    vertical_fov_degrees: float
    max_distance: float
    hits: list[SurfaceRayHit]


@dataclass(frozen=True)
class VisibleRayV0:
    index: int
    normalized_uv: tuple[float, float]
    block_id: str
    xyz: tuple[int, int, int]
    face: str
    distance: float
    sky_light: int
    block_light: int


@dataclass(frozen=True)
class SurfaceGridV0:
    schema: str
    columns: int
    rows: int
    vertical_fov_degrees: float
    max_distance: float
    hits: tuple[VisibleRayV0, ...]

    @classmethod
    def from_packet(cls, packet):
        option = packet.get_option(SurfaceGridOption)
        if option is None:
            raise RuntimeError('MCio surface grid missing: requires patched mod and GMAINCRAFT_SURFACE_GRID_V0=1')
        if option.schema != SCHEMA:
            raise ValueError('Unknown surface ray observation schema')
        cols, rows = option.columns, option.rows
        if (type(cols) is not int or type(rows) is not int or
                not (1 <= cols <= 31 and 1 <= rows <= 31) or cols*rows > 256):
            raise ValueError('Invalid sample grid shape')
        fov, limit = float(option.vertical_fov_degrees), float(option.max_distance)
        if not (isfinite(fov) and 1 <= fov < 180 and
                isfinite(limit) and 1 <= limit <= 512):
            raise ValueError('Invalid sample frustum')
        if not isinstance(option.hits, (tuple, list)) or len(option.hits) > cols*rows:
            raise ValueError('Too many surface hits')
        found, hits = set(), []
        for raw in option.hits:
            if not isinstance(raw, SurfaceRayHit):
                raise ValueError('Incorrect surface-hit record')
            i = raw.ray_index
            if type(i) is not int or not 0 <= i < cols*rows or i in found:
                raise ValueError('Duplicate or out-of-range surface ray index')
            found.add(i)
            if not isinstance(raw.block_id,str) or ':' not in raw.block_id or len(raw.block_id) > 128:
                raise ValueError('Invalid first-hit block identifier')
            p = raw.block_pos
            if not isinstance(p, (tuple,list)) or len(p) != 3 or any(
                    type(v) is not int or abs(v) > 30_000_000 for v in p):
                raise ValueError('Invalid surface block position')
            if raw.face not in ('up','down','north','south','east','west'):
                raise ValueError('Invalid sampled surface face')
            d = float(raw.distance)
            if not isfinite(d) or not 0 <= d <= limit + .001:
                raise ValueError('Invalid ray hit distance')
            if (type(raw.sky_light) is not int or type(raw.block_light) is not int or
                    not 0 <= raw.sky_light <= 15 or not 0 <= raw.block_light <= 15):
                raise ValueError('Invalid sampled surface light')
            uv = ((i % cols + .5) / cols, (i // cols + .5) / rows)
            hits.append(VisibleRayV0(i,uv,raw.block_id,tuple(p),raw.face,d,
                                     raw.sky_light,raw.block_light))
        return cls(SCHEMA,cols,rows,fov,limit,tuple(sorted(hits,key=lambda hit: hit.index)))

    def block_hits(self, block_id):
        """Return actual per-ray hits, NOT deduplicated 3D surface visibility."""
        return tuple(hit for hit in self.hits if hit.block_id == block_id)
