"""Experimental bounded first-surface *block group* memory observation.

The Fabric extension casts a finite screen-space ray grid from the camera and
only reports the first collision per ray, grouped by actual block position.
No loaded-chunk enumeration, unseen world map or destination oracle. The
sampled projected coverage is NOT yet proof of actual rendered visibility.
This v1 research sensor is not connected to any learned policy yet.
Import BEFORE opening the first Minecraft observation packet.
"""
from dataclasses import dataclass
from math import isfinite

from mcio_ctrl.cbor import MCioType

SCHEMA = 'gmaincraft_surface_objects_v1'
ENV_NAME = 'GMAINCRAFT_SURFACE_OBJECTS_V1'
FACES = ('down','up','north','south','west','east')


@MCioType
@dataclass
class SurfaceObjectHit:
    block_id: str
    block_pos: list[int]
    face_mask: int
    nearest_distance: float
    sky_light: int
    block_light: int
    ray_indices: list[int]


@MCioType
@dataclass
class SurfaceObjectsOption:
    schema: str
    columns: int
    rows: int
    vertical_fov_degrees: float
    max_distance: float
    objects: list[SurfaceObjectHit]


@dataclass(frozen=True)
class ProjectedBlockV1:
    block_id: str
    xyz: tuple[int,int,int]
    visible_faces: tuple[str,...]
    nearest_distance: float
    sky_light: int
    block_light: int
    ray_indices: tuple[int,...]
    sample_fraction: float
    screen_center_uv: tuple[float,float]
    sample_bbox_uv: tuple[float,float,float,float]


@dataclass(frozen=True)
class SurfaceObjectsV1:
    schema: str
    columns: int
    rows: int
    vertical_fov_degrees: float
    max_distance: float
    objects: tuple[ProjectedBlockV1,...]

    @classmethod
    def from_packet(cls, packet):
        option=packet.get_option(SurfaceObjectsOption)
        if option is None:
            raise RuntimeError('Missing grouped MCio objects: install patched mod and set GMAINCRAFT_SURFACE_OBJECTS_V1=1')
        if option.schema!=SCHEMA:
            raise ValueError('Unsupported surface-object schema')
        cols,rows=option.columns,option.rows
        if (type(cols) is not int or type(rows) is not int or
                not 1<=cols<=31 or not 1<=rows<=31 or cols*rows>256):
            raise ValueError('Invalid surface sampling resolution')
        fov,limit=float(option.vertical_fov_degrees),float(option.max_distance)
        if not (isfinite(fov) and 1<=fov<180 and isfinite(limit) and 1<=limit<=512):
            raise ValueError('Invalid surface-object frustum')
        if not isinstance(option.objects,(list,tuple)) or len(option.objects)>cols*rows:
            raise ValueError('Too many surface objects')
        seen_positions,seen_rays=set(),set()
        objects=[]
        for raw in option.objects:
            if not isinstance(raw,SurfaceObjectHit):
                raise ValueError('Invalid surface object record type')
            if not isinstance(raw.block_id,str) or ':' not in raw.block_id or len(raw.block_id)>128:
                raise ValueError('Invalid surface block ID')
            pos=raw.block_pos
            if not isinstance(pos,(tuple,list)) or len(pos)!=3 or any(
                    type(x) is not int or abs(x)>30_000_000 for x in pos):
                raise ValueError('Invalid surface block position')
            key=tuple(pos)
            if key in seen_positions:
                raise ValueError('Duplicate block position')
            seen_positions.add(key)
            if type(raw.face_mask) is not int or not 1<=raw.face_mask<=63:
                raise ValueError('Invalid first-hit face mask')
            d=float(raw.nearest_distance)
            if not isfinite(d) or not 0<=d<=limit+.001:
                raise ValueError('Invalid visible-block distance')
            if (type(raw.sky_light) is not int or type(raw.block_light) is not int or
                    not 0<=raw.sky_light<=15 or not 0<=raw.block_light<=15):
                raise ValueError('Invalid surface illumination')
            indices=raw.ray_indices
            if not isinstance(indices,(tuple,list)) or not 1<=len(indices)<=cols*rows:
                raise ValueError('Empty or excessive ray coverage')
            for i in indices:
                if type(i) is not int or not 0<=i<cols*rows or i in seen_rays:
                    raise ValueError('Duplicate or out-of-range ray in grouped surface')
                seen_rays.add(i)
            us=[((i%cols)+.5)/cols for i in indices]
            vs=[((i//cols)+.5)/rows for i in indices]
            objects.append(ProjectedBlockV1(
                raw.block_id,key,
                tuple(name for bit,name in enumerate(FACES) if raw.face_mask & (1<<bit)),
                d,raw.sky_light,raw.block_light,tuple(indices),
                len(indices)/(cols*rows),
                (sum(us)/len(us),sum(vs)/len(vs)),
                (min(us),min(vs),max(us),max(vs))))
        return cls(SCHEMA,cols,rows,fov,limit,tuple(objects))

    def matching(self, block_id):
        return tuple(obj for obj in self.objects if obj.block_id==block_id)

    def assert_same_first_hits(self, grid):
        """Research-only audit: grouped records must match ungrouped rays.

        Fail fast instead of accidentally letting a regrouping bug invent
        a location or leak a surface absent from direct first-hit raycasts.
        """
        if (self.columns,self.rows)!=(grid.columns,grid.rows):
            raise AssertionError('Grouped rays and raw grid use different viewport samples')
        grouped={i:(obj.block_id,obj.xyz) for obj in self.objects for i in obj.ray_indices}
        original={ray.index:(ray.block_id,ray.xyz) for ray in grid.hits}
        if grouped!=original:
            raise AssertionError('Grouped ray objects differ from the raw first-hit evidence')
        rays={hit.index:hit for hit in grid.hits}
        for obj in self.objects:
            faces={rays[i].face for i in obj.ray_indices}
            nearest=min(rays[i].distance for i in obj.ray_indices)
            if set(obj.visible_faces)!=faces or abs(obj.nearest_distance-nearest)>1e-3:
                raise AssertionError('Grouped surface face/distance disagreement')
