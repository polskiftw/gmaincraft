"""Versioned, self-collected, game-observation-only episode archive.

Records exactly what the normal player policy knew before/after an ordinary
input action, together with its *own* before/after inventory. No privileged
world scans, hidden structure locations, teacher goals, or Minecraft seed
appear in the learned-transition interface. Seeds may be in external
experiment journals, never in observations or goals.

Compressed episode files are atomic, append-only, never overwritten;
train-run checkpoint cursors govern whether episodes are committed.
"""
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import numpy as np

from .natural_online_policy import NaturalObservation
from .player_knowledge import PlayerKnowledgeV1

SCHEMA='player_known_episode_archive_v1'
FILENAME=re.compile(r'^episode_([0-9]{12})\.npz$')
FIELDS=('own','crosshair_id','surface_ids','surface_data','surface_mask')
MAX_RECORDS=100000  # bounded by the existing max six-tick episode budget
MAX_ITEM_ID=128


def owned_inventory(player):
    """Only player-owned slots; nonzero counts by item ID, independent of slot."""
    if not isinstance(player,PlayerKnowledgeV1):
        raise TypeError('Inventory must come from player-owned Minecraft knowledge')
    totals=Counter()
    for slot in player.main_inventory+player.armor_inventory+player.offhand_inventory:
        if slot.count:
            if not ':' in slot.item_id or len(slot.item_id)>MAX_ITEM_ID or slot.count<0:
                raise ValueError('Invalid player-owned item')
            totals[slot.item_id]+=slot.count
    return dict(sorted(totals.items()))


@dataclass(frozen=True)
class ObservedItemGoal:
    """An achieved increase of a player-owned item, NOT a predefined recipe."""
    item_id: str
    increase: int = 1

    def __post_init__(self):
        if (not isinstance(self.item_id,str) or ':' not in self.item_id or
            len(self.item_id)>MAX_ITEM_ID or type(self.increase) is not int or
            not 1<=self.increase<=2304):
            raise ValueError('Invalid observed-inventory goal')

    def achieved(self,before,after):
        return after.get(self.item_id,0)-before.get(self.item_id,0)>=self.increase


def observed_new_goals(before,after):
    """Hindsight outcomes measured solely from observed player inventory."""
    return tuple(ObservedItemGoal(item,change)
                 for item in sorted(set(before)|set(after))
                 if (change:=after.get(item,0)-before.get(item,0))>0)


def _json_inventory(inventory):
    return json.dumps(sorted(inventory.items()),separators=(',',':'),ensure_ascii=True)


def _inventory_from_json(text):
    pairs=json.loads(str(text))
    if not isinstance(pairs,list):
        raise ValueError('Invalid archived inventory')
    results={}
    for pair in pairs:
        if (not isinstance(pair,list) or len(pair)!=2 or
            not isinstance(pair[0],str) or ':' not in pair[0] or
            len(pair[0])>MAX_ITEM_ID or type(pair[1]) is not int or
            not 0<pair[1]<=100000):
            raise ValueError('Invalid archived inventory entry')
        if pair[0] in results:raise ValueError('Duplicate archived inventory item')
        results[pair[0]]=pair[1]
    return results


def archive_path(root,run_tag,episode):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',run_tag):
        raise ValueError('Unsafe experience run tag')
    if type(episode) is not int or not 0<=episode<10**12:
        raise ValueError('Unsafe episode number')
    return Path(root)/run_tag/f'episode_{episode:012d}.npz'


class EpisodeExperience:
    """Accumulate one world's observed motor choices, then atomically publish.

    The writer is never passed a controller, Minecraft world, seed, or raw
    packet. If the world process fails, discard this uncommitted object.
    """
    def __init__(self,*,observation_schema,action_sizes):
        if not isinstance(observation_schema,str) or not observation_schema.startswith('natural_memory_online_'):
            raise ValueError('Unexpected versioned observation schema')
        if not isinstance(action_sizes,tuple) or not action_sizes or any(
                type(n) is not int or not 1<=n<=256 for n in action_sizes):
            raise ValueError('Invalid versioned ordinary action heads')
        self.observation_schema=observation_schema
        self.action_sizes=action_sizes
        self.rows=[]

    def add(self,before,action,after,player_before,player_after,*,duration,reward,done):
        if not isinstance(before,NaturalObservation) or not isinstance(after,NaturalObservation):
            raise TypeError('Only bounded, typed observations may be archived')
        if (not isinstance(action,tuple) or len(action)!=len(self.action_sizes) or
            any(type(i) is not int or not 0<=i<n for i,n in zip(action,self.action_sizes)) or
            type(duration) is not int or not 1<=duration<=200 or
            not np.isfinite(reward) or type(done) is not bool):
            raise ValueError('Invalid ordinary action/outcome duration')
        if len(self.rows)>=MAX_RECORDS:
            raise ValueError('Episode experience limit exceeded')
        b=owned_inventory(player_before)
        a=owned_inventory(player_after)
        item_goals=observed_new_goals(b,a)
        payload={}
        for name in FIELDS:
            old=getattr(before,name)
            new=getattr(after,name)
            if old.shape!=new.shape or old.dtype!=new.dtype or not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError('Experience sensor schema drift or nonfinite data')
            payload['before_'+name]=old.copy()
            payload['after_'+name]=new.copy()
        payload.update(action=np.asarray(action,dtype=np.uint8),duration=np.int32(duration),
                       reward=np.float32(reward),done=np.bool_(done),
                       inventory_before=_json_inventory(b),inventory_after=_json_inventory(a))
        self.rows.append(payload)
        return item_goals

    def save(self,root,run_tag,episode,*,max_bytes=None):
        """Single atomic file; never replace an earlier complete episode."""
        if not self.rows:raise ValueError('Refusing empty episode archive')
        destination=archive_path(root,run_tag,episode)
        destination.parent.mkdir(parents=True,exist_ok=True)
        staged=destination.with_suffix('.npz.pending')
        if destination.exists() or staged.exists():
            raise FileExistsError('Archive episode or incomplete staged file exists: '+str(destination))
        meta=dict(schema=SCHEMA,observation_schema=self.observation_schema,
                  action_sizes=self.action_sizes,count=len(self.rows),
                  total_game_ticks=sum(int(row['duration']) for row in self.rows),
                  provenance='player_known_observations_and_owned_inventory')
        data={key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]
              if key not in ('inventory_before','inventory_after')}
        # Unicode np arrays are allow_pickle=False safe; never persist Python
        # object arrays, raw world packets, seed or secret target metadata.
        for key in ('inventory_before','inventory_after'):
            data[key]=np.asarray([row[key] for row in self.rows],dtype=np.str_)
        data['metadata']=np.asarray(json.dumps(meta,sort_keys=True))
        try:
            with staged.open('xb') as file:
                np.savez_compressed(file,**data)
                file.flush()
                os.fsync(file.fileno())
            if max_bytes is not None and staged.stat().st_size>max_bytes:
                raise ValueError('Single episode exceeds replay budget; increase archive limit')
            # Link (same directory) is an exclusive, atomic publish. Do
            # not overwrite completed archive even if an unexpected writer
            # races us. A failed stage stays available for recovery audit.
            os.link(staged,destination,follow_symlinks=False)
            staged.unlink()
            fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        except Exception:
            # Preserve a partially written file for quarantine on resume.
            raise
        return dict(path=str(destination),bytes=destination.stat().st_size,
                    steps=len(self.rows),game_ticks=meta['total_game_ticks'])


def read_episode(path):
    """Load bounded player-visible samples, verify integrity, no pickle."""
    path=Path(path)
    if not FILENAME.fullmatch(path.name):
        raise ValueError('Unexpected archive episode filename')
    with np.load(path,allow_pickle=False) as source:
        meta=json.loads(str(source['metadata'].item()))
        if meta.get('schema')!=SCHEMA or meta.get('provenance')!='player_known_observations_and_owned_inventory':
            raise ValueError('Incompatible player-observation archive')
        n=meta['count']
        if type(n) is not int or not 1<=n<=MAX_RECORDS:
            raise ValueError('Invalid experience count')
        expected={'metadata','action','duration','reward','done','inventory_before','inventory_after'}
        for name in FIELDS: expected.update(('before_'+name,'after_'+name))
        if set(source.files)!=expected:raise ValueError('Unexpected archive fields / potential privileged leak')
        data={key:source[key] for key in source.files if key!='metadata'}
    if any(len(array)!=n for array in data.values()):
        raise ValueError('Truncated experience archive')
    from .natural_online_policy import (
        SCHEMA as BASIC,PERSISTENCE_SCHEMA,WOOD_ENGAGEMENT_SCHEMA,
        MOTOR_V7_SCHEMA,TEMPORAL_V9_SCHEMA,SELF_FEATURES,WOOD_ENGAGEMENT_SELF_FEATURES,
        TEMPORAL_V9_SELF_FEATURES,
        ID_BYTES,MAX_OBJECTS,OBJECT_FEATURES)
    expected_sizes={
        BASIC:len(SELF_FEATURES),
        PERSISTENCE_SCHEMA:len(SELF_FEATURES),
        WOOD_ENGAGEMENT_SCHEMA:len(WOOD_ENGAGEMENT_SELF_FEATURES),
        MOTOR_V7_SCHEMA:len(WOOD_ENGAGEMENT_SELF_FEATURES),
        TEMPORAL_V9_SCHEMA:len(TEMPORAL_V9_SELF_FEATURES),
    }
    own_size=expected_sizes.get(meta.get('observation_schema'))
    if own_size is None:raise ValueError('Unrecognized archived sensor schema')
    shape_dtype={'own':((n,own_size),np.dtype('float32')),
                 'crosshair_id':((n,ID_BYTES),np.dtype('uint8')),
                 'surface_ids':((n,MAX_OBJECTS,ID_BYTES),np.dtype('uint8')),
                 'surface_data':((n,MAX_OBJECTS,len(OBJECT_FEATURES)),np.dtype('float32')),
                 'surface_mask':((n,MAX_OBJECTS),np.dtype('float32'))}
    for prefix in ('before_','after_'):
        for field,(shape,dtype) in shape_dtype.items():
            array=data[prefix+field]
            if array.shape!=shape or array.dtype!=dtype or not np.isfinite(array).all():
                raise ValueError('Archived player sensor field mismatch')
    if data['done'].dtype!=np.dtype('bool') or data['reward'].dtype!=np.dtype('float32') or (
            data['duration'].dtype!=np.dtype('int32')):
        raise ValueError('Invalid archive reward/terminal/duration type')
    if data['action'].shape!=(n,len(meta['action_sizes'])) or data['action'].dtype!=np.uint8:
        raise ValueError('Unknown motor action schema')
    if (not np.isfinite(data['reward']).all() or
        not np.all((data['duration']>=1)&(data['duration']<=200)) or
        int(data['duration'].sum())!=meta['total_game_ticks']):
        raise ValueError('Invalid archived game duration/reward')
    for row,heads in enumerate(data['action']):
        if any(head>=limit for head,limit in zip(heads,meta['action_sizes'])):
            raise ValueError('Invalid action index in archive')
    for key in ('inventory_before','inventory_after'):
        for value in data[key]:_inventory_from_json(value)
    return meta,data


def hindsight_goal_transitions(meta,data,*,lookback_steps=40):
    """Self-generated successful examples (goal, state_index, action).

    Relabels own observed inventory increase with earlier real actions.
    Does not invent a route, a hidden goal location, a reward, or extra
    observed information. Only positive achieved goals are emitted.
    """
    if type(lookback_steps) is not int or not 1<=lookback_steps<=10000:
        raise ValueError('Invalid hindsight goal horizon')
    n=meta['count']
    for index in range(n):
        before=_inventory_from_json(data['inventory_before'][index])
        after=_inventory_from_json(data['inventory_after'][index])
        for goal in observed_new_goals(before,after):
            start=max(0,index-lookback_steps+1)
            # A new death/episode terminal must not be crossed.
            for j in range(index-1,start-1,-1):
                if bool(data['done'][j]):start=j+1;break
            for j in range(start,index+1):
                # A preceding state holding MORE of this item than the
                # achieved state is not a successful hindsight example.
                if not goal.achieved(_inventory_from_json(data['inventory_before'][j]),after):
                    continue
                yield dict(goal_item_id=goal.item_id,goal_increase=goal.increase,
                           observation_index=j,achieved_index=index,
                           action=tuple(int(x) for x in data['action'][j]),
                           ticks_to_achievement=int(data['duration'][j:index+1].sum()))


def reconcile_uncommitted(root,run_tag,committed_cursor):
    """Move ahead-of-checkpoint archives aside; never train from orphaned data."""
    directory=Path(root)/run_tag
    if not directory.exists():return []
    moved=[]
    for entry in sorted(directory.iterdir()):
        m=FILENAME.fullmatch(entry.name)
        if m is not None and int(m.group(1))<committed_cursor:
            continue
        if not (m is not None or entry.name.endswith('.npz.pending')):
            continue
        for i in range(10000):
            target=entry.with_name(entry.name+f'.uncommitted_{i:04d}')
            if not target.exists():
                entry.rename(target)
                moved.append(str(target))
                break
        else:raise FileExistsError('Too many orphaned experience files')
    return moved


def compact_retention(root,run_tag,limit_bytes):
    """Optional bounded on-disk replay window; model weights unaffected.

    Prune earliest archive episodes only, and state each deletion explicitly.
    No hardlinks or arbitrary paths are followed. Call only while holding
    this exact run's existing trainer advisory lock.
    """
    if type(limit_bytes) is not int or not 1_000_000<=limit_bytes:
        raise ValueError('Invalid replay storage budget')
    directory=Path(root)/run_tag
    if not directory.exists():return []
    paths=sorted(p for p in directory.iterdir()
                 if p.is_file() and not p.is_symlink() and FILENAME.fullmatch(p.name))
    total=sum(p.stat().st_size for p in paths)
    deleted=[]
    for path in paths:
        if total<=limit_bytes:break
        size=path.stat().st_size
        path.unlink()
        total-=size
        deleted.append(dict(file=path.name,bytes=size))
    return deleted
