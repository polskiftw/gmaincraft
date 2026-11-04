"""V2-only nonforgetting self-play learner: protect actual past successes.

A completed prior natural-survival TRAIN stream may supply old SELF-ACHIEVED
visible wood crack transitions. The player's own new world supplies fresh
transitions, also eligible only after normal Minecraft shutdown and atomic
replay publication. NO heldout world, expert actions, target coordinates,
scripted phase switch, seed or original gameplay oracle enters the network.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .experience_archive import archive_path,read_episode
from .fresh_seed_schedule import SCHEMA as STREAM_SCHEMA
from .natural_online_policy import MOTOR_V7_HEAD_SIZES,TEMPORAL_V9_SCHEMA
from .replay_q_rare_v2 import WOOD_CRACK_GOAL,WOOD_INDEX,wood_crack_transitions


def buckets(rows):
    positive=[r for r in rows if r['achieved']]
    visible_failure=[r for r in rows if not r['achieved'] and r['before_own'][WOOD_INDEX]>.5]
    other=[r for r in rows if not r['achieved'] and r['before_own'][WOOD_INDEX]<=.5]
    return positive,visible_failure,other


@lru_cache(maxsize=3)
def committed_prior_success_source(root,run):
    """Load only prior checkpoint-committed fresh TRAIN stream, never eval.

    In-process cache is read-only, exact file roots and source tag. A fresh
    sandbox session recomputes it from the same completed source checkpoint.
    """
    import torch
    root=Path(root)
    saved=torch.load(root/'checkpoints'/f'fresh_stream_{run}.pt',map_location='cpu',weights_only=True)
    meta=saved['metadata']
    if (saved.get('stream')!=STREAM_SCHEMA or
        meta.get('kind') not in ('gmaincraft_q_collect_v1','gmaincraft_q_collect_v2_rare_outcome','gmaincraft_fresh_stream_v2_fixed_spawn') or
        meta.get('policy_schema')!=TEMPORAL_V9_SCHEMA or
        tuple(meta.get('policy_action_heads',()))!=MOTOR_V7_HEAD_SIZES or
        meta.get('reward_profile')!='wood_temporal_v9' or
        type(saved.get('next_episode')) is not int or saved['next_episode']<1):
        raise ValueError('Refusing incompatible or uncommitted prior Minecraft self-play source')
    rows=[]
    episodes=0
    for index in range(saved['next_episode']):
        path=archive_path(root/'data-local'/'experience',run,index)
        if not path.is_file() or path.is_symlink():continue
        epmeta,data=read_episode(path)
        rows.extend(wood_crack_transitions(epmeta,data))
        episodes+=1
    bins=buckets(rows)
    if not bins[0] or not bins[1] or not bins[2]:
        raise ValueError('Prior real player archive has no verified rare-success practice dataset')
    return bins,dict(source_run=run,source_cursor=saved['next_episode'],
                     source_start_index=meta.get('collector_start_index',0),
                     source_seed_root=meta['seed_root'],source_committed_episodes=episodes,
                     prior_real_successes=len(bins[0]),
                     prior_target_failures=len(bins[1]),prior_other_actions=len(bins[2]))


def new_world_buckets(root,run,current_episode,*,recent_worlds):
    if (type(current_episode) is not int or current_episode<0 or
        type(recent_worlds) is not int or not 1<=recent_worlds<=128):
        raise ValueError('Invalid new-world replay window')
    rows=[]
    count=0
    for index in range(max(0,current_episode-recent_worlds+1),current_episode+1):
        path=archive_path(root,run,index)
        if not path.is_file() or path.is_symlink():continue
        meta,data=read_episode(path)
        rows.extend(wood_crack_transitions(meta,data))
        count+=1
    return buckets(rows),count


def learn_between_real_worlds(model,root,run,episode,*,source_run=None,
                               additional_source_run=None,protected_source_runs=None,
                               updates=32,batch_size=48,recent_worlds=32,
                               anchor_weight=2.,success_weight=.6):
    """Priority-balanced REAL experience with ALL protected successes forever.

    Source and new gameplay are both disjoint TRAIN streams. The previous
    all-recent-window-only Q learner forgot successful wood actions as old
    successful episodes rolled out. Keep actual historical evidence available;
    do not create or hardcode an attack action when the target is visible.
    """
    if (not isinstance(root,(str,Path)) or type(updates) is not int or
        not 0<=updates<=1000 or type(batch_size) is not int or
        not 12<=batch_size<=256 or batch_size%3 or
        type(recent_worlds) is not int or not 1<=recent_worlds<=128 or
        not 0<=anchor_weight<=10 or not 0<success_weight<=10):
        raise ValueError('Invalid V2 online training settings')
    if protected_source_runs is not None:
        if (source_run is not None or additional_source_run is not None or
            not isinstance(protected_source_runs,tuple)):
            raise ValueError('Choose one explicit source lineage contract')
        lineage=protected_source_runs
    else:
        lineage=(source_run,)+((additional_source_run,) if additional_source_run else ())
    import re
    if (not lineage or len(lineage)>128 or len(lineage)!=len(set(lineage)) or
        any(not isinstance(tag,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',tag)
            for tag in lineage)):
        raise ValueError('duplicate or invalid protected real self-play sources')
    if run in lineage:
        raise ValueError('Do not collect into the same immutable prior source')
    prior=None
    seed_root=None
    seen_intervals=[]
    for old in lineage:
        earlier,info=committed_prior_success_source(str(Path(root).resolve()),old)
        if seed_root is None:seed_root=info['source_seed_root']
        if info['source_seed_root']!=seed_root:
            raise ValueError('Protected natural source worlds must share disjoint seed stream root')
        start,end=info['source_start_index'],info['source_cursor']
        if (type(start) is not int or start<0 or start>=end or
            any(max(start,a)<min(end,b) for a,b in seen_intervals)):
            raise ValueError('Protected prior TRAIN seed-index ranges overlap or are invalid')
        seen_intervals.append((start,end))
        prior=tuple(list(a)+list(b) for a,b in zip(prior,earlier)) if prior is not None else earlier
    current,num_new=new_world_buckets(Path(root)/'data-local'/'experience',run,episode,
                                      recent_worlds=recent_worlds)
    # ROOT is the local experience base. Read the same verified source from
    # its enclosing repository path, not a different mount or hidden file.
    if not updates:return dict(updates=0,reason='disabled',protected_source_runs=lineage,
                               source_verified_successes=len(prior[0]))
    rng=np.random.default_rng(20261104+episode)
    losses=[]
    for _ in range(updates):
        batch=[]
        for previous,recent in zip(prior,current):
            quota=batch_size//3
            # True successes remain in permanent memory even if the most
            # recent dozen worlds have NO achievements; fresh events still
            # enter the same learner when they occur.
            count_new=quota//2 if recent else 0
            for pool,count in ((previous,quota-count_new),(recent,count_new)):
                if not count:continue
                inds=rng.integers(0,len(pool),size=count)
                batch.extend(pool[int(ix)] for ix in inds)
        rng.shuffle(batch)
        report=model.update(batch,WOOD_CRACK_GOAL,anchor_weight=anchor_weight,
                            success_weight=success_weight)
        losses.append(report['loss'])
    return dict(updates=updates,worlds_used=num_new,source_verified_successes=len(prior[0]),
                new_own_successes=len(current[0]),source_run=lineage[0],
                additional_source_run=lineage[1] if len(lineage)>1 else None,
                protected_source_runs=lineage,
                model_optimizer_updates=model.updates,mean_loss=round(float(np.mean(losses)),6),
                no_evaluator_hidden_game_state=True,
                protected_successes_are_actual_prior_player_observations=True)
