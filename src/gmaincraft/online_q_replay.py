"""Checkpoint-safe replay updates BETWEEN true vanilla self-play worlds.

The collector's only stored labels come from real before/after player-own
observations. Never read uncommitted prior episodes; the current episode is
eligible only after the real Minecraft server exited and its archive was
atomically published. Caller MUST persist Q weights/optimizer and cursor
before declaring an episode committed. No scripts/teacher routing/seed input.
"""
from pathlib import Path

import numpy as np

from .experience_archive import archive_path,read_episode
from .replay_q_skill import goal_transitions


def recent_player_replay(root,tag,current_episode,*,goal,max_episodes=32):
    if (type(current_episode) is not int or current_episode<0 or
        type(max_episodes) is not int or not 1<=max_episodes<=128):
        raise ValueError('Invalid replay window')
    rows=[]
    episodes=[]
    for index in range(max(0,current_episode-max_episodes+1),current_episode+1):
        path=archive_path(root,tag,index)
        if not path.is_file() or path.is_symlink():continue
        meta,data=read_episode(path)
        rows.extend(goal_transitions(meta,data,goal))
        episodes.append(index)
    return rows,episodes


def update_from_player_replay(model,root,tag,current_episode,*,goal,
                              updates,batch_size,max_episodes):
    """Only successful PLAYER-observed consequences, no fabricated examples.

    This is training, not frozen evaluation. Sampling with an episode-keyed
    RNG makes an exact crash+checkpoint rollback replayable, without revealing
    any seed or episode index to the acting policy.
    """
    if (type(updates) is not int or not 0<=updates<=1000 or
        type(batch_size) is not int or not 8<=batch_size<=256):
        raise ValueError('Invalid bounded Q optimizer budget')
    if not updates:return dict(updates=0,reason='disabled')
    rows,episodes=recent_player_replay(root,tag,current_episode,goal=goal,
                                       max_episodes=max_episodes)
    positives=sum(int(r['achieved']) for r in rows)
    if not positives:return dict(updates=0,episodes=len(episodes),actions=len(rows),
                                 genuine_outcomes=0,reason='no_player_observed_outcomes')
    priorities=np.asarray([r['priority'] for r in rows],np.float64)
    priorities/=priorities.sum()
    rng=np.random.default_rng(20261104+current_episode)
    losses=[]
    for _ in range(updates):
        selected=rng.choice(len(rows),size=batch_size,replace=True,p=priorities)
        outcome=model.update([rows[int(i)] for i in selected],goal)
        losses.append(outcome['loss'])
    return dict(updates=updates,episodes=len(episodes),actions=len(rows),
                genuine_outcomes=positives,last_optimizer_update=model.updates,
                mean_loss=round(float(np.mean(losses)),6),
                source='only_committed_or_current_staged_player_observations')
