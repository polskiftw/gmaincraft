"""Event-derived vanilla Minecraft reward, distinct from oracle geometry.

Statistics are obtained by the privileged evaluator from Minecraft's own
logged mined/picked-up events. They are NEVER passed into AgentView or neural
policy inputs. Count only forward changes within a scored real-game episode;
reset/reused statistics must not manufacture reward from stale items.
"""
from math import isfinite


def mined_event_delta(previous, current):
    if not (isinstance(previous,int) and isinstance(current,int)):
        raise TypeError('Event statistics must be integer counts')
    if previous<0 or current<previous:
        raise ValueError('Mined statistic went backwards during a scored episode')
    return current-previous


def mined_event_reward(previous,current,*,reward_per_block=5.,cap=3):
    if not isinstance(cap,int) or cap<1:
        raise ValueError('Invalid per-action block count cap')
    if not isinstance(reward_per_block,(int,float)) or not isfinite(reward_per_block) or reward_per_block<0:
        raise ValueError('Invalid event reward amount')
    return min(mined_event_delta(previous,current),cap)*float(reward_per_block)


class CumulativeMinedTracker:
    """World-session counter safe across per-episode observation cache resets.

    MCio's StatsUpdateOption sends only *changed* stats. Each new bridge starts
    with cache value zero even if the player's cumulative minecraft:mined
    statistic has been incrementing in prior episodes. The next mined block
    may therefore update the new bridge from 0 straight to 7. THIS IS ONE
    newly mined block, not seven! Keep the last absolute value in the shared
    training arena, not in disposable per-episode bridges.
    """
    def __init__(self):
        self.absolute=0
        self.last=0
        self.in_episode=False

    def begin_episode(self,initial_report=0):
        if not isinstance(initial_report,int) or initial_report<0:
            raise ValueError('Invalid initial mined count')
        self.absolute=max(self.absolute,initial_report)
        self.last=self.absolute
        self.in_episode=True

    def observe(self,reported):
        if not self.in_episode:
            raise RuntimeError('begin_episode() required')
        if not isinstance(reported,int) or reported<0:
            raise ValueError('Invalid mined count')
        current=max(self.absolute,reported)
        increased=current-self.last
        self.last=current
        self.absolute=current
        return increased
