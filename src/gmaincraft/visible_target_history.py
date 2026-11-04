"""V9: bounded history of what the player actually targeted and attempted.

History is constructed ONLY from successive own vanilla crosshair readings,
own break-animation feedback and already-issued ordinary player actions.
No world/chunk scans, privileged break event, item location or target steering.
The model remains free to ignore all of these fields.
"""
from math import isfinite

from .agent_interface import AgentAction
from .mcio_mining_feedback import MiningFeedbackV1
from .mcio_status import PlayerStatusV1
from .natural_online_policy import wood_id

SCHEMA='player_visible_target_history_v9'
FEATURE_NAMES=(
    'same_vanilla_crosshair_target_as_previous',
    'crosshair_same_target_ticks_div_72',
    'previous_sample_crosshair_was_wood',
    'last_action_attack_pressed',
    'last_action_stationary_attack',
    'stationary_attack_same_target_ticks_div_72',
    'crack_known_on_current_target',
    'crack_stage_rose_same_target_div_10',
    'highest_seen_crack_for_current_target_div_10',
    'known_crack_no_progress_ticks_div_72',
)


class VisibleTargetHistory:
    """Ephemeral per-world history; reset automatically with each real seed.

    Every live observation is sampled AFTER the last normal action chunk;
    no extra target sensor probes or invocations affect Minecraft gameplay.
    """
    def __init__(self):
        self.previous_key=None
        self.previous_stage=-1
        self.previous_wood=False
        self.continuous_target_ticks=0
        self.stationary_same_target_ticks=0
        self.highest_stage=-1
        self.stall_ticks=0
        self.samples=0
        self.same_target_samples=0
        self.known_crack_rises=0
        self.last_features=(0.,)*len(FEATURE_NAMES)

    def observe(self,status,mining,*,elapsed_ticks,previous_action=None):
        if not isinstance(status,PlayerStatusV1):
            raise TypeError('History requires the actual typed vanilla player status')
        if mining is not None and not isinstance(mining,MiningFeedbackV1):
            raise TypeError('History requires actual player-visible crack feedback')
        if (type(elapsed_ticks) is not int or not 0<=elapsed_ticks<=200 or
                (previous_action is None)!=(elapsed_ticks==0) or
                (previous_action is not None and (
                    not isinstance(previous_action,AgentAction) or
                    previous_action.ticks!=elapsed_ticks))):
            raise ValueError('History requires the actual preceding ordinary game action/ticks')
        target=status.crosshair_block
        key=(target.block_id,tuple(target.xyz)) if target is not None else None
        same=bool(key is not None and key==self.previous_key)
        attack=bool(previous_action and 'LEFT_BUTTON' in previous_action.keys)
        still=bool(attack and not previous_action.look_dx and not previous_action.look_dy
                   and not any(k in previous_action.keys for k in ('W','S','A','D','SPACE')))
        stage=-1
        if (target is not None and mining is not None and
                mining.crosshair_on_block and mining.actively_destroying and
                mining.crack_stage>=0):
            stage=min(10,mining.crack_stage)
        known=stage>=0
        rise=max(0,stage-self.previous_stage) if same and known and self.previous_stage>=0 else 0
        self.continuous_target_ticks=(min(72,self.continuous_target_ticks+elapsed_ticks)
                                      if same else 0)
        self.stationary_same_target_ticks=(min(72,self.stationary_same_target_ticks+elapsed_ticks)
                                           if same and still else 0)
        if not same:self.highest_stage=-1
        if known:self.highest_stage=max(self.highest_stage,stage)
        # Stall is meaningful ONLY on the same block with both crack stages
        # actually known. Unknown or lost animations are NOT stage zero.
        if same and known and self.previous_stage>=0 and rise==0:
            self.stall_ticks=min(72,self.stall_ticks+elapsed_ticks)
        else:self.stall_ticks=0
        features=(float(same),self.continuous_target_ticks/72.,
                  float(self.previous_wood),float(attack),float(still),
                  self.stationary_same_target_ticks/72.,float(known),
                  rise/10.,max(0,self.highest_stage)/10.,self.stall_ticks/72.)
        if len(features)!=len(FEATURE_NAMES) or not all(isfinite(x) and 0<=x<=1 for x in features):
            raise AssertionError('Invalid bounded player target history')
        self.previous_key=key
        self.previous_stage=stage
        self.previous_wood=bool(target is not None and wood_id(target.block_id))
        self.samples+=1
        self.same_target_samples+=int(same)
        self.known_crack_rises+=int(rise>0)
        self.last_features=features
        return features

    def summary(self):
        return dict(schema=SCHEMA,samples=self.samples,
                    same_target_samples=self.same_target_samples,
                    visible_crack_rise_samples=self.known_crack_rises)
