"""Honest *diagnostic* transfer of frozen pickup practice to survival.

This is NOT a learned high-level skill selector. A transparent visibility-
gated experiment tests whether a pretrained pickup motor works on vanilla
natural drops when the original frozen mining Q actor is otherwise active.
No evaluator-only mined counters, hidden item locations, teleport, commands,
spawned objects or controller-only phase flags are used to select an action.
"""
from collections import Counter
from math import dist

from .agent_interface import AgentAction
from .mcio_visible_items import VisibleItemsV1
from .natural_online_policy import wood_id
from .numeric_item_pickup import ACTIONS,motor_action
from .numeric_item_pickup_v2 import EgocentricPickupMemory
from .natural_option_q_v1 import option_features

CONTRACT='player_seen_dropped_wood_pickup_transfer_v1'
BUDGET_CONTRACT='player_seen_dropped_wood_pickup_transfer_budget_v2'
OPTION_CONTRACT='player_seen_dropped_wood_learned_option_selector_v1'
SELECTORS=('shadow','visible_memory_v1','shadow_budget_v2','visible_memory_budget_v2',
           'shadow_learned_option_v1','learned_option_v1')
BUDGET_ATTEMPT_TICKS=96

def contract_for_selector(selector):
    if selector not in SELECTORS:raise ValueError('Unknown pickup transfer version')
    return (OPTION_CONTRACT if selector.endswith('learned_option_v1') else
            BUDGET_CONTRACT if selector.endswith('budget_v2') else CONTRACT)

MAX_OWN_MEMORY_PURSUIT_TICKS=90


def visible_wood_only(items):
    """Filter the CURRENT already camera-and-ray-visible objects to wood."""
    if not isinstance(items,VisibleItemsV1):
        raise TypeError('Need typed same-action camera-filtered dropped items')
    return VisibleItemsV1(items.schema,items.max_distance,
                          tuple(item for item in items.items if wood_id(item.item_id)))


class NaturalPickupTransfer:
    def __init__(self,brain,*,selector='shadow',greedy=True,option_brain=None):
        if selector not in SELECTORS:
            raise ValueError('Unknown diagnostic pickup selector')
        if brain is None or not callable(getattr(brain,'act',None)):
            raise TypeError('Need existing frozen pickup skill')
        if selector.endswith('learned_option_v1') and option_brain is None:
            raise ValueError('Learned option selector requires frozen trained option checkpoint')
        self.brain=brain
        self.option_brain=option_brain
        self.latest_mining=None
        self.selector=selector
        self.greedy=bool(greedy)
        self.memory=EgocentricPickupMemory()
        self.ticks=0
        self.latest=None
        self.last_own_wood=None
        self.last_action=0
        self.engaged=False
        self.in_pursuit=False
        self.stats=Counter()
        self.action_names=Counter()
        self.budget_target=None
        self.budget_blocked=None
        self.budget_since=None

    def observe(self,player,items,*,elapsed_ticks,mining_observation=None):
        if not isinstance(elapsed_ticks,int) or elapsed_ticks<self.ticks:
            raise ValueError('Nonmonotonic real-game elapsed ticks')
        self.ticks=elapsed_ticks
        if self.selector.endswith('learned_option_v1'):
            if mining_observation is None:
                raise ValueError('Learned option requires same-tick player-visible V9 mining observation')
            self.latest_mining=mining_observation
        held=sum(int(slot.count) for slot in player.main_inventory+player.offhand_inventory
                 if wood_id(slot.item_id))
        # Observing OWN inventory gain is legitimate, not privileged
        # `minecraft:mined`/server pickup-stat evaluation. Forget a consumed
        # item instead of hallucinating its remembered coordinates forever.
        if self.last_own_wood is not None and held>self.last_own_wood:
            self.stats['inventory_gains_seen']+=held-self.last_own_wood
            self.memory=EgocentricPickupMemory()
            self.engaged=False
            self.in_pursuit=False
            self.budget_target=None
            self.budget_blocked=None
            self.budget_since=None
        self.last_own_wood=held
        visible=visible_wood_only(items)
        self.latest=self.memory.observe(player,visible,
                       elapsed_ticks=elapsed_ticks,last_action=self.last_action)
        self.stats['observations']+=1
        self.stats['visible_wood_item_observations']+=int(self.latest.item_seen)
        self.stats['max_visible_wood_items']=max(
            self.stats['max_visible_wood_items'],self.latest.visible_candidate_count)
        if self.latest.item_seen:
            self.engaged=True
            self.stats['seen_opportunities']+=1
        return self.latest

    def choose(self,remaining_ticks):
        if not isinstance(remaining_ticks,int) or remaining_ticks<=0:
            raise ValueError('No vanilla game time remaining')
        if self.latest is None:raise RuntimeError('Pickup selector never observed Minecraft')
        was=self.in_pursuit
        if self.selector.endswith('learned_option_v1'):
            if self.latest_mining is None:
                raise RuntimeError('Learned option never observed actual current mining state')
            if self.selector=='shadow_learned_option_v1':
                self.in_pursuit=False
                self.last_action=0
                return None
            selected=self.option_brain.act(
                option_features(self.latest,self.latest_mining),greedy=True)
            self.stats['learned_option_selections']+=1
            if selected==0:
                self.in_pursuit=False
                self.last_action=0
                return None
            if selected!=1:
                raise ValueError('Trained option chose an unknown controller')
            self.in_pursuit=True
            self.stats['learned_pickup_option_selections']+=1
            index=self.brain.act(self.latest,greedy=self.greedy)
            action=motor_action(index)
            self.last_action=index
            self.stats['pickup_policy_actions']+=1
            self.action_names[ACTIONS[index][0]]+=1
            return AgentAction(keys=action.keys,look_dx=action.look_dx,
                               look_dy=action.look_dy,
                               ticks=min(action.ticks,remaining_ticks))
        known=bool(self.latest.features[1])
        age=int(round(float(self.latest.features[9])*120))
        eligible=(self.engaged and known and
                  (self.latest.item_seen or age<=MAX_OWN_MEMORY_PURSUIT_TICKS))
        if self.selector.endswith('budget_v2'):
            # Honest single-target effort cap, based only on the player's
            # first REAL sighting and elapsed Minecraft ticks. No privileged
            # item-entity UUID, server inventory events or rewards.
            current=(self.memory.last_item_id,self.memory.xyz) if known else None
            def same(a,b):
                return (a is not None and b is not None and a[0]==b[0] and
                        dist(a[1],b[1])<=1.5)
            if not same(current,self.budget_target) and self.latest.item_seen:
                if not same(current,self.budget_blocked):
                    self.budget_target=current
                    self.budget_since=self.ticks
                    self.stats['distinct_visible_targets_pursued']+=1
                else:
                    eligible=False
            elif self.budget_target is None:
                eligible=False
            if self.budget_target is not None:
                if self.budget_since is None:
                    raise RuntimeError('Pickup target budget was not initialized')
                if self.ticks-self.budget_since>=BUDGET_ATTEMPT_TICKS:
                    self.budget_blocked=self.budget_target
                    self.budget_target=None
                    self.budget_since=None
                    eligible=False
                    self.stats['pickup_effort_budget_exhaustions']+=1
            if same(current,self.budget_blocked):
                eligible=False
        self.in_pursuit=bool(eligible and self.selector in
                         ('visible_memory_v1','visible_memory_budget_v2'))
        if not self.in_pursuit:
            self.last_action=0
            return None
        if not was:self.stats['own_visible_pickup_engagements']+=1
        index=self.brain.act(self.latest,greedy=self.greedy)
        action=motor_action(index)
        self.last_action=index
        self.stats['pickup_policy_actions']+=1
        self.action_names[ACTIONS[index][0]]+=1
        return AgentAction(keys=action.keys,look_dx=action.look_dx,
                           look_dy=action.look_dy,
                           ticks=min(action.ticks,remaining_ticks))

    def audit(self):
        return dict(contract=contract_for_selector(self.selector),selector=self.selector,
                    **dict(self.stats),pickup_motor_counts=dict(self.action_names))


def coarse_audit_heads(action):
    """Approximate old factorized heads only for scoring non-learning V9.

    Mixed diagonals cannot be represented by the old reward-history schema:
    map to the forward component for *reward bookkeeping only*. The actual
    physical key holds are kept intact and separately counted as pickup
    motors; nothing is fed back into frozen model weights.
    """
    if not isinstance(action,AgentAction):raise TypeError('Expected real motor')
    move=(1 if 'W' in action.keys else 2 if 'S' in action.keys else
          3 if 'A' in action.keys else 4 if 'D' in action.keys else 0)
    look=(1 if action.look_dx<0 else 2 if action.look_dx>0 else
          5 if action.look_dy<0 else 6 if action.look_dy>0 else 0)
    return (move,look,int('LEFT_BUTTON' in action.keys))
