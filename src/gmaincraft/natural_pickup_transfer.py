"""Honest *diagnostic* transfer of frozen pickup practice to survival.

This is NOT a learned high-level skill selector. A transparent visibility-
gated experiment tests whether a pretrained pickup motor works on vanilla
natural drops when the original frozen mining Q actor is otherwise active.
No evaluator-only mined counters, hidden item locations, teleport, commands,
spawned objects or controller-only phase flags are used to select an action.
"""
from collections import Counter

from .agent_interface import AgentAction
from .mcio_visible_items import VisibleItemsV1
from .natural_online_policy import wood_id
from .numeric_item_pickup import ACTIONS,motor_action
from .numeric_item_pickup_v2 import EgocentricPickupMemory

CONTRACT='player_seen_dropped_wood_pickup_transfer_v1'
SELECTORS=('shadow','visible_memory_v1')
MAX_OWN_MEMORY_PURSUIT_TICKS=90


def visible_wood_only(items):
    """Filter the CURRENT already camera-and-ray-visible objects to wood."""
    if not isinstance(items,VisibleItemsV1):
        raise TypeError('Need typed same-action camera-filtered dropped items')
    return VisibleItemsV1(items.schema,items.max_distance,
                          tuple(item for item in items.items if wood_id(item.item_id)))


class NaturalPickupTransfer:
    def __init__(self,brain,*,selector='shadow',greedy=True):
        if selector not in SELECTORS:
            raise ValueError('Unknown diagnostic pickup selector')
        if brain is None or not callable(getattr(brain,'act',None)):
            raise TypeError('Need existing frozen pickup skill')
        self.brain=brain
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

    def observe(self,player,items,*,elapsed_ticks):
        if not isinstance(elapsed_ticks,int) or elapsed_ticks<self.ticks:
            raise ValueError('Nonmonotonic real-game elapsed ticks')
        self.ticks=elapsed_ticks
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
        known=bool(self.latest.features[1])
        age=int(round(float(self.latest.features[9])*120))
        eligible=(self.engaged and known and
                  (self.latest.item_seen or age<=MAX_OWN_MEMORY_PURSUIT_TICKS))
        self.in_pursuit=bool(eligible and self.selector=='visible_memory_v1')
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
        return dict(contract=CONTRACT,selector=self.selector,
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
