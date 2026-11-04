"""Bounded own-memory novelty and monotonic player stats across natural runs.

The controller never inspects world geometry. Discovery counts only first-hit
camera surfaces, the player's own location and actual vanilla own stats.
All state is world-indexed so switching genuine natural seeds does not make
one world's coordinates appear explored in another world.
"""
from collections import defaultdict
from math import floor, sqrt

from .natural_online_policy import wood_id

MAX_VISITS_PER_WORLD=150000
MAX_SEEN_SURFACES_PER_WORLD=120000


class NaturalSessionMemory:
    def __init__(self):
        self.visits={}
        self.surfaces={}
        self.starts={}
        self.last_actions={}
        self.ticks={}
        self.absolute_stats={}
        self.inventory_logs={}

    def state(self):
        return dict(visits={k:dict(v) for k,v in self.visits.items()},
                    surfaces={k:list(v) for k,v in self.surfaces.items()},
                    starts=dict(self.starts),last_actions=dict(self.last_actions),
                    ticks=dict(self.ticks),absolute_stats={k:dict(v) for k,v in self.absolute_stats.items()},
                    inventory_logs=dict(self.inventory_logs))

    def restore(self,state):
        self.visits={int(k):dict(v) for k,v in state['visits'].items()}
        self.surfaces={int(k):set(tuple(v) for v in rows) for k,rows in state['surfaces'].items()}
        self.starts={int(k):tuple(v) for k,v in state['starts'].items()}
        self.last_actions={int(k):tuple(v) for k,v in state['last_actions'].items()}
        self.ticks={int(k):int(v) for k,v in state['ticks'].items()}
        self.absolute_stats={int(k):dict(v) for k,v in state['absolute_stats'].items()}
        self.inventory_logs={int(k):int(v) for k,v in state['inventory_logs'].items()}
        return self

    def ensure_world(self,index,player,stats):
        if index not in self.starts:
            self.starts[index]=tuple(player.player_xyz)
            self.visits[index]={}
            self.surfaces[index]=set()
            self.last_actions[index]=(0,0,0)
            self.ticks[index]=0
            self.absolute_stats[index]={}
            self.inventory_logs[index]=self.count_logs(player)
        # Baseline old per-bridge cumulative stat resets. Do not reward any
        # counters reported merely because a new process starts.
        for key,value in self._stat_values(stats).items():
            self.absolute_stats[index][key]=max(self.absolute_stats[index].get(key,0),value)
        self.inventory_logs[index]=self.count_logs(player)

    @staticmethod
    def count_logs(player):
        return sum(int(slot.count) for slot in
                   player.main_inventory+player.offhand_inventory
                   if wood_id(slot.item_id))

    @staticmethod
    def cell(player):
        x,_,z=player.player_xyz
        return (floor(x/2),floor(z/2))

    def cell_visits(self,index,player):
        return self.visits[index].get(self.cell(player),0)

    @staticmethod
    def _stat_values(cache):
        output={}
        for category in ('minecraft:mined','minecraft:picked_up'):
            for name,value in cache.get(category,{}).items():
                if ':' not in name or int(value)<0:
                    raise ValueError('Invalid player-owned Minecraft cumulative statistic')
                output[(category,name)]=int(value)
        return output

    def accrue(self,index,player,surface,stats,action,ticks,*,death=False,
               reward_profile='explore_v1'):
        if index not in self.visits:raise RuntimeError('Must open real world first')
        if reward_profile not in ('explore_v1','wood_discovery_v2','wood_target_v5','wood_engagement_v6'):
            raise ValueError('Unknown natural exploration reward version')
        cell=self.cell(player)
        visits=self.visits[index]
        # Once memory reaches its explicit cap, do not grant novelty credit
        # for cells we cannot actually remember seeing before.
        fresh_cell=int(cell not in visits and len(visits)<MAX_VISITS_PER_WORLD)
        if cell in visits:
            visits[cell]+=1
        elif len(visits)<MAX_VISITS_PER_WORLD:
            visits[cell]=1
        # Novelty is 2-block spatial discovery, not a goal coordinate.
        visible=self.surfaces[index]
        seen_new=0
        seen_new_wood=0
        visible_wood_count=sum(wood_id(obj.block_id) for obj in surface.objects)
        for obj in surface.objects:
            key=(tuple(obj.xyz),obj.block_id)
            if key not in visible and len(visible)<MAX_SEEN_SURFACES_PER_WORLD:
                visible.add(key)
                seen_new+=1
                seen_new_wood+=int(wood_id(obj.block_id))
        absolute=self.absolute_stats[index]
        new_mined=new_wood_mined=new_picked_wood=0
        for key,value in self._stat_values(stats).items():
            previous=absolute.get(key,0)
            delta=max(0,value-previous)
            absolute[key]=max(previous,value)
            if delta:
                if key[0]=='minecraft:mined':
                    new_mined+=delta
                    if wood_id(key[1]):new_wood_mined+=delta
                elif key[0]=='minecraft:picked_up' and wood_id(key[1]):
                    new_picked_wood+=delta
        logs=self.count_logs(player)
        prior_logs=self.inventory_logs[index]
        inventory_gain=max(logs-prior_logs,0)
        self.inventory_logs[index]=logs
        if type(ticks) is not int or ticks<1 or not all(type(v) is int for v in action):
            raise ValueError('Invalid natural action/tick count')
        self.last_actions[index]=tuple(action)
        self.ticks[index]+=ticks
        # No scripted goal bearing/distance reward. Small curiosity for
        # actually entering unvisited territory, plus first-hit surfaces;
        # real vanilla block events and inventory are distinct external goals.
        if reward_profile=='wood_engagement_v6':
            # V6: genuine player-visible contact, mining and pickup must
            # dominate passive scanning. Both contact and attack are paid
            # separately, once per observed block, in the live loop.
            curiosity=.05*fresh_cell+.002*min(seen_new,6)
            wood_sighting=.08*min(seen_new_wood,6)
            task_reward=(.015*min(new_mined,3)+12.*min(new_wood_mined,3)
                         +40.*min(inventory_gain,3))
        elif reward_profile=='wood_target_v5':
            # V5: make actual wood targeting/breaking/pickup more important
            # than mowing through unrelated terrain. All reward events are
            # from the own-player/camera-visible inputs or actual own stats.
            curiosity=.12*fresh_cell+.008*min(seen_new,6)
            wood_sighting=.40*min(seen_new_wood,6)
            task_reward=(.02*min(new_mined,3)+5.*min(new_wood_mined,3)
                         +20.*min(inventory_gain,3))
        else:
            curiosity=.65*fresh_cell+.018*min(seen_new,6)
            # Wood sighting reward is from actual first-hit CAMERA surfaces
            # only, deduplicated by block identity and position.
            wood_sighting=(.32*min(seen_new_wood,6) if
                            reward_profile=='wood_discovery_v2' else 0.)
            task_reward=.09*min(new_mined,3)+1.8*min(new_wood_mined,3)+7.*min(inventory_gain,3)
        reward=curiosity+wood_sighting+task_reward-.025-(3. if death else 0.)
        record=dict(fresh_cell=fresh_cell,new_visible_surfaces=seen_new,
                    new_visible_wood=seen_new_wood,visible_wood_count=visible_wood_count,
                    new_mined_blocks=new_mined,new_wood_mined=new_wood_mined,
                    new_wood_picked_stat=new_picked_wood,
                    wood_inventory_gain=inventory_gain,
                    wood_inventory=logs,death=bool(death),
                    intrinsic_reward=round(curiosity+wood_sighting,5),
                    extrinsic_reward=round(task_reward,5),
                    reward=round(reward,5))
        return record
