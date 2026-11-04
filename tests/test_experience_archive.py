"""No game launch: player-only archived transitions, durability and goals."""
from dataclasses import replace
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.experience_archive import (
    SCHEMA,EpisodeExperience,ObservedItemGoal,owned_inventory,observed_new_goals,
    archive_path,read_episode,hindsight_goal_transitions,reconcile_uncommitted,compact_retention)
from gmaincraft.natural_online_policy import NaturalObservation,MOTOR_V7_SCHEMA
from gmaincraft.player_knowledge import OwnedItem,PlayerKnowledgeV1


def player(items=()):
    return PlayerKnowledgeV1('player_knowledge_v1',(0.,64.,0.),0.,0.,20.,
                             tuple(OwnedItem(i,ident,count) for i,(ident,count) in enumerate(items)),
                             (),(),())


def observation():
    arrays=[np.zeros(31,np.float32),np.zeros(48,np.uint8),np.zeros((24,48),np.uint8),
            np.zeros((24,10),np.float32),np.zeros(24,np.float32)]
    for a in arrays:a.setflags(write=False)
    return NaturalObservation(*arrays)


class ArchiveTests(unittest.TestCase):
    def test_owned_goals_are_generic_and_observed_only(self):
        p=player((('minecraft:oak_log',2),('minecraft:oak_log',3),('minecraft:stone',1)))
        self.assertEqual(owned_inventory(p),{'minecraft:oak_log':5,'minecraft:stone':1})
        goals=observed_new_goals({'minecraft:oak_log':4,'minecraft:stone':5},
                                 {'minecraft:oak_log':5,'minecraft:stone':3,'minecraft:bread':2})
        self.assertEqual([(x.item_id,x.increase) for x in goals],
                         [('minecraft:bread',2),('minecraft:oak_log',1)])
        self.assertTrue(ObservedItemGoal('minecraft:bread',2).achieved({}, {'minecraft:bread':2}))
        self.assertFalse(ObservedItemGoal('minecraft:bread',2).achieved({}, {'minecraft:bread':1}))
        with self.assertRaises(ValueError):ObservedItemGoal('../foo')

    def test_atomic_episode_roundtrip_and_hindsight(self):
        with tempfile.TemporaryDirectory() as temp:
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            state=observation()
            before=player()
            mid=player((('minecraft:stone',1),))
            after=player((('minecraft:stone',1),('minecraft:birch_log',1)))
            a.add(state,(1,),state,before,mid,duration=24,reward=.1,done=False)
            goals=a.add(state,(14,),state,mid,after,duration=72,reward=4.,done=False)
            self.assertEqual(goals,(ObservedItemGoal('minecraft:birch_log',1),))
            info=a.save(temp,'newrun',0)
            self.assertEqual(info['steps'],2)
            meta,data=read_episode(archive_path(temp,'newrun',0))
            self.assertEqual(meta['total_game_ticks'],96)
            self.assertEqual(meta['observation_schema'],MOTOR_V7_SCHEMA)
            self.assertEqual(data['action'].tolist(),[[1],[14]])
            self.assertEqual(len(data['before_surface_ids']),2)
            labels=list(hindsight_goal_transitions(meta,data))
            self.assertEqual(len(labels),3)  # stone in first step; birch log for two steps
            self.assertEqual(labels[-1]['goal_item_id'],'minecraft:birch_log')
            self.assertEqual(labels[-1]['ticks_to_achievement'],72)
            self.assertEqual(labels[-2]['ticks_to_achievement'],96)
            with self.assertRaises(FileExistsError):a.save(temp,'newrun',0)
            self.assertEqual(reconcile_uncommitted(temp,'newrun',1),[])

    def test_replay_must_be_player_knowable_and_ordinary_actions(self):
        obs=observation()
        a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
        for act in ((20,),(),(False,),('5',)):
            with self.assertRaises(ValueError):
                a.add(obs,act,obs,player(),player(),duration=6,reward=0.,done=False)
        with self.assertRaises(ValueError):
            a.add(obs,(1,),obs,player(),player(),duration=0,reward=0.,done=False)
        with self.assertRaises(TypeError):
            a.add('raw world',(1,),obs,player(),player(),duration=6,reward=0.,done=False)
        with self.assertRaises(ValueError):
            archive_path('data-local','../secret',0)

    def test_reconcile_preserves_committed_and_quarantines_orphans(self):
        with tempfile.TemporaryDirectory() as temp:
            obs=observation()
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            a.add(obs,(1,),obs,player(),player(),duration=24,reward=0.,done=False)
            for idx in range(3):a.save(temp,'runx',idx)
            orphan=archive_path(temp,'runx',3).with_suffix('.npz.pending')
            orphan.write_bytes(b'partial')
            renamed=reconcile_uncommitted(temp,'runx',2)
            self.assertEqual(len(renamed),2)
            self.assertTrue(archive_path(temp,'runx',0).exists())
            self.assertTrue(archive_path(temp,'runx',1).exists())
            self.assertFalse(archive_path(temp,'runx',2).exists())
            self.assertTrue(pathlib.Path(renamed[0]).exists())
            self.assertTrue(pathlib.Path(renamed[1]).exists())

    def test_retention_prunes_oldest_only(self):
        with tempfile.TemporaryDirectory() as temp:
            obs=observation()
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            a.add(obs,(1,),obs,player(),player(),duration=24,reward=0.,done=False)
            paths=[archive_path(temp,'runx',i) for i in range(3)]
            for idx in range(3):a.save(temp,'runx',idx)
            size=sum(p.stat().st_size for p in paths)
            deleted=compact_retention(temp,'runx',max(1_000_000,size-1))
            # Files normally fit in <1MB, so exercising deletion needs larger fixtures.
            self.assertEqual(len(deleted),0)
            with self.assertRaises(ValueError):compact_retention(temp,'runx',10)


if __name__=='__main__':unittest.main()
