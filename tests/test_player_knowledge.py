"""Pure contract tests: the new structured input must not leak world secrets."""
import pathlib
import sys
import unittest
from dataclasses import fields
from types import SimpleNamespace

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.player_knowledge import PlayerKnowledgeV1, SCHEMA


class TestKnowledge(unittest.TestCase):
    def packet(self):
        return SimpleNamespace(player_pos=(10.5,64.0,-2.5), player_yaw=92,
            player_pitch=-15,health=17.5,
            inventory_main=[SimpleNamespace(slot=0,id='minecraft:oak_log',count=2),
                            SimpleNamespace(slot=4,id='minecraft:air',count=0)],
            inventory_armor=[],inventory_offhand=[],world_seed=112233,
            other_players=[{'x':5}],hidden_blocks=[{'id':'diamond_ore'}],
            goal=(123,45,67))

    def test_player_fields_and_defensive_immutable_snapshot(self):
        packet=self.packet()
        raw={'minecraft:mined':{'minecraft:oak_log':2},
             'minecraft:picked_up':{'minecraft:oak_log':2},
             'minecraft:custom':{'minecraft:secret':999}}
        view=PlayerKnowledgeV1.from_packet(packet,stats=raw)
        self.assertEqual(view.schema,SCHEMA)
        self.assertEqual(view.player_xyz,(10.5,64.0,-2.5))
        self.assertEqual(view.main_inventory[0].count,2)
        self.assertEqual(view.progress_stats,(('minecraft:mined','minecraft:oak_log',2),
                                              ('minecraft:picked_up','minecraft:oak_log',2)))
        self.assertFalse(any(hasattr(view,key) for key in ('world_seed','hidden_blocks',
                                                          'other_players','goal','packet')))
        self.assertEqual(tuple(f.name for f in fields(view)),
            ('schema','player_xyz','yaw_degrees','pitch_degrees','health',
             'main_inventory','armor_inventory','offhand_inventory','progress_stats'))
        packet.inventory_main[0].count=99
        raw['minecraft:mined']['minecraft:oak_log']=999
        self.assertEqual(view.main_inventory[0].count,2)
        self.assertEqual(view.progress_stats[0][-1],2)

    def test_stats_unknown_and_invalid_packets(self):
        packet=self.packet()
        self.assertEqual(PlayerKnowledgeV1.from_packet(packet).progress_stats,())
        packet.player_yaw=float('nan')
        with self.assertRaises(ValueError):PlayerKnowledgeV1.from_packet(packet)
        packet.player_yaw=0
        packet.inventory_main[0].count=-1
        with self.assertRaises(ValueError):PlayerKnowledgeV1.from_packet(packet)

    def test_bounded_stats_budget(self):
        packet=self.packet()
        stats={'minecraft:mined':{f'minecraft:dummy_{i}':i for i in range(513)}}
        with self.assertRaises(ValueError):PlayerKnowledgeV1.from_packet(packet,stats=stats)

if __name__=='__main__':unittest.main()
