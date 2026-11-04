"""Pure parser/bridge tests, no runtime world enumeration or Minecraft."""
import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,AgentAction
from gmaincraft.mcio_visible_items import (
    VisibleItemsV1,VisibleItemHit,VisibleItemsOption,VisibleItemsRequest,SCHEMA)


class Packet:
    version=7
    cursor_pos=(0.,0.)
    def __init__(self,opt=None):
        self.opt=opt
    def get_option(self,key):
        return self.opt if key is VisibleItemsOption else None


class VisibleItemSensorTest(unittest.TestCase):
    def test_empty_distinct_from_missing_and_reject_hidden_untested_data(self):
        self.assertEqual(VisibleItemsV1.from_packet(Packet(VisibleItemsOption(
            SCHEMA,24.,[]))).items,())
        with self.assertRaisesRegex(RuntimeError,'Missing same-action'):
            VisibleItemsV1.from_packet(Packet())
        hit=VisibleItemHit('minecraft:oak_log',[0.,64.,2.],2.3,.5,.5,3)
        option=VisibleItemsOption(SCHEMA,24.,[hit])
        record=VisibleItemsV1.from_packet(Packet(option)).items[0]
        self.assertEqual(record.item_id,'minecraft:oak_log')
        self.assertEqual(record.visible_samples,3)
        self.assertEqual(record.screen_uv,(.5,.5))
        self.assertEqual(record.xyz,(0.,64.,2.))
        for bad in (
            VisibleItemHit('minecraft:oak_log',[0.,64.,2.],2.3,.5,.5,0),
            VisibleItemHit('minecraft:oak_log',[0.,64.,2.],2.3,.5,.5,6),
            VisibleItemHit('minecraft:oak_log',[0.,64.,2.],2.3,10.,.5,1),
            VisibleItemHit('minecraft:oak_log',[0.,64.,float('nan')],2.3,.5,.5,1),
            VisibleItemHit('minecraft:oak_log',[0.,64.,2.],float('inf'),.5,.5,1),
            VisibleItemHit('/hidden', [0.,64.,2.],2.3,.5,.5,1),
        ):
            with self.subTest(hit=bad),self.assertRaises(ValueError):
                VisibleItemsV1.from_packet(Packet(VisibleItemsOption(SCHEMA,24.,[bad])))
        with self.assertRaises(ValueError):
            VisibleItemsV1.from_packet(Packet(VisibleItemsOption(
                SCHEMA,24.,[hit]*17)))
        with self.assertRaises(ValueError):
            VisibleItemsV1.from_packet(Packet(VisibleItemsOption(
                SCHEMA,24.,[VisibleItemHit('minecraft:dirt',[0.,1.,3.],4.,.5,.5,1),hit])))

    def test_bridge_requests_fresh_only_on_last_action_tick(self):
        sent=[]
        pkt=Packet(VisibleItemsOption(SCHEMA,24.,[]))
        bridge=object.__new__(MCioAgentBridge)
        bridge.packet=pkt
        bridge.input_map={'W':1}
        bridge.input_manager=SimpleNamespace(process_action=lambda keys,lookup:[])
        bridge.controller=SimpleNamespace(
            send_action=lambda x:sent.append(x),recv_observation=lambda:pkt)
        bridge._stats=SimpleNamespace(update_cache=lambda x:None)
        self.assertIsNone(bridge.apply(AgentAction(ticks=3),request_visible_items=True,
                                       return_view=False))
        self.assertEqual(len(sent),3)
        self.assertEqual(len(sent[0].options),0)
        self.assertEqual(len(sent[1].options),0)
        self.assertEqual(len(sent[2].options),1)
        self.assertIsInstance(sent[2].options[0],VisibleItemsRequest)
        self.assertEqual(bridge.visible_items().items,())
        sent.clear()
        bridge.apply(AgentAction(ticks=1),return_view=False)
        self.assertEqual(sent[0].options,[])


if __name__=='__main__':
    unittest.main()
