"""No privileged mine/drop oracle; train/eval contract and transition audit."""
import contextlib,io,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
import torch
from gmaincraft.agent_interface import AgentAction
from gmaincraft.natural_pickup_transfer import (
    NaturalPickupTransfer,visible_wood_only,CONTRACT,coarse_audit_heads)
from gmaincraft.mcio_visible_items import VisibleItemsV1,VisibleDroppedItemV1,SCHEMA as ITEM_SCHEMA
from gmaincraft.player_knowledge import OwnedItem,PlayerKnowledgeV1,SCHEMA as PLAYER_SCHEMA
from gmaincraft.natural_diagnostics import NaturalActionAudit
from gmaincraft.numeric_item_pickup_v2 import PickupReading
from fresh_seed_treadmill import parse_args,metadata,load_frozen_pickup_control


def own(x=0.,logs=0,health=20.):
    slots=(OwnedItem(0,'minecraft:oak_log',logs),) if logs else ()
    return PlayerKnowledgeV1(PLAYER_SCHEMA,(x,70.,0.),0.,26.,health,slots,(),(),())


def items(*kinds):
    records=tuple(VisibleDroppedItemV1(kind,(x,70.,3.5+x),4.,(.52,.6),3)
                  for x,kind in enumerate(kinds))
    return VisibleItemsV1(ITEM_SCHEMA,24.,records)


class FakeBrain:
    def __init__(self,action=0):self.action=action;self.calls=0
    def act(self,observation,*,greedy=False):
        self.calls+=1
        assert isinstance(observation,PickupReading)
        assert greedy
        return self.action


class TransferPolicyTests(unittest.TestCase):
    def test_filter_accepts_only_camera_visible_real_wood(self):
        mixed=items('minecraft:oak_log','minecraft:dirt','minecraft:spruce_log')
        filtered=visible_wood_only(mixed)
        self.assertEqual([x.item_id for x in filtered.items],
                         ['minecraft:oak_log','minecraft:spruce_log'])
        with self.assertRaises(TypeError):visible_wood_only(object())

    def test_no_hidden_trigger_and_real_player_sighting_can_start_pickup(self):
        agent=FakeBrain(0)
        pickup=NaturalPickupTransfer(agent,selector='visible_memory_v1')
        pickup.observe(own(),items('minecraft:dirt'),elapsed_ticks=0)
        self.assertIsNone(pickup.choose(24))
        self.assertEqual(agent.calls,0)
        pickup.observe(own(),items('minecraft:oak_log'),elapsed_ticks=6)
        action=pickup.choose(24)
        self.assertIsInstance(action,AgentAction)
        self.assertEqual(action.keys,('W',))
        self.assertEqual(agent.calls,1)
        self.assertEqual(pickup.audit()['own_visible_pickup_engagements'],1)
        self.assertEqual(pickup.audit()['pickup_policy_actions'],1)
        # Own remembered observation allows bounded pursuit with item gone.
        pickup.observe(own(x=.5),items(),elapsed_ticks=30)
        self.assertIsNotNone(pickup.choose(6))
        # Own inventory acquisition resets prior targets without hidden stats.
        pickup.observe(own(x=.5,logs=1),items(),elapsed_ticks=36)
        self.assertIsNone(pickup.choose(6))
        self.assertEqual(pickup.audit()['inventory_gains_seen'],1)
        self.assertEqual(pickup.audit()['contract'],CONTRACT)

    def test_shadow_never_takes_control_and_stale_target_expires(self):
        brain=FakeBrain()
        shadow=NaturalPickupTransfer(brain,selector='shadow')
        shadow.observe(own(),items('minecraft:oak_log'),elapsed_ticks=0)
        self.assertIsNone(shadow.choose(6))
        self.assertEqual(brain.calls,0)
        active=NaturalPickupTransfer(FakeBrain(),selector='visible_memory_v1')
        active.observe(own(),items('minecraft:oak_log'),elapsed_ticks=0)
        self.assertIsNotNone(active.choose(6))
        active.observe(own(),items(),elapsed_ticks=120)
        self.assertIsNone(active.choose(6))
        with self.assertRaises(ValueError):active.choose(0)
        with self.assertRaises(ValueError):active.observe(own(),items(),elapsed_ticks=100)

    def test_audit_counts_unfactored_motor_without_fabricating_q_heads(self):
        action=AgentAction(keys=('W','A'),ticks=12)
        self.assertEqual(coarse_audit_heads(action),(1,0,0))
        from types import SimpleNamespace
        observed=SimpleNamespace(own=[0.]*23)
        audit=NaturalActionAudit()
        audit.observe_primitive(action,observed,None)
        audit.record_duration(1,12,attack=False,wood_crosshair=False)
        result=audit.summary()
        self.assertEqual(result['unfactored_primitive_actions'],1)
        self.assertEqual(result['action_elapsed_ticks'],12)
        self.assertEqual(sum(result['action_counts'][0]),0)

    def test_strict_eval_optin_only_and_metadata_version(self):
        def args(*flags):
            return parse_args(['--run-tag','transfer', '--mode','eval',
                  '--control','q_sampled','--q-model','qv2memf1104',
                  '--q-goal','effect:visible_wood_crack_progress_v2',
                  '--reward-profile','wood_temporal_v9',*flags])
        old=args()
        self.assertNotIn('pickup_model',metadata(old))
        new=args('--pickup-model','hardrep1105b',
                 '--pickup-selector','visible_memory_v1')
        self.assertEqual(metadata(new)['pickup_transfer_contract'],CONTRACT)
        self.assertNotEqual(metadata(new),metadata(old))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):args('--pickup-selector','visible_memory_v1')
            with self.assertRaises(SystemExit):parse_args(['--run-tag','badtrain',
                  '--pickup-model','hardrep1105b'])
            with self.assertRaises(SystemExit):args('--pickup-model','../bad')

    def test_pickup_model_loading_does_not_perturb_mining_q_rng(self):
        args=parse_args(['--run-tag','transfer2','--mode','eval',
            '--control','q_sampled','--q-model','qv2memf1104',
            '--q-goal','effect:visible_wood_crack_progress_v2',
            '--reward-profile','wood_temporal_v9','--pickup-model','hardrep1105b',
            '--device','cpu'])
        # This uses an actual frozen on-disk model, never a synthetic teacher.
        path=ROOT/'checkpoints'/'numeric_pickup_hardrep1105b.pt'
        if not path.exists():self.skipTest('Local ignored real training checkpoint not mounted')
        torch.manual_seed(9811)
        before=torch.get_rng_state().clone()
        net=load_frozen_pickup_control(ROOT,args)
        self.assertFalse(net.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
