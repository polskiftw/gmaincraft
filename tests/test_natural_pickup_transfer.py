"""No privileged mine/drop oracle; train/eval contract and transition audit."""
import contextlib,io,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
import torch
import numpy as np
from gmaincraft.agent_interface import AgentAction
from gmaincraft.natural_pickup_transfer import (
    NaturalPickupTransfer,visible_wood_only,CONTRACT,BUDGET_CONTRACT,
    BUDGET_ATTEMPT_TICKS,OPTION_CONTRACT,contract_for_selector,coarse_audit_heads)
from gmaincraft.mcio_visible_items import VisibleItemsV1,VisibleDroppedItemV1,SCHEMA as ITEM_SCHEMA
from gmaincraft.player_knowledge import OwnedItem,PlayerKnowledgeV1,SCHEMA as PLAYER_SCHEMA
from gmaincraft.natural_diagnostics import NaturalActionAudit
from gmaincraft.numeric_item_pickup_v2 import PickupReading
from fresh_seed_treadmill import (parse_args,metadata,load_frozen_pickup_control,
    load_frozen_natural_option_control)
from gmaincraft.natural_online_policy import NaturalObservation,TEMPORAL_V9_SELF_FEATURES
from gmaincraft.natural_option_q_v1 import NaturalOptionQ,SCHEMA as OPTION_SCHEMA


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

    def test_new_version_budget_quarantines_stubborn_visible_log_without_teacher(self):
        agent=FakeBrain(0)
        p=NaturalPickupTransfer(agent,selector='visible_memory_budget_v2')
        for tick in range(0,186,6):
            p.observe(own(),items('minecraft:oak_log'),elapsed_ticks=tick)
            move=p.choose(6)
            if tick<BUDGET_ATTEMPT_TICKS:
                self.assertIsNotNone(move)
            else:
                self.assertIsNone(move)
        self.assertEqual(agent.calls,BUDGET_ATTEMPT_TICKS//6)
        self.assertEqual(p.audit()['pickup_effort_budget_exhaustions'],1)
        self.assertEqual(p.audit()['contract'],BUDGET_CONTRACT)
        p.observe(own(),items('minecraft:birch_log'),elapsed_ticks=192)
        self.assertIsNotNone(p.choose(6))
        # A real own inventory gain resets the blocked item, no oracle needed.
        p.observe(own(logs=1),items('minecraft:oak_log'),elapsed_ticks=198)
        self.assertIsNotNone(p.choose(6))
        shadow=NaturalPickupTransfer(agent,selector='shadow_budget_v2')
        shadow.observe(own(),items('minecraft:oak_log'),elapsed_ticks=0)
        self.assertIsNone(shadow.choose(6))
        self.assertEqual(shadow.audit()['contract'],BUDGET_CONTRACT)
        self.assertNotEqual(contract_for_selector('shadow'),BUDGET_CONTRACT)

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

    def test_genuinely_learned_option_uses_model_choice_not_item_visibility_gate(self):
        class Option:
            def __init__(self):self.choice=0;self.calls=0
            def act(self,features,*,greedy=False):
                self.calls+=1
                self.asserted=features.shape[0]>15
                assert greedy
                return self.choice
        option=Option()
        pickup=NaturalPickupTransfer(FakeBrain(1),selector='learned_option_v1',
                                     option_brain=option)
        miner=NaturalObservation(np.zeros(len(TEMPORAL_V9_SELF_FEATURES),np.float32),
            np.zeros(48,np.uint8),np.zeros((24,48),np.uint8),
            np.zeros((24,10),np.float32),np.zeros(24,np.float32))
        pickup.observe(own(),items(),elapsed_ticks=0,mining_observation=miner)
        self.assertIsNone(pickup.choose(24))
        option.choice=1
        # No item is visible. A TRAINED option model can still choose its
        # pretrained pickup SEARCH motor; there is no if-visible script gate.
        action=pickup.choose(24)
        self.assertIsInstance(action,AgentAction)
        self.assertEqual(action.keys,('W',))
        self.assertEqual(action.ticks,24)
        self.assertEqual(option.calls,2)
        self.assertTrue(option.asserted)
        self.assertEqual(pickup.audit()['contract'],OPTION_CONTRACT)
        self.assertEqual(pickup.audit()['learned_pickup_option_selections'],1)
        shadow=NaturalPickupTransfer(FakeBrain(),selector='shadow_learned_option_v1',
                                     option_brain=option)
        shadow.observe(own(),items('minecraft:oak_log'),elapsed_ticks=0,
                       mining_observation=miner)
        self.assertIsNone(shadow.choose(6))
        self.assertEqual(option.calls,2)
        with self.assertRaises(ValueError):NaturalPickupTransfer(
            FakeBrain(),selector='learned_option_v1')
        with self.assertRaises(ValueError):pickup.observe(own(),items(),elapsed_ticks=6)

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
