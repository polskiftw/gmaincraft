#!/usr/bin/env python3
"""Verify paired natural Minecraft Q-only / visibility-gated pickup evaluation.

Both modes use EXACT same Q network, visible item packet, policy RNG,
vanilla world seeds and horizons. The only requested difference is a
transparent NONLEARNED visibility-gated pickup motor override. This is
NOT a test of autonomous learned skill selection nor a causal confidence
bound. Report inventory and dropped-item opportunity before activity.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(tag):
    path=ROOT/'logs'/f'fresh_eval_{tag}.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows or rows[0].get('stage')!='config' or rows[-1].get('stage')!='completed':
        raise ValueError(f'{tag}: incomplete heldout Minecraft evaluation')
    cfg=rows[0];last=rows[-1]
    es=[r for r in rows if r.get('stage')=='episode_committed']
    if (cfg.get('mode')!='eval' or cfg.get('evaluation_control')!='q_sampled' or
        cfg.get('q_goal')!='effect:visible_wood_crack_progress_v2' or
        cfg.get('metadata',{}).get('pickup_transfer_contract')!='player_seen_dropped_wood_pickup_transfer_v1' or
        not cfg.get('metadata',{}).get('pickup_model') or
        cfg.get('metadata',{}).get('gameplay_commands')!=0 or
        last.get('mode')!='eval' or last.get('new_optimizer_updates_in_eval')!=0 or
        len(es)!=last.get('evaluated_episodes') or
        sum(e.get('decisions',0) for e in es)!=last.get('decisions') or
        any(e['episode']!=cfg['evaluation_seed_start']+i or
            e['updates']!=0 or
            e.get('pickup_transfer_audit',{}).get('contract')!='player_seen_dropped_wood_pickup_transfer_v1'
            for i,e in enumerate(es))):
        raise ValueError(f'{tag}: sensor/motor/evaluation contract invalid')
    return cfg,es,last


def compare(shadow,active):
    control=read(shadow);switch=read(active)
    cm,sm=control[0]['metadata'],switch[0]['metadata']
    if cm.get('pickup_selector')!='shadow' or sm.get('pickup_selector')!='visible_memory_v1':
        raise ValueError('Order must be mining-only shadow THEN pickup transfer')
    if ({k:v for k,v in cm.items() if k!='pickup_selector'}!=
        {k:v for k,v in sm.items() if k!='pickup_selector'}):
        raise ValueError('Baseline and transfer used different world/sensor/actor contracts')
    if (control[0]['q_model']!=switch[0]['q_model'] or
        control[0]['q_goal']!=switch[0]['q_goal'] or
        control[0]['evaluation_seed_start']!=switch[0]['evaluation_seed_start'] or
        control[0]['evaluation_control']!=switch[0]['evaluation_control']):
        raise ValueError('Different Q actor/world index/action control')
    es0,es1=control[1],switch[1]
    if (len(es0)!=len(es1) or
        any(a['seed']!=b['seed'] or a['decisions']!=b['decisions']
            for a,b in zip(es0,es1))):
        raise ValueError('Not real matched Minecraft seed/time horizon evaluations')
    poses=[]
    for a,b in zip(es0,es1):
        pa,pb=a.get('first_player_pose'),b.get('first_player_pose')
        if pa is None and pb is None:
            poses.append(None)  # Older smoke lacked pose-level matching.
        elif not (isinstance(pa,list) and isinstance(pb,list) and
                  len(pa)==5 and len(pb)==5):
            raise ValueError('Missing pose in one of the paired Minecraft worlds')
        elif any(abs(x-y)>0.13 for x,y in zip(pa,pb)):
            raise ValueError(f'Minecraft initial spawn pose differs on eval index {a["episode"]}: {pa} vs {pb}')
        else:
            poses.append(True)
    def summarize(name,events):
        r=dict(mode=name,worlds=len(events),wood_gained=0,wood_mined=0,
               positive_worlds=0,deaths=0,drop_visible_decisions=0,
               pickup_engagements=0,pickup_actions=0,
               pickups_during_pickup_actions=0,first_seen_worlds=0)
        for e in events:
            c=e['counts'];p=e['pickup_transfer_audit'];g=c.get('wood_inventory_gain',0)
            r['wood_gained']+=g;r['wood_mined']+=c.get('new_wood_mined',0)
            r['positive_worlds']+=int(g>0);r['deaths']+=c.get('death',0)
            r['drop_visible_decisions']+=p.get('visible_wood_item_observations',0)
            r['first_seen_worlds']+=int(p.get('seen_opportunities',0)>0)
            r['pickup_engagements']+=p.get('own_visible_pickup_engagements',0)
            r['pickup_actions']+=p.get('pickup_policy_actions',0)
            r['pickups_during_pickup_actions']+=c.get('own_log_gain_during_pickup_motor',0)
        return r
    a=summarize(shadow,es0);b=summarize(active,es1)
    if a['pickup_actions'] or a['pickup_engagements']:
        raise ValueError('Shadow control secretly executed pickup actions')
    if any(e['pickup_transfer_audit'].get('pickup_policy_actions',0)!=
           e['action_audit'].get('unfactored_primitive_actions',0) for e in es1):
        raise ValueError('Pickup action audit does not match actual game motor count')
    result=dict(schema='natural_pickup_transfer_eval_v1',
                q_model=control[0]['q_model'],pickup_model=cm['pickup_model'],
                start_index=control[0]['evaluation_seed_start'],
                q_action_rng=cm['policy_seed'],
                seed_root=cm['seed_root'],
                units_per_world=cm['episode_decisions'],
                pose_matched_worlds=sum(v is True for v in poses),
                historical_pose_unverified=sum(v is None for v in poses),
                shadow=a,visible_memory=b,
                world_comparison=[dict(index=e0['episode'],seed=e0['seed'],
                    shadow_wood=e0['counts'].get('wood_inventory_gain',0),
                    active_wood=e1['counts'].get('wood_inventory_gain',0),
                    shadow_visible=e0['pickup_transfer_audit'].get('seen_opportunities',0),
                    active_visible=e1['pickup_transfer_audit'].get('seen_opportunities',0),
                    active_pickup_actions=e1['pickup_transfer_audit'].get('pickup_policy_actions',0))
                    for e0,e1 in zip(es0,es1)],
                warning='Visibility-gated handoff is NOT learned; inventory success not causal effect of takeover')
    print('NATURAL TRANSFER REPORT',json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('shadow_tag');p.add_argument('active_tag')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=compare(args.shadow_tag,args.active_tag)
    if args.output:
        if args.output.exists():p.error(f'Refusing to overwrite existing report {args.output}')
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
