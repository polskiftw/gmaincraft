"""Frozen V2 model audit on SAME held-out Minecraft cases, real inventory.

No optimization, new copied flat world, safe reset after every episode,
independent from all existing ignored worlds/checkpoints. Only pixel history
enters trained policy. Held-out near cases use the same seed convention as
factorized_controls.py for a meaningful baseline comparison.

The restricted attack-only check DISABLES movement/look and is labeled as a
diagnostic, never as a fair full-policy score or natural-world transfer.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.joint_curriculum import sample_cases,sample_near_cases,sample_stationary_cases
from gmaincraft.vision_factorized import FactorizedPPOTrainer,action_schema
from joint_log_factorized import run_episode
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'
PHASES={'near':(sample_near_cases,(True,True,True),0),
        'broad':(sample_cases,(True,True,True),1),
        'stationary_restricted':(sample_stationary_cases,(False,False,True),2),
        'aim_restricted':(sample_near_cases,(False,True,True),3)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--phases',nargs='+',choices=list(PHASES),default=['near','stationary_restricted'])
    parser.add_argument('--eval-seed',type=int,default=3737)
    parser.add_argument('--cases',type=int,default=12)
    parser.add_argument('--max-ticks',type=int,default=360)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    parser.add_argument('--frame-width',type=int,default=320)
    parser.add_argument('--frame-height',type=int,default=180)
    parser.add_argument('--visible',action='store_true')
    args=parser.parse_args(argv)
    tags=[args.run_tag,*args.models]
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',t) for t in tags):
        parser.error('Invalid tag')
    if args.cases<1 or args.max_ticks<1:parser.error('Invalid counts')
    if len(args.models)!=len(set(args.models)):parser.error('Repeated model tags')
    output=ROOT/'logs'/f'factorized_frozen_{args.run_tag}.jsonl'
    save=MCIO/'instances'/'DemoInstance'/'saves'/f'MDL_FactorAudit_{args.run_tag}'
    if output.exists() or save.exists():parser.error('Refusing overwrite')
    if not 96<=min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height)<=1920:
        parser.error('Frame dimensions 96..1920 only')
    models=[]
    for tag in args.models:
        path=ROOT/'checkpoints'/f'joint_factorized_{tag}.pt'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        trainer=FactorizedPPOTrainer(device=args.device,seed=0)
        meta=trainer.load_weights(path)
        decision_ticks=int(meta.get('decision_ticks',6))
        if meta.get('schema')!=action_schema(decision_ticks):
            raise ValueError(f'Incompatible {tag} schema')
        models.append((tag,digest,trainer,decision_ticks))
    cases={phase:PHASES[phase][0](args.eval_seed+PHASES[phase][2],args.cases)
           for phase in args.phases}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as out:
        out.write(json.dumps(dict(stage='config',seed=args.eval_seed,cases=args.cases,
                                  max_ticks=args.max_ticks,models=args.models,phases=args.phases))+'\n')
        out.flush()
        def evaluate(arena):
            totals=Counter()
            for phase in args.phases:
                mask=PHASES[phase][1]
                for i,case in enumerate(cases[phase],1):
                    for tag,digest,trainer,decision_ticks in models:
                        result,_=run_episode(arena,trainer,case,max_ticks=args.max_ticks,
                                              training=False,active_heads=mask,
                                              decision_ticks=decision_ticks)
                        totals[tag+'_'+phase]+=int(result['outcome']=='success')
                        out.write(json.dumps(dict(stage='eval',model=tag,model_sha256=digest,
                                                  phase=phase,case=i,active_heads=mask,
                                                  decision_ticks=decision_ticks,
                                                  case_data=case.__dict__,**result))+'\n')
                        out.flush()
                        print(f'{phase} {i}/{len(cases[phase])} {tag} outcome={result["outcome"]} '
                              f'attack_ticks={result["attack_ticks"]} '
                              f'end_dist={result["end_distance"]} '
                              f'best_angle={result["best_angle"]}',flush=True)
            return totals
        scores=launch_world(f'MDL_FactorAudit_{args.run_tag}',args,evaluate)
        print('FROZEN SCORES:',dict(scores),'cases per model per phase',args.cases,flush=True)
    print('LOCAL JSONL:',output,flush=True)

if __name__=='__main__':main()
