"""Separate-process frozen held-out Minecraft evaluation for V4 memory+RGB.

Use a FRESH unique --run-tag. Loads a saved local checkpoint and enforces its
player-state/RGB schema; never trains or resumes weights. Running evaluation
in its own Gamescope process avoids the occasional nested swapchain assertion
when tearing down one Minecraft and launching a second in the same viewer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.memory_factorized import SCHEMA,FEATURE_NAMES,MemoryFactorizedPPOTrainer
from gmaincraft.status_factorized import (StatusFactorizedPPOTrainer,
    SCHEMA as STATUS_SCHEMA,FEATURE_NAMES as STATUS_FEATURES,
    BLOCK_TOKEN_LENGTH,BLOCK_TOKENIZER)
from gmaincraft.joint_curriculum import sample_cases,sample_near_cases,sample_stationary_cases
from joint_log_factorized import run_episode
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--model-tag',required=True)
    ap.add_argument('--eval-seed',type=int,default=311037)
    ap.add_argument('--eval',type=int,default=4,help='cases per evaluation phase')
    ap.add_argument('--max-ticks',type=int,default=240)
    ap.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    ap.add_argument('--frame-width',type=int,default=320)
    ap.add_argument('--frame-height',type=int,default=180)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--with-status',action='store_true',
                    help='evaluate opt-in V5 patched-MCio crosshair/hunger checkpoint')
    args=ap.parse_args(argv)
    for tag in (args.run_tag,args.model_tag):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',tag):
            ap.error('Invalid model/run tag')
    if not 1<=args.eval<=1000 or not 10<=args.max_ticks<=72000:
        ap.error('Invalid eval or game tick count')
    if not 96<=min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height)<=1920:
        ap.error('Invalid viewport')
    return args


def main(argv=None):
    args=parse_args(argv)
    family='status' if args.with_status else 'memory'
    cp=ROOT/'checkpoints'/f'joint_{family}_{args.model_tag}.pt'
    log=ROOT/'logs'/f'{family}_frozen_{args.run_tag}.jsonl'
    world_name=('MDL_StatusFrozen_' if args.with_status else 'MDL_MemoryFrozen_')+args.run_tag
    dest=MCIO/'instances'/'DemoInstance'/'saves'/world_name
    if log.exists() or dest.exists(): raise FileExistsError('Choose unused tag; output already exists')
    if not cp.is_file(): raise FileNotFoundError(f'Model checkpoint missing {cp}')
    agent=(StatusFactorizedPPOTrainer if args.with_status else
           MemoryFactorizedPPOTrainer)(device=args.device,seed=0)
    metadata=agent.load_weights(cp)
    schema=STATUS_SCHEMA if args.with_status else SCHEMA
    feature_names=STATUS_FEATURES if args.with_status else FEATURE_NAMES
    if metadata.get('schema')!=schema or tuple(metadata.get('feature_names',()))!=feature_names:
        raise ValueError('Refusing incompatible memory feature / model schema')
    if args.with_status and (metadata.get('status_schema')!='gmaincraft_status_v1' or
                             metadata.get('block_tokenizer')!=BLOCK_TOKENIZER or
                             int(metadata.get('block_token_length',-1))!=BLOCK_TOKEN_LENGTH):
        raise ValueError('Refusing incompatible V5 block tokenization and sensor schema')
    if int(metadata.get('decision_ticks',-1))!=6:
        raise ValueError('Incompatible Minecraft action duration')
    opts=dict(metadata.get('game_options') or {})
    digest=hashlib.sha256(cp.read_bytes()).hexdigest()
    scenarios=(('stationary_restricted',sample_stationary_cases(args.eval_seed+2,args.eval),(False,False,True)),
               ('aim_restricted',sample_near_cases(args.eval_seed+3,args.eval),(False,True,True)),
               ('near_unrestricted',sample_near_cases(args.eval_seed,args.eval),(True,True,True)),
               ('broad_unrestricted',sample_cases(args.eval_seed+1,args.eval),(True,True,True)))
    label='FROZEN V5' if args.with_status else 'FROZEN V4'
    print(f'{label} MODEL tag={args.model_tag} sha256={digest} '
          f'observation={schema} game_options={opts}',flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with temporary_game_options(MCIO/'instances'/'DemoInstance',opts,run_tag=args.run_tag), \
            log.open('x',encoding='utf-8') as output:
        output.write(json.dumps(dict(stage='config',run_tag=args.run_tag,model_tag=args.model_tag,
            checkpoint_sha256=digest,game_options=opts,model_metadata=metadata,
            eval_seed=args.eval_seed,eval_cases=args.eval,max_ticks=args.max_ticks))+'\n')
        output.flush()
        def scored(arena):
            counts={}
            for phase,cases,mask in scenarios:
                n=0
                for i,case in enumerate(cases,1):
                    result,_=run_episode(arena,agent,case,max_ticks=args.max_ticks,
                                         training=False,active_heads=mask,decision_ticks=6)
                    result['io_ms_per_tick']=round(result['game_io_wall_s']*1000/result['ticks'],3)
                    result['checkpoint_sha256']=digest
                    n+=int(result['outcome']=='success')
                    output.write(json.dumps(dict(stage='eval',phase=phase,case=i,**result))+'\n')
                    output.flush()
                    print(f'FROZEN {phase} {i}/{len(cases)} {result["outcome"]} '
                          f'mined={result["mined_oak_events"]} '
                          f'io_ms/tick={result["io_ms_per_tick"]}',flush=True)
                counts[phase]=dict(success=n,total=len(cases))
            return counts
        print('SCORED START',flush=True)
        started=time.perf_counter()
        outcome=launch_world(world_name,args,scored)
        print(label+' RESULTS',outcome,'total_seconds',round(time.perf_counter()-started,2),flush=True)
    print(label+' LOG',log,flush=True)

if __name__=='__main__': main()
