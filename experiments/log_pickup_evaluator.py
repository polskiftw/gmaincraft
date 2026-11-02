"""Independent fresh-world audit of pixel-driven dropped-item pickup policies.

No training occurs. New copied world, every run removes previous item entities,
clears player inventory and restores player pose before summoning an actual
vanilla dropped oak_log. Score is observed inventory only. Controls: random,
forward-only, untrained scratch policy and trained policies from checkpoints.
The fixture item is controller-summoned: this does NOT test natural mining.
"""
import argparse
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
from mcio_ctrl import controller,instance,types,world

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.log_pickup import PickupPolicy
from gmaincraft.log_curriculum import inventory_count
from log_interaction_probe import Client
from log_pickup_learner import DropArena,DropCase,cases_for_seed,run_case,report
DATA=ROOT/'mcio-data'


def load_policy(tag):
    path=ROOT/'checkpoints'/f'drop_pickup_policy_{tag}.npz'
    policy=PickupPolicy(seed=1)
    with np.load(path) as data:
        weights=data['weights'].copy()
        if weights.shape!=policy.weights.shape or not np.isfinite(weights).all():
            raise ValueError(f'Invalid saved weight shape or values at {path}')
        policy.weights=weights
    return policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--models',nargs='+',default=['p01','p02'])
    p.add_argument('--baseline-seed',type=int,default=4401)
    p.add_argument('--eval-seed',type=int,default=8803)
    p.add_argument('--eval',type=int,default=30)
    p.add_argument('--decisions',type=int,default=22)
    p.add_argument('--ticks',type=int,default=5)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',args.run_tag):p.error('Invalid tag')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',m) for m in args.models):
        p.error('Invalid model tag')
    if len(args.models)!=len(set(args.models)):p.error('Duplicate model tag')
    if min(args.eval,args.decisions,args.ticks)<1:p.error('All counts must be positive')
    name='MDL_DropAudit_'+args.run_tag
    save=DATA/'instances'/'DemoInstance'/'saves'/name
    log=ROOT/'logs'/f'drop_pickup_audit_{args.run_tag}.csv'
    if save.exists() or log.exists():p.error('Refusing to overwrite existing audit world or CSV')
    policies=[('untrained',PickupPolicy(seed=args.baseline_seed),'policy'),
        ('random',None,'random'),('forward',None,'forward')]
    for tag in args.models:
        policies.append((tag,load_policy(tag),'policy'))
    print(f'Pickup audit: new copied world={name}, camera scenarios seed={args.eval_seed}, cases={args.eval}',flush=True)
    print('SCORE: actual inventory acquired. Negative control, fixed-forward and random controls; no training.',flush=True)
    world.WorldManager(mcio_dir=DATA).copy('storage','DemoWorld','DemoInstance',name,force=False)
    opts=types.RunOptions(mcio_dir=DATA,instance_name='DemoInstance',world_name=name,
        mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
        gpu_lib=None if args.visible else 'nvidia',width=640 if args.visible else 320,height=360 if args.visible else 180)
    launch=instance.Launcher(opts);ctrl=None
    try:
        launch.launch(wait=False)
        ctrl=controller.ControllerSync(action_port=opts.action_port,observation_port=opts.observation_port,connection_timeout=100)
        client=Client(ctrl)
        arena=DropArena(client,client.step(clear=True));arena.prepare()
        print('NEGATIVE CONTROL: stand still with a newly summoned dropped log...',flush=True)
        ob=arena.reset(DropCase(dx=0,dz=2.5,yaw=0))
        client.advance(25)
        print('NO_ACTION oak_count=',inventory_count(client.last),flush=True)
        if inventory_count(client.last):raise RuntimeError('An item was collected without moving')
        print('POSITIVE CONTROL: walk forward to a fresh dropped log...',flush=True)
        positive=run_case(arena,None,random.Random(0),DropCase(dx=0,dz=2.5,yaw=0),
            decisions=args.decisions,ticks=args.ticks,mode='forward')
        print('FORWARD_POSITIVE',positive,flush=True)
        if positive['outcome']!='success':raise RuntimeError('Fixture item was not collectable')
        cases=cases_for_seed(args.eval_seed,args.eval)
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=['model','case','target_dx','target_dz','yaw',
                'outcome','decisions','reward','final_distance','action_count_w',
                'action_count_a','action_count_d','action_count_s'])
            writer.writeheader()
            for label,model,mode in policies:
                initial_weights=model.weights.copy() if model is not None else None
                rows=[]
                for i,case in enumerate(cases,1):
                    r=run_case(arena,model,random.Random(args.eval_seed+i*13),case,
                        train=False,decisions=args.decisions,ticks=args.ticks,mode=mode)
                    if r['outcome']=='success' and r['final_distance']>2.8:
                        # The controller's fixture origin is not guaranteed to
                        # match a moving item exactly. Exclude implausible
                        # successes rather than silently counting them.
                        print(f'EXCLUDED SUSPECT PICKUP: {label} case={i}, '
                              f'distance_from_original_spawn={r["final_distance"]}',flush=True)
                        r['outcome']='suspect_pickup_excluded'
                    rows.append(r)
                    writer.writerow(dict(model=label,case=i,target_dx=round(case.dx,4),
                        target_dz=round(case.dz,4),yaw=round(case.yaw,3),**r))
                file.flush()
                if model is not None and not np.array_equal(model.weights,initial_weights):
                    raise RuntimeError(f'Evaluator unexpectedly modified {label} weights')
                report(label,rows)
        print('PASS: fresh-world frozen pickup audit completed; item cleanup and actual inventory verified',flush=True)
        print('CSV',log,flush=True)
    finally:
        if ctrl:
            try:ctrl.send_stop()
            except Exception as exc:print('Stop warning',repr(exc),flush=True)
            ctrl.close()
        launch.close()

if __name__=='__main__':main()
