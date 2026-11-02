"""Audited frozen evaluation of RGB log-aiming models in a FRESH world.

Each trial reset clears all previous item drops before replacing target logs.
One negative control walks to the log WITHOUT clicking; it must collect zero.
Policies then choose mouse moves and ATTACK from screenshot-derived features;
only Minecraft-observed inventory counts as success. Zero learning occurs.
"""
import argparse
import csv
import random
import re
import sys
from pathlib import Path
import numpy as np
from mcio_ctrl import controller,instance,network,types,world
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.log_vision import VisualAimPolicy
from gmaincraft.log_curriculum import inventory_count
from log_interaction_probe import Client
from log_visual_aim_learner import Arena,run_case,make_cases,report,wrap_yaw
DATA=ROOT/'mcio-data'


def load_model(tag):
    p=ROOT/'checkpoints'/f'log_aim_policy_{tag}.npz'
    policy=VisualAimPolicy(seed=1)
    with np.load(p) as z:
        weights=z['weights'].copy()
        if weights.shape!=(5,12) or not np.isfinite(weights).all():
            raise ValueError('Malformed visual model '+str(p))
        policy.weights=weights
    return policy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--eval-seed',type=int,default=8815)
    ap.add_argument('--eval',type=int,default=20)
    ap.add_argument('--decisions',type=int,default=18)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--models',nargs='+',default=['rgb01','rgb02'],
                    help='existing local log_aim_policy_<tag>.npz checkpoint tags')
    ap.add_argument('--baseline-seed',type=int,default=5501)
    args=ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}',args.run_tag):ap.error('Bad tag')
    if any(not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}',tag) for tag in args.models):ap.error('Invalid checkpoint tag')
    if len(set(args.models))!=len(args.models):ap.error('Duplicate model tags')
    if args.eval<1 or args.decisions<1:ap.error('Positive counts required')
    name='MDL_LogAimAudit_'+args.run_tag
    save=DATA/'instances'/'DemoInstance'/'saves'/name
    log=ROOT/'logs'/f'log_aim_audit_{args.run_tag}.csv'
    if save.exists() or log.exists():ap.error('Existing world/log: choose unique --run-tag')
    policies=[(f'untrained_{args.baseline_seed}',VisualAimPolicy(seed=args.baseline_seed))] + [(tag,load_model(tag)) for tag in args.models]
    print(f'New independent held-out camera seed={args.eval_seed} cases={args.eval}',flush=True)
    print('World uses original DemoWorld template in isolated copied audit world, with item-entity cleanup before EVERY case.',flush=True)
    world.WorldManager(mcio_dir=DATA).copy('storage','DemoWorld','DemoInstance',name,force=False)
    opts=types.RunOptions(mcio_dir=DATA,instance_name='DemoInstance',world_name=name,
        mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
        gpu_lib=None if args.visible else 'nvidia',width=640 if args.visible else 320,
        height=360 if args.visible else 180)
    launch=instance.Launcher(opts);ctrl=None
    try:
        launch.launch(wait=False)
        ctrl=controller.ControllerSync(action_port=opts.action_port,observation_port=opts.observation_port,connection_timeout=100)
        client=Client(ctrl)
        arena=Arena(client,client.step(clear=True))
        arena.prepare()
        print('NEGATIVE CONTROL: walk into intact log WITHOUT clicking...',flush=True)
        ob=arena.reset(0,0)
        for _ in range(25):ob=client.step(keys=('W',))
        ob=client.step()
        print(f'NEGATIVE_CONTROL no_attack_oak_count={inventory_count(ob)}',flush=True)
        if inventory_count(ob):raise RuntimeError('Invalid reset: collected log WITHOUT any attack')
        cases=make_cases(args.eval_seed,args.eval)
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['model','case','start_yaw','start_pitch',
                'outcome','decisions','reward','camera_error_deg','attack_yaw','attack_pitch','proxy_at_attack'])
            writer.writeheader()
            for name,policy in policies:
                results=[]
                for i,(yaw,pitch) in enumerate(cases,1):
                    res=run_case(arena,policy,random.Random(100+i),yaw,pitch,
                        training=False,max_decisions=args.decisions)
                    results.append(res)
                    writer.writerow({'model':name,'case':i,'start_yaw':round(yaw,3),
                                    'start_pitch':round(pitch,3),**res})
                    if res['outcome']=='success' and abs(res['attack_yaw'])>22:
                        raise RuntimeError(f'IMPOSSIBLE_SUSPECT_WIN model={name} case={i} '
                                           f'yaw={res["attack_yaw"]}: leaked old item?')
                    if res['outcome']=='success' and abs(res['attack_pitch']-3.45)>25:
                        raise RuntimeError(f'IMPOSSIBLE_SUSPECT_PITCH model={name} case={i}')
                f.flush()
                report(name,results)
        print('PASS: audited frozen evaluation completed with negative control and drop cleanup',flush=True)
        print('CSV:',log,flush=True)
    finally:
        if ctrl:
            try:ctrl.send_stop()
            except Exception as exc:print('Stop warning',repr(exc),flush=True)
            ctrl.close()
        launch.close()

if __name__=='__main__':main()
