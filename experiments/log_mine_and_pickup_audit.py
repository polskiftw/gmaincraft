"""Fresh-world audit of TWO trained policies: visual mouse/attack + pickup.

The aiming policy chooses camera and ATTACK from rendered RGB-derived features;
the separate collection policy chooses W/A/S/D from rendered RGB-derived
features after the mine. No scripted walk or teleport during scored play.

Compare untrained/always-forward/trained collection actions with the SAME
pretrained aim policy. Score genuine oak-log inventory only after vanilla
mining. Controller reset uses /setblock, /kill-items and /tp, never in rollout.
A log remains controller-placed on a fixed flat-arena template; not Any%.
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
from gmaincraft.log_vision import VisualAimPolicy,visual_features
from gmaincraft.log_pickup import PickupPolicy,PICKUP_ACTIONS,pickup_features
from gmaincraft.log_curriculum import inventory_count
from log_interaction_probe import Client
from log_visual_aim_learner import Arena,make_cases,wrap_yaw

DATA=ROOT/'mcio-data'


def load_aim(tag):
    path=ROOT/'checkpoints'/f'log_aim_policy_{tag}.npz'
    policy=VisualAimPolicy(seed=0)
    with np.load(path) as z:
        w=z['weights'].copy()
        if w.shape!=policy.weights.shape or not np.isfinite(w).all():raise ValueError('Bad aim checkpoint')
        policy.weights=w
    return policy


def load_pickup(tag):
    path=ROOT/'checkpoints'/f'drop_pickup_policy_{tag}.npz'
    policy=PickupPolicy(seed=0)
    with np.load(path) as z:
        w=z['weights'].copy()
        if w.shape!=policy.weights.shape or not np.isfinite(w).all():raise ValueError('Bad pickup checkpoint')
        policy.weights=w
    return policy


def case_run(arena,aim,pickup,mode,yaw,pitch,*,aim_steps,pickup_steps,movement_ticks):
    ob=arena.reset(yaw,pitch)
    attack_angle=None
    look_count=0
    for j in range(1,aim_steps+1):
        features=visual_features(ob.get_frame())
        action,_=aim.choose(features,random.Random(141+j),greedy=True)
        if action==4:
            attack_angle=(round(wrap_yaw(ob.player_yaw),2),round(ob.player_pitch,2))
            ob=arena.client.advance(100,keys=('LEFT_BUTTON',))
            ob=arena.client.step()
            break
        ob=arena.look(action)
        look_count+=1
    if attack_angle is None:
        return dict(outcome='no_attack',attack_yaw=None,attack_pitch=None,
                    look_decisions=look_count,pickup_decisions=0,needed_pickup=False,
                    move_w=0,move_a=0,move_s=0,move_d=0)
    collected=bool(inventory_count(ob))
    needed_pickup=not collected
    used=[]
    for k in range(1,pickup_steps+1):
        if collected:break
        if mode=='forward':
            direction=0
        elif mode=='random':
            direction=random.Random(100000+j+k*391).randrange(4)
        else:
            features=pickup_features(ob.get_frame())
            direction,_=pickup.choose(features,random.Random(141+k),greedy=True)
        key=PICKUP_ACTIONS[direction]
        used.append(key)
        for t in range(movement_ticks):
            ob=arena.client.step(keys=(key,))
            if inventory_count(ob):
                collected=True
                break
    return dict(outcome='success' if collected else 'miss',
                attack_yaw=attack_angle[0],attack_pitch=attack_angle[1],
                look_decisions=look_count,pickup_decisions=len(used),
                needed_pickup=needed_pickup,
                move_w=used.count('W'),move_a=used.count('A'),
                move_s=used.count('S'),move_d=used.count('D'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--aim-model',default='rgb02')
    p.add_argument('--pickup-models',nargs='+',default=['p01','p02'])
    p.add_argument('--eval-seed',type=int,default=8816)
    p.add_argument('--eval',type=int,default=24)
    p.add_argument('--aim-decisions',type=int,default=18)
    p.add_argument('--pickup-decisions',type=int,default=22)
    p.add_argument('--pickup-ticks',type=int,default=5)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',args.run_tag):p.error('Invalid run tag')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',tag) for tag in [args.aim_model,*args.pickup_models]):p.error('Invalid model tag')
    if min(args.eval,args.aim_decisions,args.pickup_decisions,args.pickup_ticks)<1:p.error('Counts must be positive')
    name=f'MDL_FullLogAudit_{args.run_tag}'
    save=DATA/'instances'/'DemoInstance'/'saves'/name
    log=ROOT/'logs'/f'full_log_pickup_audit_{args.run_tag}.csv'
    if save.exists() or log.exists():p.error('Audit run already exists: refusing overwrite')
    aim=load_aim(args.aim_model)
    policies=[('forward',None,'forward'),('untrained',PickupPolicy(seed=4401),'policy')]
    policies.extend((tag,load_pickup(tag),'policy') for tag in args.pickup_models)
    print(f'INTEGRATED NEW-WORLD AIM+PICKUP AUDIT: seed={args.eval_seed}  cases={args.eval}',flush=True)
    print(f'Aim model={args.aim_model}; movement controls + {args.pickup_models}',flush=True)
    print('Physical Minecraft controller actions ONLY during play; no predetermined W pickup for policy modes.',flush=True)
    world.WorldManager(mcio_dir=DATA).copy('storage','DemoWorld','DemoInstance',name,force=False)
    opts=types.RunOptions(mcio_dir=DATA,instance_name='DemoInstance',world_name=name,
       mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
       gpu_lib=None if args.visible else 'nvidia',width=640 if args.visible else 320,height=360 if args.visible else 180)
    launch=instance.Launcher(opts);ctrl=None
    try:
        launch.launch(wait=False)
        ctrl=controller.ControllerSync(action_port=opts.action_port,
          observation_port=opts.observation_port,connection_timeout=100)
        client=Client(ctrl)
        arena=Arena(client,client.step(clear=True));arena.prepare()
        print('NEGATIVE CONTROL: approach intact oak log without any ATTACK...',flush=True)
        ob=arena.reset(0.0,0.0)
        for _ in range(24):
            ob=client.step(keys=('W',))
        ob=client.step()
        print(f'NO_ATTACK_CONTROL oak_count={inventory_count(ob)}',flush=True)
        if inventory_count(ob):
            raise RuntimeError('Unexpected pickup without mining: contaminated episode')
        cases=make_cases(args.eval_seed,args.eval)
        # Exactly the same camera-angle scenarios for all four pickup policies;
        # every reset removes earlier item entities and returns player to pose.
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=['movement','case','start_yaw','start_pitch','outcome',
                'attack_yaw','attack_pitch','look_decisions','pickup_decisions','needed_pickup',
                'move_w','move_a','move_s','move_d'])
            writer.writeheader()
            for label,policy,mode in policies:
                weights=policy.weights.copy() if policy is not None else None
                results=[]
                for i,(yaw,pitch) in enumerate(cases,1):
                    result=case_run(arena,aim,policy,mode,yaw,pitch,
                        aim_steps=args.aim_decisions,pickup_steps=args.pickup_decisions,
                        movement_ticks=args.pickup_ticks)
                    if result['outcome']=='success' and (result['attack_yaw'] is None or abs(result['attack_yaw'])>22):
                        raise RuntimeError(f'IMPOSSIBLE PICKUP: {label} case={i} {result}')
                    writer.writerow(dict(movement=label,case=i,start_yaw=round(yaw,3),
                                         start_pitch=round(pitch,3),**result))
                    results.append(result)
                if policy is not None and not np.array_equal(weights,policy.weights):
                    raise RuntimeError('Frozen evaluator modified movement policy weights')
                file.flush()
                wins=sum(r['outcome']=='success' for r in results)
                attempts=sum(r['attack_yaw'] is not None for r in results)
                moved=sum(r['outcome']=='success' and r['needed_pickup'] for r in results)
                print(f'INTEGRATED {label}: {wins}/{len(results)} natural mined-log inventory pickups, '
                      f'attack attempts={attempts}, success requiring movement={moved}',flush=True)
        print('PASS: combined real mining + neural visual movement evaluation completed',flush=True)
        print('CSV',log,flush=True)
    finally:
        if ctrl:
            try:ctrl.send_stop()
            except Exception as exc:print('Stop warning',repr(exc),flush=True)
            ctrl.close()
        launch.close()

if __name__=='__main__':main()
