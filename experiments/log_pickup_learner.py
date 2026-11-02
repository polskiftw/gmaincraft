"""Scratch-train Minecraft W/A/S/D pickup decisions from RGB screenshots.

A disposable fixture world summons one visible *dropped* oak log at randomized
relative ground coordinates. During policy rollout, NO server/admin commands.
The agent sees ONLY 14 hand-engineered pixel features and chooses actual MCio
movement keys. The trainer privately uses distance to the target for scalar
reward shaping; success is ALWAYS checked against the actual inventory.

This is a standalone pickup curriculum; the fixture is controller-spawned, not
mined in these training episodes. Combined mined-log evaluation is separate.
"""
import argparse
import csv
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcio_ctrl import controller,instance,network,types,world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.log_curriculum import inventory_count
from gmaincraft.log_pickup import PICKUP_ACTIONS,PickupPolicy,pickup_features,fixture_item_command
from log_interaction_probe import Client

DATA=ROOT/'mcio-data'


@dataclass(frozen=True)
class DropCase:
    dx:float
    dz:float
    yaw:float
    pitch:float=30.0


def cases_for_seed(seed,n):
    rng=random.Random(seed)
    cases=[]
    for i in range(n):
        # Encourage lateral movement, not a trainable always-forward loophole.
        # All item locations remain reachable in the same bedrock training arena.
        magnitude=rng.uniform(1.25,2.45) if rng.random()<.67 else rng.uniform(0.0,1.25)
        dx=magnitude*rng.choice([-1,1])
        dz=rng.uniform(2.45,3.75)
        yaw=rng.uniform(-15,15)
        cases.append(DropCase(dx=dx,dz=dz,yaw=yaw))
    return cases


class DropArena:
    def __init__(self,client,initial):
        self.client=client
        self.fx=math.floor(initial.player_pos[0]);self.fz=math.floor(initial.player_pos[2]);self.fy=math.floor(initial.player_pos[1])-1
        self.px=self.fx+.5;self.py=self.fy+1.0;self.pz=self.fz+.5
        self.target=None

    def prepare(self):
        f,z,y=self.fx,self.fz,self.fy
        self.client.step(commands=['difficulty peaceful','gamerule doMobSpawning false',
            'gamerule doDaylightCycle false','time set day','weather clear',
            f'fill {f-12} {y+1} {z-12} {f+12} {y+7} {z+12} minecraft:air replace',
            f'fill {f-12} {y} {z-12} {f+12} {y} {z+12} minecraft:bedrock replace',
            f'tp @s {self.px:.5f} {self.py:.5f} {self.pz:.5f} 0 30',
            'gamemode survival @s','clear @s'],clear=True)
        self.client.advance(20)

    def reset(self,case):
        self.client.scoring=False
        self.client.manager=env_util.InputStateManager()
        tx=self.px+case.dx;tz=self.pz+case.dz
        self.target=(tx,tz)
        cmd=fixture_item_command(tx,self.fy+1.20,tz)
        self.client.step(commands=[
            'kill @e[type=minecraft:item]',
            f'tp @s {self.px:.5f} {self.py:.5f} {self.pz:.5f} {case.yaw:.4f} {case.pitch:.4f}',
            'clear @s','effect clear @s','effect give @s minecraft:instant_health 1 5 true',
            'effect give @s minecraft:saturation 1 20 true',cmd],clear=True)
        ob=self.client.advance(6)
        if (math.hypot(ob.player_pos[0]-self.px,ob.player_pos[2]-self.pz)>.13
                or abs(((ob.player_yaw-case.yaw+180)%360)-180)>1.5
                or abs(ob.player_pitch-case.pitch)>1.5):
            raise RuntimeError(f'Not teleported to fresh scenario: {ob.player_pos}, yaw={ob.player_yaw}')
        if inventory_count(ob):
            raise RuntimeError('Stale item contaminated new episode despite cleanup')
        self.client.scoring=True
        return ob

    def move(self,index,ticks):
        key=PICKUP_ACTIONS[index]
        last=None
        for _ in range(ticks):
            last=self.client.step(keys=(key,))
            if inventory_count(last):break
        return last

    def distance(self,ob):
        return math.hypot(ob.player_pos[0]-self.target[0],ob.player_pos[2]-self.target[1])


def run_case(arena,policy,rng,case,*,train=False,decisions=22,ticks=5,mode='policy'):
    ob=arena.reset(case)
    total=0.0;outcome='timeout';history=[];lastdist=arena.distance(ob)
    for step in range(1,decisions+1):
        features=pickup_features(ob.get_frame())
        if mode=='forward':
            action=0;prob=None
        elif mode=='random':
            action=rng.randrange(4);prob=None
        else:
            action,prob=policy.choose(features,rng,greedy=not train)
        ob=arena.move(action,ticks)
        dist=arena.distance(ob)
        acquired=bool(inventory_count(ob))
        reward=3.0*(lastdist-dist)-.03
        if acquired:
            reward+=12.0;outcome='success'
        elif math.hypot(ob.player_pos[0]-arena.px,ob.player_pos[2]-arena.pz)>7.5:
            reward-=4.0;outcome='out_of_bounds'
        if train:
            policy.learn(features,action,prob,reward)
        total+=reward
        history.append(PICKUP_ACTIONS[action]);lastdist=dist
        if outcome!='timeout':break
    return dict(outcome=outcome,decisions=step,reward=round(total,4),
                final_distance=round(lastdist,4),
                action_count_w=history.count('W'),action_count_a=history.count('A'),
                action_count_d=history.count('D'),action_count_s=history.count('S'))


def report(label,rows):
    wins=sum(r['outcome']=='success' for r in rows)
    mean_d=sum(r['final_distance'] for r in rows)/len(rows)
    mean_steps=sum(r['decisions'] for r in rows)/len(rows)
    print(f'{label}: {wins}/{len(rows)} real Minecraft pickups; '
          f'mean final distance={mean_d:.2f} blocks; mean decisions={mean_steps:.2f}',flush=True)
    return wins


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--episodes',type=int,default=400)
    ap.add_argument('--eval',type=int,default=24)
    ap.add_argument('--decisions',type=int,default=22)
    ap.add_argument('--ticks',type=int,default=5)
    ap.add_argument('--policy-seed',type=int,default=4401)
    ap.add_argument('--train-seed',type=int,default=6601)
    ap.add_argument('--eval-seed',type=int,default=8801)
    ap.add_argument('--visible',action='store_true')
    args=ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}',args.run_tag):ap.error('Invalid run tag')
    if min(args.episodes,args.eval,args.decisions,args.ticks)<1:ap.error('All counts must be positive')
    if args.eval_seed==args.train_seed:ap.error('Train/eval seeds must differ')
    world_name=f'MDL_DropPickup_{args.run_tag}'
    save=DATA/'instances'/'DemoInstance'/'saves'/world_name
    log=ROOT/'logs'/f'drop_pickup_training_{args.run_tag}.csv'
    cp=ROOT/'checkpoints'/f'drop_pickup_policy_{args.run_tag}.npz'
    if any(p.exists() for p in (save,log,cp)):ap.error('Tag collision: refusing overwrite')
    print(f'DROP POLICY: {args.episodes} episodes seeds={args.policy_seed}/{args.train_seed}/{args.eval_seed}',flush=True)
    print('Policy observations = RGB-derived pixels only; controller scalar reward uses fixture item location.',flush=True)
    world.WorldManager(mcio_dir=DATA).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    opts=types.RunOptions(mcio_dir=DATA,instance_name='DemoInstance',world_name=world_name,
        mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
        gpu_lib=None if args.visible else 'nvidia',width=640 if args.visible else 320,height=360 if args.visible else 180)
    launch=instance.Launcher(opts);ctrl=None
    try:
        start=time.perf_counter()
        launch.launch(wait=False)
        ctrl=controller.ControllerSync(action_port=opts.action_port,observation_port=opts.observation_port,connection_timeout=100)
        client=Client(ctrl)
        arena=DropArena(client,client.step(clear=True));arena.prepare()
        policy=PickupPolicy(seed=args.policy_seed)
        cases=cases_for_seed(args.eval_seed,args.eval)
        print('Negative control: no-movement with dropped item...',flush=True)
        ob=arena.reset(cases[0]);client.advance(17)
        if inventory_count(client.last):raise RuntimeError('No-action negative control acquired item: invalid scenario')
        print('PASS: no-movement inventory remains empty',flush=True)
        for mode in ('random','forward','policy'):
            baseline=[run_case(arena,policy,random.Random(args.eval_seed+17+i),case,
                train=False,decisions=args.decisions,ticks=args.ticks,mode=mode)
                for i,case in enumerate(cases)]
            report('BEFORE '+mode,baseline)
        train_rng=random.Random(args.train_seed)
        cp.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
        learn_start=time.perf_counter()
        with log.open('w',newline='',encoding='utf-8') as f:
            fields=['episode','dx','dz','yaw','outcome','decisions','reward','final_distance',
                    'action_count_w','action_count_a','action_count_d','action_count_s']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for ep,case in enumerate(cases_for_seed(args.train_seed,args.episodes),1):
                res=run_case(arena,policy,train_rng,case,train=True,
                             decisions=args.decisions,ticks=args.ticks)
                writer.writerow(dict(episode=ep,dx=round(case.dx,4),
                    dz=round(case.dz,4),yaw=round(case.yaw,4),**res))
                if ep==1 or ep%50==0:
                    print(f'TRAIN {ep}/{args.episodes} outcome={res["outcome"]} '
                          f'reward={res["reward"]:+.2f} final_distance={res["final_distance"]:.2f}',flush=True)
                if ep%100==0:
                    f.flush()
                    np.savez_compressed(cp,weights=policy.weights,baseline=policy.baseline,updates=policy.updates)
        print(f'TRAIN TIME={time.perf_counter()-learn_start:.2f}s',flush=True)
        frozen=[run_case(arena,policy,random.Random(args.eval_seed+1000+i),case,
            train=False,decisions=args.decisions,ticks=args.ticks) for i,case in enumerate(cases)]
        report('AFTER frozen-policy (same in-process eval set; fresh-world audit still required)',frozen)
        np.savez_compressed(cp,weights=policy.weights,baseline=policy.baseline,updates=policy.updates)
        print('MODEL',cp,'LOG',log,'elapsed_total_s',round(time.perf_counter()-start,2),flush=True)
        print('NOTE: training fixture item is controller-summoned; no mining performed in this stage.',flush=True)
    finally:
        if ctrl:
            try:ctrl.send_stop()
            except Exception as exc:print('Stop warning:',repr(exc),flush=True)
            ctrl.close()
        launch.close()

if __name__=='__main__':main()
