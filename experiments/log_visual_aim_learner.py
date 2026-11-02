"""Early REAL Minecraft RGB-to-camera-action reinforcement learning curriculum.

An agent starts from random weights and receives ONLY engineered features of the
RGB Minecraft screenshot. It may look left/right/up/down or choose ATTACK.
The privileged *trainer* knows which direction a controller-placed wood target
is for SCALAR reward shaping, but never passes target coordinates, Minecraft
player pose or the correct action to the policy. Evaluation is frozen: the
agent selects when to attack; success requires a real oak log in inventory.
A simple fixed W approach is used solely *after* agent-selected ATTACK.

Do not confuse this simplified same-world block task with vision-based Any%.
Run only with a NEW run tag. All output/worlds are ignored by git.
"""
import argparse
import csv
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from mcio_ctrl import controller,instance,network,types,world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.log_vision import visual_features,VisualAimPolicy
from gmaincraft.log_curriculum import inventory_count
from log_interaction_probe import Client

DATA=ROOT/'mcio-data'
ACTIONS=VisualAimPolicy.ACTIONS


def wrap_yaw(yaw):
    return (yaw+180.0)%360.0-180.0


def score_angle(ob):
    """Controller-side scalar training reward; never passed to policy features."""
    # Target's center is two blocks forward and approximately 0.12 below eye.
    return math.hypot(wrap_yaw(ob.player_yaw), ob.player_pitch - 3.45)


def ray_alignment_proxy(ob):
    """Conservative geometric training criterion; real eval uses inventory."""
    return abs(wrap_yaw(ob.player_yaw)) <= 9 and abs(ob.player_pitch-3.45) <= 8


class Arena:
    def __init__(self, client, initial):
        self.client=client
        self.fx=math.floor(initial.player_pos[0])
        self.fy=math.floor(initial.player_pos[1])-1
        self.fz=math.floor(initial.player_pos[2])
        self.player_x=self.fx+0.5
        self.player_y=self.fy+1.0
        self.player_z=self.fz+0.5
        self.log_z=self.fz+2
        self.look_pixels=45
        self.ticks=0

    def prepare(self):
        f=self.fx;y=self.fy;z=self.fz;logz=self.log_z
        self.client.step(commands=[
            'difficulty peaceful','gamerule doMobSpawning false',
            'gamerule doDaylightCycle false','time set day','weather clear',
            f'fill {f-8} {y+1} {z-8} {f+8} {y+7} {z+8} minecraft:air replace',
            f'fill {f-8} {y} {z-8} {f+8} {y} {z+8} minecraft:bedrock replace',
            f'setblock {f} {y+1} {logz} minecraft:oak_log replace',
            f'setblock {f} {y+2} {logz} minecraft:oak_log replace',
            f'setblock {f} {y+3} {logz} minecraft:oak_log replace',
            f'tp @s {self.player_x:.4f} {self.player_y:.4f} {self.player_z:.4f} 0 0',
            'gamemode survival @s','clear @s',
        ],clear=True)
        self.client.advance(20)

    def reset(self, yaw, pitch):
        self.client.scoring=False
        self.client.manager=env_util.InputStateManager()
        ob=self.client.step(commands=[
            # A soft reset does not clean dropped items. Without this, a later
            # episode could pick up a PREVIOUSLY mined log and score a false win.
            # This is controller-only environment restoration, never an agent action.
            'kill @e[type=minecraft:item]',
            f'setblock {self.fx} {self.fy+1} {self.log_z} minecraft:oak_log replace',
            f'setblock {self.fx} {self.fy+2} {self.log_z} minecraft:oak_log replace',
            f'setblock {self.fx} {self.fy+3} {self.log_z} minecraft:oak_log replace',
            f'tp @s {self.player_x:.5f} {self.player_y:.5f} {self.player_z:.5f} {yaw:.5f} {pitch:.5f}',
            'clear @s','effect clear @s','effect give @s minecraft:instant_health 1 5 true',
            'effect give @s minecraft:saturation 1 20 true'],clear=True)
        for tick in range(20):
            ob=self.client.step()
            if (abs(wrap_yaw(ob.player_yaw-yaw))<1 and
                    abs(ob.player_pitch-pitch)<1 and
                    math.hypot(ob.player_pos[0]-self.player_x,ob.player_pos[2]-self.player_z)<0.1):
                break
        else:raise RuntimeError('Reset yaw/pitch/position not observed')
        ob=self.client.advance(2)
        if inventory_count(ob):raise RuntimeError('Inventory not empty after reset')
        self.client.scoring=True
        return ob

    def look(self, index):
        dx=dy=0
        if index==0:dx=-self.look_pixels
        if index==1:dx=self.look_pixels
        if index==2:dy=-self.look_pixels
        if index==3:dy=self.look_pixels
        if index not in range(4):raise ValueError('Invalid look action')
        ob=self.client.step(look=(dx,dy))
        return self.client.step()

    def mine_and_collect(self):
        """Agent chose ATTACK; actual break+pickup with no controller commands.

        Walking to collect is a FIXED diagnostic finisher, not yet learned.
        """
        ob=self.client.advance(100,keys=('LEFT_BUTTON',))
        ob=self.client.step()
        if inventory_count(ob)>0:return True
        for tick in range(1,44):
            ob=self.client.step(keys=('W',) if tick<=25 else ())
            if inventory_count(ob)>0:
                return True
        return False


def run_case(arena,policy,rng,yaw,pitch,*,training,max_decisions):
    ob=arena.reset(yaw,pitch)
    total=0.0
    outcome='timeout'
    attack_angle=None
    for idx in range(1,max_decisions+1):
        # Crucial: the neural policy reads image features, not ob.player_yaw,
        # ob.player_pitch or any target block coordinates.
        features=visual_features(ob.get_frame())
        action,prob=policy.choose(features,rng,greedy=not training)
        before=score_angle(ob)
        if action==4:
            attack_angle=(round(wrap_yaw(ob.player_yaw),3),round(ob.player_pitch,3))
            if training:
                success=ray_alignment_proxy(ob)
            else:
                success=arena.mine_and_collect()  # observed inventory only
            reward=12.0 if success else -7.0
            outcome='success' if success else 'miss'
        else:
            ob=arena.look(action)
            after=score_angle(ob)
            reward=0.32*(before-after)-0.04
            if after>58:
                reward-=1.25
        if training:policy.learn(features,action,prob,reward)
        total+=reward
        if action==4:break
    return {'outcome':outcome,'decisions':idx,'reward':total,
            'camera_error_deg':round(score_angle(ob),3),
            'attack_yaw':attack_angle[0] if attack_angle else None,
            'attack_pitch':attack_angle[1] if attack_angle else None,
            'proxy_at_attack':bool(attack_angle and abs(attack_angle[0])<=9 and abs(attack_angle[1]-3.45)<=8)}


def make_cases(seed,n):
    rng=random.Random(seed)
    # Keep the block on screen in this very first curriculum.
    return [(rng.uniform(-28,28),rng.uniform(-8,13)) for _ in range(n)]


def report(label,results):
    win=sum(r['outcome']=='success' for r in results)
    mean=sum(r['camera_error_deg'] for r in results)/len(results)
    print(f'{label}: {win}/{len(results)} real oak-log acquisitions, '
          f'avg final angular error={mean:.2f} degrees, '
          f'avg decisions={sum(r["decisions"] for r in results)/len(results):.1f}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--episodes',type=int,default=500)
    p.add_argument('--eval',type=int,default=16)
    p.add_argument('--decisions',type=int,default=16)
    p.add_argument('--policy-seed',type=int,default=5501)
    p.add_argument('--train-seed',type=int,default=7723)
    p.add_argument('--eval-seed',type=int,default=8813)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    if min(args.episodes,args.eval,args.decisions)<1:p.error('Counts must be positive')
    if args.eval_seed==args.train_seed:p.error('Training/evaluation seeds must differ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',args.run_tag):p.error('Invalid run tag')
    name=f'MDL_LogAim_{args.run_tag}'
    save=DATA/'instances'/'DemoInstance'/'saves'/name
    cp=ROOT/'checkpoints'/f'log_aim_policy_{args.run_tag}.npz'
    log=ROOT/'logs'/f'log_aim_training_{args.run_tag}.csv'
    if any(x.exists() for x in (save,cp,log)):
        p.error('Tagged world/checkpoint/log exists: refusing overwrite')
    print(f'RGB policy, seeds policy={args.policy_seed} train={args.train_seed} '
          f'eval={args.eval_seed}, episodes={args.episodes} eval={args.eval}',flush=True)
    print(f'Copying DemoWorld into ONLY {name}',flush=True)
    world.WorldManager(mcio_dir=DATA).copy('storage','DemoWorld','DemoInstance',name,force=False)
    opts=types.RunOptions(mcio_dir=DATA,instance_name='DemoInstance',world_name=name,
        mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
        gpu_lib=None if args.visible else 'nvidia',width=640 if args.visible else 320,
        height=360 if args.visible else 180)
    launch=instance.Launcher(opts);ctrl=None
    try:
        started=time.perf_counter()
        launch.launch(wait=False)
        ctrl=controller.ControllerSync(action_port=opts.action_port,
            observation_port=opts.observation_port,connection_timeout=100)
        client=Client(ctrl); initial=client.step(clear=True)
        arena=Arena(client,initial);arena.prepare()
        policy=VisualAimPolicy(seed=args.policy_seed)
        cases=make_cases(args.eval_seed,args.eval)
        rng=random.Random(args.policy_seed+1)
        print('Baseline: untrained random-weights image-only policy...',flush=True)
        before=[run_case(arena,policy,rng,*case,training=False,
                         max_decisions=args.decisions) for case in cases]
        report('BEFORE',before)
        print('Starting real-game camera-action RL (controller angle only for SCALAR reward)...',flush=True)
        train_rng=random.Random(args.train_seed)
        cp.parent.mkdir(parents=True,exist_ok=True)
        log.parent.mkdir(parents=True,exist_ok=True)
        learn_start=time.perf_counter()
        with log.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['episode','outcome','decisions','reward','camera_error_deg','attack_yaw','attack_pitch','proxy_at_attack'])
            writer.writeheader()
            for episode,(yaw,pitch) in enumerate(make_cases(args.train_seed,args.episodes),1):
                res=run_case(arena,policy,train_rng,yaw,pitch,training=True,max_decisions=args.decisions)
                writer.writerow({'episode':episode,**res})
                if episode%50==0 or episode==1:
                    print(f'  training {episode}/{args.episodes}: '
                          f'{res["outcome"]}, reward={res["reward"]:.2f}, '
                          f'err={res["camera_error_deg"]:.1f}',flush=True)
                if episode%100==0:
                    f.flush()
                    np.savez_compressed(cp,weights=policy.weights,
                        updates=policy.updates,baseline=policy.baseline)
        learner_time=time.perf_counter()-learn_start
        print('Frozen evaluation on untouched held-out camera angles...',flush=True)
        after=[run_case(arena,policy,random.Random(args.eval_seed+23),*case,
               training=False,max_decisions=args.decisions) for case in cases]
        report('AFTER',after)
        np.savez_compressed(cp,weights=policy.weights,updates=policy.updates,
                           baseline=policy.baseline)
        print(f'Elapsed learning time={learner_time:.2f}s; '
              f'wall total={time.perf_counter()-started:.2f}s',flush=True)
        print('Checkpoint:',cp,flush=True)
        print('CSV:',log,flush=True)
        print('NOTE: visual encoder uses a handcrafted oak-brown pixel filter. '
              'Trainer uses privileged angular-distance *rewards*, NEVER target direction in policy inputs. '
              'The policy selects ATTACK; forward pickup is still scripted.',flush=True)
    finally:
        if ctrl:
            try:ctrl.send_stop()
            except Exception as exc:print('Stop warning',repr(exc),flush=True)
            ctrl.close()
        launch.close()

if __name__=='__main__':main()
