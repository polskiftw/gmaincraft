"""Joint RGB look/move/mine actor-critic with real Minecraft rollouts.

Controller MAY place a real log column and reset player in a disposable flat
training world. Only rendered pixels reach policy; privileged position/target
geometry drives scalar shaping. Real Minecraft inventory verifies success.
This is a controlled curriculum, NOT independent-world/general Any% learning.

Policy learns decisions over 11 discrete vanilla input macros with actual
Minecraft ticks, full-trajectory GAE and PPO. The 80-tick mining hold is
still a fixed motor primitive. Evaluation freezes policy and uses held-out
position/yaw draws, same arena template. Natural seed transfer is assessed
separately using natural_world_benchmark.py --modes joint_ppo.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from mcio_ctrl import controller, instance, types, world
from mcio_ctrl.envs import env_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gmaincraft.agent_interface import MCioAgentBridge, natural_log_count
from gmaincraft.vision_ppo import PPOTrainer, PixelHistory, Transition, action_for_index
from gmaincraft.joint_curriculum import LogCase, sample_cases
from log_interaction_probe import Client

MCIO = ROOT / 'mcio-data'


def yaw_error(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class VariableLogArena:
    """Only the privileged experiment controller may edit this disposable world."""
    def __init__(self, client, first_packet):
        self.client = client
        self.fx = math.floor(first_packet.player_pos[0])
        self.fy = math.floor(first_packet.player_pos[1]) - 1
        self.fz = math.floor(first_packet.player_pos[2])
        self.target = None

    def prepare(self):
        x,y,z = self.fx,self.fy,self.fz
        self.client.step(commands=[
            'difficulty peaceful', 'gamerule doMobSpawning false',
            'gamerule doDaylightCycle false', 'time set day', 'weather clear',
            f'fill {x-16} {y+1} {z-16} {x+16} {y+8} {z+16} minecraft:air replace',
            f'fill {x-16} {y} {z-16} {x+16} {y} {z+16} minecraft:bedrock replace',
            'gamemode survival @s', 'clear @s'], clear=True)
        self.client.advance(4)

    def reset(self, case):
        self.client.scoring = False
        self.client.manager = env_util.InputStateManager()
        x,y,z = self.fx,self.fy,self.fz
        px = x+.5+case.player_dx
        pz = z+.5+case.player_dz
        tx = x+case.tree_dx
        tz = z+case.tree_dz
        self.target = (tx+.5, y+1.5, tz+.5)
        ob = self.client.step(commands=[
            'kill @e[type=minecraft:item]',
            f'fill {x-10} {y+1} {z-10} {x+10} {y+5} {z+10} minecraft:air replace minecraft:oak_log',
            f'setblock {tx} {y+1} {tz} minecraft:oak_log replace',
            f'setblock {tx} {y+2} {tz} minecraft:oak_log replace',
            f'setblock {tx} {y+3} {tz} minecraft:oak_log replace',
            f'tp @s {px:.5f} {y+1.0:.4f} {pz:.5f} {case.yaw:.4f} {case.pitch:.4f}',
            'clear @s', 'effect clear @s',
            'effect give @s minecraft:instant_health 1 5 true',
            'effect give @s minecraft:saturation 1 20 true'], clear=True)
        for _ in range(20):
            ob = self.client.step()
            if (math.hypot(ob.player_pos[0]-px,ob.player_pos[2]-pz) < .12 and
                abs(yaw_error(ob.player_yaw, case.yaw)) < 1.0 and
                abs(ob.player_pitch-case.pitch) < 1.0):
                break
        else:
            raise RuntimeError('Could not reset player pose to configured scenario')
        if natural_log_count(ob):
            raise RuntimeError('Stale logs in inventory at episode start')
        self.client.scoring = True
        return ob

    def geometry(self, observation):
        """PRIVATE TRAINER-ONLY reward shaping; never a policy observation."""
        px,py,pz = observation.player_pos
        tx,ty,tz = self.target
        dx,dz = tx-px,tz-pz
        distance = math.hypot(dx,dz)
        desired_yaw = math.degrees(math.atan2(-dx,dz))
        desired_pitch = -math.degrees(math.atan2(ty-(py+1.62),max(distance,.0001)))
        err = math.hypot(yaw_error(observation.player_yaw,desired_yaw),
                         observation.player_pitch-desired_pitch)
        return distance,err


def run_episode(arena, trainer, case, *, max_ticks, training):
    observation = arena.reset(case)
    bridge = MCioAgentBridge(arena.client.ctrl, observation)
    history = PixelHistory()
    image = history.reset(bridge.view())
    episode_started = time.perf_counter()
    ticks = 0
    total_reward = 0.
    trajectory = []
    real_success = False
    actions = 0
    previous_distance, previous_error = arena.geometry(observation)
    while ticks < max_ticks:
        action_index, logp, value = trainer.act(image, greedy=not training)
        action = action_for_index(action_index,max_ticks-ticks)
        next_view = bridge.apply(action)
        ticks += action.ticks
        actions += 1
        current_distance, current_error = arena.geometry(bridge.packet)
        real_success = natural_log_count(bridge.packet) > 0
        dead = bridge.packet.health <= 0
        # Bounded teacher shaping only in the CONTROLLED curriculum. It does
        # not prove that a natural-world agent has learned general goals.
        reward = (-.004 * action.ticks/5
                  + .28 * float(np.clip(previous_distance-current_distance,-1,1))
                  + .014 * float(np.clip(previous_error-current_error,-25,25)))
        if real_success:
            reward += 16.
        if dead:
            reward -= 8.
        done = bool(real_success or dead or ticks >= max_ticks)
        if training:
            trajectory.append(Transition(image.copy(),action_index,float(reward),
                                         value,logp,action.ticks,done))
        total_reward += reward
        image = history.append(next_view)
        previous_distance,previous_error = current_distance,current_error
        if done:
            break
    return dict(outcome='success' if real_success else 'dead' if bridge.packet.health<=0 else 'timeout',
                ticks=ticks,actions=actions,reward=round(total_reward,4),
                final_distance=round(previous_distance,3),
                final_angular_error=round(previous_error,3),
                wall_seconds=round(time.perf_counter()-episode_started,3)),trajectory


def launch_world(world_name, args, evaluator):
    manager = world.WorldManager(mcio_dir=MCIO)
    manager.copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    opts = types.RunOptions(mcio_dir=MCIO, instance_name='DemoInstance',world_name=world_name,
                            mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
                            gpu_lib=None if args.visible else 'nvidia',
                            width=args.frame_width, height=args.frame_height)
    launcher = instance.Launcher(opts)
    ctrl = None
    try:
        launcher.launch(wait=False)
        ctrl = controller.ControllerSync(action_port=opts.action_port,
                                         observation_port=opts.observation_port,
                                         connection_timeout=100)
        client = Client(ctrl)
        arena = VariableLogArena(client,client.step(clear=True))
        arena.prepare()
        return evaluator(arena)
    finally:
        if ctrl:
            try:
                ctrl.send_stop()
            except Exception as e:
                print('Stop warning',repr(e),flush=True)
            ctrl.close()
        launcher.close()


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--eval',type=int,default=16)
    ap.add_argument('--max-ticks',type=int,default=320)
    ap.add_argument('--update-every',type=int,default=8)
    ap.add_argument('--train-seed',type=int,default=670101)
    ap.add_argument('--eval-seed',type=int,default=890101)
    ap.add_argument('--policy-seed',type=int,default=4201)
    ap.add_argument('--init-model', default=None,
                    help='optional checkpoint tag for a NEW run; restores model + optimizer')
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--frame-width', type=int, default=320)
    ap.add_argument('--frame-height', type=int, default=180)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--skip-eval',action='store_true')
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid run tag')
    if min(args.episodes,args.eval,args.max_ticks,args.update_every)<1:
        ap.error('Counts must be positive')
    if min(args.frame_width,args.frame_height) < 96 or max(args.frame_width,args.frame_height) > 1920:
        ap.error('Frame dimensions must be in the range 96..1920')
    if args.train_seed==args.eval_seed:
        ap.error('Train/eval scenario seeds must differ')
    if args.init_model is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.init_model):
        ap.error('Invalid --init-model tag')
    if args.init_model == args.run_tag:
        ap.error('Warm start must use a NEW run-tag')
    return args


def main(argv=None):
    args=parse_args(argv)
    train_world=f'MDL_JointLogTrain_{args.run_tag}'
    eval_world=f'MDL_JointLogEval_{args.run_tag}'
    log=ROOT/'logs'/f'joint_log_ppo_{args.run_tag}.jsonl'
    model=ROOT/'checkpoints'/f'joint_log_ppo_{args.run_tag}.pt'
    save_base=MCIO/'instances'/'DemoInstance'/'saves'
    if log.exists() or model.exists() or (save_base/train_world).exists() or (save_base/eval_world).exists():
        raise FileExistsError('Run tag collision: refusing to overwrite experiment artifacts')
    trainer=PPOTrainer(device=args.device,seed=args.policy_seed)
    if args.init_model:
        source=ROOT/'checkpoints'/f'joint_log_ppo_{args.init_model}.pt'
        if not source.is_file():
            raise FileNotFoundError(f'Cannot warm start: {source}')
        ancestor=trainer.load_weights(source,restore_optimizer=True)
        if (args.train_seed == ancestor.get('train_seed') or
                args.policy_seed == ancestor.get('policy_seed')):
            raise ValueError('Warm start requires new train-seed AND policy-seed to avoid replaying previous cases and action RNG')
        print(f'WARM START: trained weights and Adam state from {source}',flush=True)
    print(f'JOINT RGB PPO: {trainer.device}, {trainer.parameter_count} parameters; '
          f'{args.episodes} episodes, {args.max_ticks} game ticks each; '
          f'render={args.frame_width}x{args.frame_height}, CNN=2xRGB84',flush=True)
    if str(trainer.device)=='cpu':
        print('CUDA NOT USED; CPU learner in this environment. GPU requires NVIDIA UVM passthrough.',flush=True)
    train_cases=sample_cases(args.train_seed,args.episodes)
    eval_cases=sample_cases(args.eval_seed,args.eval)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as out:
        out.write(json.dumps(dict(stage='config',tag=args.run_tag,
                                  policy_seed=args.policy_seed,train_seed=args.train_seed,
                                  eval_seed=args.eval_seed,initialized_from=args.init_model,
                                  device=str(trainer.device),
                                  frame_width=args.frame_width,frame_height=args.frame_height,
                                  max_ticks=args.max_ticks))+'\n')
        out.flush()
        def train(arena):
            collected=[]
            successes=0
            for i,case in enumerate(train_cases,1):
                row,transitions=run_episode(arena,trainer,case,max_ticks=args.max_ticks,training=True)
                successes+=int(row['outcome']=='success')
                out.write(json.dumps(dict(stage='train',episode=i,case=case.__dict__,**row))+'\n')
                out.flush()
                collected.extend(transitions)
                if i%args.update_every==0 or i==len(train_cases):
                    stats=trainer.update(collected)
                    collected=[]
                    trainer.save(model,dict(policy_seed=args.policy_seed,train_seed=args.train_seed,
                                            episodes_completed=i,max_ticks=args.max_ticks,
                                            observation='2xRGB84',action='joint11timed',
                                            frame_width=args.frame_width,
                                            frame_height=args.frame_height,
                                            initialized_from=args.init_model or 'scratch'))
                    print(f'TRAIN {i}/{len(train_cases)} successes={successes} '
                          f'optimizer={stats} latest={row}',flush=True)
                elif i==1:
                    print(f'TRAIN first episode {row}',flush=True)
            return successes
        t0=time.perf_counter()
        train_successes=launch_world(train_world,args,train)
        print(f'TRAIN COMPLETED success={train_successes}/{len(train_cases)} '
              f'elapsed_wall_s={time.perf_counter()-t0:.1f}',flush=True)
        if not args.skip_eval:
            def evaluate(arena):
                successes=0
                for i,case in enumerate(eval_cases,1):
                    row,_=run_episode(arena,trainer,case,max_ticks=args.max_ticks,training=False)
                    successes+=int(row['outcome']=='success')
                    out.write(json.dumps(dict(stage='heldout',case_index=i,case=case.__dict__,**row))+'\n')
                    out.flush()
                    print(f'EVAL {i}/{len(eval_cases)} {row}',flush=True)
                return successes
            eval_successes=launch_world(eval_world,args,evaluate)
            print(f'FROZEN HELDOUT: {eval_successes}/{len(eval_cases)} acquired actual logs',flush=True)
        print(f'Checkpoint (LOCAL ONLY): {model}',flush=True)
        print(f'Results (LOCAL ONLY): {log}',flush=True)


if __name__=='__main__':main()
