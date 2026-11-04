"""Self-experience ONLY: train a goal-conditioned ordinary motor policy.

Read *committed* player-observation archives from a natural survival train
stream, relabel actually achieved inventory gains as optional learning goals,
and train a small scratch policy to imitate actions that preceded success.
This does NOT prove goal-directed play, mastery, planning, or Any% progress.
Only self-collected positives (no expert actions or hidden world state).
Can be resumed repeatedly after more natural Minecraft training accrues.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from gmaincraft.experience_archive import (archive_path,read_episode,
                                           hindsight_goal_transitions)
from gmaincraft.hindsight_skills import HindsightSkillModel,SCHEMA


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True,help='Existing self-play natural train tag with experience archive')
    p.add_argument('--run-tag',required=True,help='New learned goal checkpoint tag')
    p.add_argument('--resume',action='store_true',help='Continue a prior goal model; epochs are cumulative')
    p.add_argument('--epochs',type=int,default=10,help='Cumulative epochs through current committed experience')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--max-examples',type=int,default=100000,
                   help='Deterministic reservoir cap on achieved-goal samples per split')
    p.add_argument('--max-episodes',type=int,default=2000,
                   help='Read at most N newest committed training episodes')
    p.add_argument('--lookback',type=int,default=12,
                   help='Max own actions preceding each observed inventory gain')
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='auto')
    p.add_argument('--seed',type=int,default=20261103)
    args=p.parse_args(argv)
    pattern=r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}'
    if not re.fullmatch(pattern,args.source_run) or not re.fullmatch(pattern,args.run_tag):
        p.error('Unsafe source/model run tag')
    if (not 1<=args.epochs<=10000000 or not 1<=args.batch_size<=1024 or
        not 1<=args.max_examples<=1000000 or not 1<=args.max_episodes<=10000000 or
        not 1<=args.lookback<=10000 or not 0<=args.seed<2**63):
        p.error('Invalid goal-learning budget')
    return args


def committed_source_meta(root,source_run):
    import torch
    path=root/'checkpoints'/f'fresh_stream_{source_run}.pt'
    if not path.is_file():raise FileNotFoundError('Missing original Minecraft PPO source checkpoint')
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if (not ckpt.get('metadata',{}).get('policy_schema','').startswith('natural_memory_online_') or
        ckpt.get('stream')!='minecraft_seed_stream_v2_full64_interleaved_train_eval' or
        type(ckpt.get('next_episode')) is not int or ckpt['next_episode']<=0):
        raise ValueError('Source is not a committed fresh-seed training stream')
    return ckpt['metadata']['policy_schema'],ckpt['next_episode']


def load_examples(root,source_run,*,max_episodes,max_examples,lookback,seed):
    """Scan only episodes BEFORE original PPO checkpoint's committed cursor.

    A writer publishes experience *before* its PPO checkpoint. Snapshotting
    cursor protects concurrent readers from consuming uncommitted episodes.
    An episode's last digits are book-keeping, never policy inputs.
    """
    obs_schema,committed_cursor=committed_source_meta(root,source_run)
    directory=root/'data-local'/'experience'/source_run
    paths=sorted(p for p in directory.glob('episode_*.npz') if
                 p.is_file() and not p.is_symlink() and
                 p.name.startswith('episode_') and p.name.endswith('.npz') and
                 p.name[8:20].isdigit() and len(p.name)==24 and
                 int(p.name[8:20])<committed_cursor)[-max_episodes:]
    if not paths:raise ValueError('No COMMITTED recorded experience. Run natural training with --record-experience first.')
    training=[]
    validation=[]
    rng=random.Random(seed)
    seen=[0,0]
    action_sizes=own_size=None
    goal_counts={}
    for path in paths:
        meta,data=read_episode(path)
        if meta['observation_schema']!=obs_schema:
            raise ValueError('Source observation schema mismatch; archive not from active model')
        sizes=tuple(meta['action_sizes'])
        dim=data['before_own'].shape[1]
        if action_sizes is None:action_sizes,own_size=sizes,dim
        if action_sizes!=sizes or own_size!=dim:
            raise ValueError('Incompatible action/observation versions in source archive')
        # A fixed subset of whole real-world episodes is withheld from
        # cloning (internal diagnostics, NOT independent final Minecraft eval).
        split=1 if int(path.name[8:20])%5==4 else 0
        out=validation if split else training
        for x in hindsight_goal_transitions(meta,data,lookback_steps=lookback):
            j=x['observation_index']
            ex={key:data['before_'+key][j] for key in
                ('own','crosshair_id','surface_ids','surface_data','surface_mask')}
            # Map to model's input format but *never* include Minecraft seed,
            # evaluator stats, raw hidden geometry, or desired route.
            ex={'before_'+k:v for k,v in ex.items()}
            ex.update(action=x['action'],goal_item_id=x['goal_item_id'],
                      ticks_to_achievement=x['ticks_to_achievement'])
            goal_counts[x['goal_item_id']]=goal_counts.get(x['goal_item_id'],0)+1
            seen[split]+=1
            if len(out)<max_examples:out.append(ex)
            else:
                replace_index=rng.randrange(seen[split])
                if replace_index<max_examples:out[replace_index]=ex
    if not training:
        raise ValueError('No TRAINING examples with real observed inventory gains yet. '
                         'More self-play or an actual resource-learning breakthrough is needed.')
    return obs_schema,own_size,action_sizes,training,validation,dict(
        committed_cursor=committed_cursor,episodes_scanned=len(paths),
        positive_training_seen=seen[0],positive_validation_seen=seen[1],
        goals_by_item=goal_counts,training_examples=len(training),
        validation_examples=len(validation))


def accuracy(model,examples,batch_size=256):
    if not examples:return None
    correct=total=0
    for index in range(0,len(examples),batch_size):
        chunk=examples[index:index+batch_size]
        model.net.eval()
        with model.torch.no_grad():
            logits=model.net(*model.batch(chunk))
            for j,head in enumerate(logits):
                predicted=head.argmax(-1)
                actual=model.torch.tensor([x['action'][j] for x in chunk],
                                          device=model.device)
                correct+=int((predicted==actual).sum().item())
                total+=len(chunk)
    return round(correct/max(total,1),4)


def atomic_save(path,model,meta,epoch):
    pending=path.with_suffix('.pt.pending')
    if pending.exists():raise FileExistsError('Unfinished goal-model checkpoint stage exists; inspect before resuming')
    try:
        with pending.open('xb') as out:
            model.torch.save(dict(schema=SCHEMA,metadata=meta,epochs=epoch,model=model.state()),out)
            out.flush();os.fsync(out.fileno())
        os.replace(pending,path)
    finally:
        # If an I/O failure interrupted a write, preserve the staged file
        # and do not claim a new epoch is committed.
        pass


def main(argv=None):
    args=cli(argv)
    root=ROOT
    ckpt=root/'checkpoints'/f'hindsight_{args.run_tag}.pt'
    journal=root/'logs'/f'hindsight_{args.run_tag}.jsonl'
    lock_path=root/'checkpoints'/f'hindsight_{args.run_tag}.lock'
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    journal.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Hindsight trainer already running') from exc
        if args.resume and (not ckpt.exists() or not journal.exists()):
            raise FileNotFoundError('Cannot resume a missing learned-skill checkpoint')
        if not args.resume and (ckpt.exists() or journal.exists()):
            raise FileExistsError('Refusing to overwrite previous learned-skill experiment')
        observation_schema,own_size,action_sizes,train,validation,stats=load_examples(
            root,args.source_run,max_episodes=args.max_episodes,max_examples=args.max_examples,
            lookback=args.lookback,seed=args.seed)
        meta=dict(source_run=args.source_run,source_observation_schema=observation_schema,
                  own_size=own_size,action_sizes=action_sizes,lookback=args.lookback,
                  max_episodes=args.max_episodes,max_examples=args.max_examples,
                  batch_size=args.batch_size,seed=args.seed,algorithm=SCHEMA,
                  no_expert_demonstrations=True,no_hidden_map=True)
        model=HindsightSkillModel(own_size=own_size,action_sizes=action_sizes,
                                 device=args.device,seed=args.seed)
        epoch=0
        if args.resume:
            loaded=model.torch.load(ckpt,map_location=model.device,weights_only=True)
            if loaded['schema']!=SCHEMA or loaded['metadata']!=meta:
                raise ValueError('Exact goal-learning checkpoint metadata mismatch')
            epoch=int(loaded['epochs'])
            model.restore(loaded['model'])
        with journal.open('a' if args.resume else 'x') as log:
            log.write(json.dumps(dict(stage='data_snapshot',resume=args.resume,
                     current_epoch=epoch,source=stats),sort_keys=True)+'\n');log.flush()
            print('HINDSIGHT DATA',json.dumps(stats,sort_keys=True),flush=True)
            if epoch>=args.epochs:
                print('HINDSIGHT TARGET ALREADY COMPLETED',epoch,flush=True)
                return
            for ix in range(epoch,args.epochs):
                started=time.perf_counter()
                outcome=model.train_examples(train,batch_size=args.batch_size,seed=args.seed+ix)
                atomic_save(ckpt,model,meta,ix+1)
                row=dict(stage='epoch_committed',epoch=ix+1,loss=outcome['mean_weighted_loss'],
                         updates=outcome['updates'],train_imitation_accuracy=accuracy(model,train),
                         validation_imitation_accuracy=accuracy(model,validation),
                         source_committed_cursor=stats['committed_cursor'],
                         trained_on_achieved_self_goals=True,
                         wall_s=round(time.perf_counter()-started,3))
                log.write(json.dumps(row,sort_keys=True)+'\n');log.flush()
                print('HINDSIGHT EPOCH',json.dumps(row,sort_keys=True),flush=True)
            print('HINDSIGHT SAVED',str(ckpt),flush=True)


if __name__=='__main__':main()
