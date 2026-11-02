"""Recurrent action-effect prediction on stored, self-collected Minecraft RGB.

Compares independently held-out EPISODES to majority-class baselines and a
feed-forward RGB CNN. This is a causal self-supervised perceptual diagnostic,
NOT a deployed Minecraft agent, imitation of experts, or oracle state input.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.recurrent_inverse import episode_action_sequences,make_recurrent_inverse
from gmaincraft.vision_factorized import HEAD_SIZES


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--train-data',required=True)
    p.add_argument('--validation-data',required=True)
    p.add_argument('--encoding',choices=('raw','delta'),default='raw')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seed',type=int,default=7313)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--learning-rate',type=float,default=3e-4)
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):p.error('Invalid tag')
    if args.train_data==args.validation_data:p.error('Train and validation must differ')
    if not 1<=args.epochs<=300:p.error('Invalid epoch count')
    return args


def main(argv=None):
    args=parse_args(argv)
    import torch
    from torch import nn
    cp=ROOT/'checkpoints'/f'visual_recurrent_{args.run_tag}.pt'
    output=ROOT/'logs'/f'visual_recurrent_{args.run_tag}.json'
    if cp.exists() or output.exists():
        raise FileExistsError('Refusing to overwrite recurrent diagnostic')
    train=episode_action_sequences(args.train_data,encoding=args.encoding)
    val=episode_action_sequences(args.validation_data,encoding=args.encoding)
    if not torch.cuda.is_available() and args.device=='cuda':
        raise RuntimeError('No CUDA; use tools/gpu-preflight.py')
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    model=make_recurrent_inverse().to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    def score(sequences):
        model.eval();total=np.zeros(3,dtype=float);count=0;loss_total=0.
        with torch.no_grad():
            for pixels,prev_actions,labels in sequences:
                x=torch.as_tensor(pixels[None],device=args.device)
                p=torch.as_tensor(prev_actions[None],device=args.device)
                y=torch.as_tensor(labels[None],device=args.device,dtype=torch.long)
                logits,_=model(x,p)
                for i,head in enumerate(logits):
                    total[i]+=(head.argmax(-1)==y[:,:,i]).sum().item()
                    loss_total+=nn.functional.cross_entropy(head.flatten(0,1),
                                y[:,:,i].flatten(),reduction='sum').item()
                count+=len(labels)
        return [round(float(x/count),4) for x in total],round(loss_total/count,4)
    baselines=[]
    tr=np.concatenate([y for _,_,y in train]);va=np.concatenate([y for _,_,y in val])
    for i,n in enumerate(HEAD_SIZES):
        majority=int(np.bincount(tr[:,i],minlength=n).argmax())
        baselines.append(round(float(np.mean(va[:,i]==majority)),4))
    initial,initial_loss=score(val)
    print(f'CAUSAL GRU RGB: {len(train)} train episodes, {len(val)} held-out episodes, '
          f'labels {len(tr)}/{len(va)}, device={args.device}, encoding={args.encoding}',flush=True)
    print('BASELINE MAJORITY:',baselines,'initial untrained:',initial,flush=True)
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        order=torch.randperm(len(train)).tolist()
        for index in order:
            pixels,prev_actions,labels=train[index]
            x=torch.as_tensor(pixels[None],device=args.device)
            p=torch.as_tensor(prev_actions[None],device=args.device)
            y=torch.as_tensor(labels[None],device=args.device,dtype=torch.long)
            logits,_=model(x,p)
            loss=sum(nn.functional.cross_entropy(head.flatten(0,1),y[:,:,i].flatten())
                     for i,head in enumerate(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (1,2,5,10,args.epochs) or epoch%10==0:
            accuracy,vl=score(val)
            print(f'EPOCH {epoch}/{args.epochs} train_loss={np.mean(losses):.3f} '
                  f'validation_acc={accuracy} validation_loss={vl}',flush=True)
    final,final_loss=score(val)
    metadata=dict(train_data=args.train_data,validation_data=args.validation_data,
                  training_episodes=len(train),validation_episodes=len(val),
                  train_decisions=len(tr),validation_decisions=len(va),
                  epochs=args.epochs,encoding=args.encoding,device=args.device,
                  from_random_weights=True,uses_only_agent_RGB_and_own_actions=True)
    cp.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='self_action_recurrent_inverse_v1',
                    weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    metadata=metadata),cp)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(metadata=metadata,majority=baselines,
                           untrained=initial,trained=final,
                           untrained_loss=initial_loss,trained_loss=final_loss,
                           wall_seconds=round(time.perf_counter()-started,2)),indent=2)+'\n')
    print('FINAL VALIDATION:',final,'baseline:',baselines,flush=True)
    print('SAVED LOCAL:',cp,output,flush=True)

if __name__=='__main__':main()
