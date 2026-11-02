"""CUDA self-supervised image-motion action predictor from logged MCio data.

This model learns inverse dynamics from random Minecraft actions and images:
frame(t), frame(t+1) -> action(t) (move/look/attack). No oracle labels,
pretrained third-party weights, video demonstrations, hidden world state.

Entire EPISODES must be held out; NEVER split adjacent frames randomly.
CNN visual trunk can later initialize a from-scratch RL policy trained
entirely from the agent's own observations. Predictor accuracy is NOT wood
acquisition and not a claim that this pretraining helps downstream RL.
"""
import argparse
import json
from collections import Counter
import re
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.inverse_dynamics import (load_rgb_action_pairs,
                                          make_action_predictor,INVERSE_SCHEMA,
                                          compute_action_metrics,load_encoder_checkpoint)
from gmaincraft.vision_factorized import HEAD_SIZES


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--train-data',required=True)
    p.add_argument('--validation-data',required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=88777)
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    p.add_argument('--objective',choices=('all','movement'),default='all')
    p.add_argument('--pixel-encoding',choices=('raw','delta'),default='raw')
    p.add_argument('--init-visual',default=None,
                   help='optional local self-supervised encoder checkpoint, no external models')
    a=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',a.run_tag):p.error('Invalid tag')
    if a.train_data==a.validation_data:p.error('Train and validation must be different directories')
    if not 1<=a.epochs<=1000 or not 1<=a.batch_size<=512 or not 0<a.learning_rate<1:
        p.error('Invalid hyperparameters')
    return a


def main(argv=None):
    args=parse_args(argv)
    import torch
    from torch import nn
    dest=ROOT/'checkpoints'/f'visual_inverse_{args.run_tag}.pt'
    result_path=ROOT/'logs'/f'visual_inverse_{args.run_tag}.json'
    if dest.exists() or result_path.exists():
        raise FileExistsError('Refusing to overwrite inverse-dynamics results')
    train_dir=Path(args.train_data)
    val_dir=Path(args.validation_data)
    x_train,y_train,tr_episodes=load_rgb_action_pairs(train_dir,pixel_encoding=args.pixel_encoding)
    x_val,y_val,val_episodes=load_rgb_action_pairs(val_dir,pixel_encoding=args.pixel_encoding)
    if set(tr_episodes)&set(val_episodes) and train_dir.resolve()==val_dir.resolve():
        raise RuntimeError('Training-validation episode contamination')
    # Verify disjoint source file content, even if names/tags differ.
    def dataset_hashes(folder):
        return {hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(folder).glob('rollout_*.npz')}
    if dataset_hashes(train_dir)&dataset_hashes(val_dir):
        raise RuntimeError('Train and validation share recorded rollout files')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else args.device
    if device=='auto':device='cpu'
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run tools/gpu-preflight.py')
    selected_heads=(0,) if args.objective=='movement' else (0,1,2)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    net=make_action_predictor().to(device)
    if args.init_visual:
        prior,prior_meta=load_encoder_checkpoint(args.init_visual)
        net.visual.load_state_dict(prior)
        print('INIT VISUAL FROM OWN RGB/ACTIONS:',args.init_visual,
              'prior objective:',prior_meta.get('objective'),flush=True)
    optim=torch.optim.Adam(net.parameters(),lr=args.learning_rate)
    # Keep uint8 screen buffers in host RAM; transfer one GPU batch at a time.
    def tensor(x):return torch.as_tensor(x,dtype=torch.uint8,device=device)
    ytr=torch.as_tensor(y_train,dtype=torch.long,device=device)
    yva=torch.as_tensor(y_val,dtype=torch.long,device=device)
    def evaluate():
        net.eval()
        correct=np.zeros(3,dtype=np.float64)
        losses=[]
        with torch.no_grad():
            for start in range(0,len(x_val),args.batch_size):
                end=start+args.batch_size
                logits,_=net(tensor(x_val[start:end]))
                labels=yva[start:end]
                for i,head in enumerate(logits):
                    correct[i]+=(head.argmax(-1)==labels[:,i]).sum().item()
                losses.append(sum(nn.functional.cross_entropy(logits[i],labels[:,i],reduction='sum')
                                  for i in selected_heads).item())
        return (correct/len(y_val)).round(4).tolist(),round(float(sum(losses)/len(y_val)),4)
    baseline=[]
    for k,n in enumerate(HEAD_SIZES):
        counts=np.bincount(y_train[:,k],minlength=n)
        majority=int(counts.argmax())
        base=float(np.mean(y_val[:,k]==majority))
        baseline.append(dict(classes=n,train_majority_class=majority,
                             validation_majority_accuracy=round(base,4),
                             uniform_guess=round(1/n,4)))
    before_acc,before_loss=evaluate()
    print('INPUT:',len(x_train),'train pairs',len(set(tr_episodes)),'episodes;',
          len(x_val),'validation pairs',len(set(val_episodes)),'episodes; CUDA:',device,flush=True)
    print('MAJORITY BASELINES:',baseline,flush=True)
    print('UNTRAINED accuracy:',before_acc,'loss:',before_loss,flush=True)
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        net.train()
        permutation=torch.randperm(len(x_train),device=device)
        loss_sum=0.
        for pick in permutation.split(args.batch_size):
            logits,_=net(tensor(x_train[pick.cpu().numpy()]))
            labels=ytr[pick]
            loss=sum(nn.functional.cross_entropy(logits[i],labels[:,i])
                     for i in selected_heads)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step()
            loss_sum+=float(loss.item())*len(pick)
        if epoch in (1,2,5,10,args.epochs) or epoch%10==0:
            va,vl=evaluate()
            print(f'EPOCH {epoch}/{args.epochs}: train_crossentropy={loss_sum/len(x_train):.4f} '
                  f'validation_acc={va} validation_crossentropy={vl}',flush=True)
    after_acc,after_loss=evaluate()
    dest.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(train_data=str(train_dir),validation_data=str(val_dir),
                  n_train=int(len(x_train)),n_validation=int(len(x_val)),
                  episodes_train=len(set(tr_episodes)),episodes_validation=len(set(val_episodes)),
                  seed=args.seed,epochs=args.epochs,device=device,objective=args.objective,
                  initialized_from=args.init_visual or 'random',
                  pixel_encoding=args.pixel_encoding,
                  initialization='random_weights_then_own_Minecraft_RGB_action_pairs',
                  output_is_not_Minecraft_speedrunning_skill=True)
    torch.save(dict(schema=INVERSE_SCHEMA,visual={k:v.cpu() for k,v in net.visual.state_dict().items()},
                    metadata=metadata),dest)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    results=dict(schema=INVERSE_SCHEMA,model=str(dest),metadata=metadata,
                 baseline=baseline,untrained_val_accuracy=before_acc,
                 trained_val_accuracy=after_acc,untrained_loss=before_loss,
                 trained_loss=after_loss,wall_seconds=round(time.perf_counter()-started,2))
    result_path.write_text(json.dumps(results,indent=2)+'\n')
    print('FINAL HELDOUT VALIDATION:',results['trained_val_accuracy'],
          'baseline_majority:',[x['validation_majority_accuracy'] for x in baseline],flush=True)
    print('VISUAL ENCODER SAVED (LOCAL):',dest,flush=True)
    print('RESULTS:',result_path,flush=True)

if __name__=='__main__':main()
