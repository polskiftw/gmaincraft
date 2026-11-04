#!/usr/bin/env python3
"""Summarize only PPO-committed self-collected player experience."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.experience_archive import read_episode,observed_new_goals

if len(sys.argv)!=2 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',sys.argv[1]):
    sys.exit('Usage: .venv/bin/python tools/inspect-experience.py RUN_TAG')
run=sys.argv[1]
archive=ROOT/'data-local'/'experience'/run
checkpoint=ROOT/'checkpoints'/f'fresh_stream_{run}.pt'
if not checkpoint.exists():
    sys.exit('No committed source model for this replay tag')
import torch
source=torch.load(checkpoint,map_location='cpu',weights_only=True)
cursor=source['next_episode']
episodes=actions=ticks=bytes_total=0
goals=Counter()
for item in sorted(archive.glob('episode_*.npz')):
    if item.is_symlink() or not re.fullmatch(r'episode_[0-9]{12}\.npz',item.name):
        continue
    if int(item.name[8:20])>=cursor:continue
    metadata,arrays=read_episode(item)
    episodes+=1
    actions+=metadata['count']
    ticks+=metadata['total_game_ticks']
    bytes_total+=item.stat().st_size
    for before,after in zip(arrays['inventory_before'],arrays['inventory_after']):
        b=dict(json.loads(str(before)))
        a=dict(json.loads(str(after)))
        for goal in observed_new_goals(b,a):goals[goal.item_id]+=goal.increase
print(json.dumps(dict(run=run,committed_ppo_cursor=cursor,
    archived_episodes=episodes,neural_actions=actions,minecraft_game_ticks=ticks,
    compressed_bytes=bytes_total,observed_item_gains=dict(sorted(goals.items()))),
    indent=2,sort_keys=True))
