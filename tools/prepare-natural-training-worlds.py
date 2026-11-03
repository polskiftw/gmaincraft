#!/usr/bin/env python3
"""Make diverse VERIFIED Minecraft noise-survival source worlds for RL.

This generates NORMAL Minecraft 1.21.3 worlds by the upstream MCio server,
not a flat scripted tree arena. One deterministic user-chosen random seed per
source. Existing source saves and manifests are never overwritten or deleted.
Safe to rerun after an interrupted generation: exact matching existing worlds
are verified; newly missing worlds are created. This changes only the scoped
mcio-data/world_storage and its local ignored MCio configuration.
"""
import argparse
import json
from pathlib import Path
import random
import re
import sys

from mcio_ctrl import world
from nbt import nbt

ROOT=Path(__file__).resolve().parents[1]
MCIO=ROOT/'mcio-data'


def source_metadata(path):
    data=nbt.NBTFile(str(path/'level.dat'))['Data']
    settings=data['WorldGenSettings']
    return (int(settings['seed'].value),
            settings['dimensions']['minecraft:overworld']['generator']['type'].value,
            data['GameRules']['spawnRadius'].value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True,help='unique label for the generated world cohort')
    p.add_argument('--count',type=int,default=8,help='independent natural seeds: 2..16')
    p.add_argument('--seed',type=int,default=917211,help='repeatable seed of the seed generator')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,20}',a.tag) or not 2<=a.count<=16:
        p.error('Tag must be 1..21 safe characters and count must be 2..16')
    rng=random.Random(a.seed)
    generated=[]
    for index in range(a.count):
        world_seed=rng.randrange(-2**63,2**63)
        name=f'MDL_NaturalSource_{a.tag}_{index:02d}'
        if world_seed in [r['seed'] for r in generated]:raise ValueError('Duplicate seed; choose another root')
        generated.append(dict(source=name,seed=world_seed))
    manifest=ROOT/'logs'/f'natural_sources_{a.tag}.json'
    if manifest.is_file():
        current=json.loads(manifest.read_text())
        if current.get('sources')!=generated:
            raise ValueError('Existing generation manifest differs; use same --seed and --count')
    if a.dry_run:
        print('WORLD SOURCE PLAN',json.dumps(dict(tag=a.tag,sources=generated,
              manifest=str(manifest)),sort_keys=True),flush=True)
        return
    manager=world.WorldManager(mcio_dir=MCIO)
    for entry in generated:
        name,seed=entry['source'],entry['seed']
        location=MCIO/'world_storage'/name
        if location.exists():
            if not (location/'level.dat').is_file():
                raise RuntimeError(f'Incomplete existing world: {name}; inspect before retry')
        else:
            print('CREATE VANILLA NOISE SURVIVAL',name,'seed',seed,flush=True)
            manager.create(name,mc_version='1.21.3',gamemode='survival',
                difficulty='normal',seed=seed,reset_server_properties=True,
                server_properties=dict(**{'level-type':'minecraft:normal',
                    'generate-structures':'true','spawn-protection':'0',
                    'online-mode':'false','server-ip':'127.0.0.1'}))
        if not (location/'level.dat').is_file():
            raise RuntimeError('Minecraft server did not produce actual world metadata: '+name)
        parsed_seed,kind,spawn_radius=source_metadata(location)
        if parsed_seed!=seed or kind!='minecraft:noise':
            raise RuntimeError('Natural generation verification failed: '+name)
        # Spawn radius is fixed for reproducibility, *not* a player command.
        # Same metadata convention as the existing natural benchmark.
        if spawn_radius!='0':
            data=nbt.NBTFile(str(location/'level.dat'))
            data['Data']['GameRules']['spawnRadius'].value='0'
            data.write_file()
        verified=source_metadata(location)
        if verified!=(seed,'minecraft:noise','0'):
            raise RuntimeError('World seed/noise/spawn check failed after setup')
        print('VERIFIED NATURAL SURVIVAL WORLD',name,seed,flush=True)
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():
        with manifest.open('x',encoding='utf8') as out:
            json.dump(dict(tag=a.tag,sources=generated,
                    world_generator='minecraft:noise',game_version='1.21.3',
                    total=len(generated)),out,indent=2)
            out.write('\n')
    print('NATURAL SOURCE LIST',' '.join(item['source'] for item in generated),flush=True)
    print('SAVED',manifest,flush=True)

if __name__=='__main__':main()
