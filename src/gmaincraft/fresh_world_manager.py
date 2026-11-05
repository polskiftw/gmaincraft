"""Own-only disposable Minecraft natural-world lifecycle for seed treadmill.

The vanilla Minecraft server creates each new world, never a template whose
chunks are relabeled. Never touch any world outside the uniquely named
MDL_FreshStream_* namespace. A clean partial pause consumes its seed;
a hard-crashed uncommitted seed is retried ONLY after quarantining the old
world and loading the last committed model checkpoint.
"""
import re
from pathlib import Path
from nbt import nbt
from mcio_ctrl import world

PREFIX='MDL_FreshStream_'


def names(run_tag,episode):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',run_tag):
        raise ValueError('Unsafe stream run tag')
    if type(episode) is not int or episode<0 or episode>=10**12:
        raise ValueError('Invalid episode cursor')
    name=f'{PREFIX}{run_tag}_{episode:012d}'
    return name+'S',name+'W'


def verify_noise_world(path,expected_seed,*,require_fixed_spawn=False):
    path=Path(path)
    if type(expected_seed) is not int:
        raise ValueError('Expected seed must be int')
    data=nbt.NBTFile(str(path/'level.dat'))['Data']
    gen=data['WorldGenSettings']['dimensions']['minecraft:overworld']['generator']['type'].value
    seed=int(data['WorldGenSettings']['seed'].value)
    if gen!='minecraft:noise' or seed!=expected_seed:
        raise RuntimeError(f'World generation mismatch at {path.name}: {seed}/{gen} != {expected_seed}/minecraft:noise')
    if require_fixed_spawn and data['GameRules']['spawnRadius'].value!='0':
        raise RuntimeError(f'World has nonzero spawn scatter: {path.name}')
    return seed


def fix_spawn_radius(path,expected_seed):
    """Set ONLY vanilla gamerule spawnRadius=0, to match test starts.

    Vanilla terrain/seed/world spawn XYZ are untouched. This removes
    per-player random scatter around spawn that invalidated earlier paired
    policy-versus-random comparisons on the same Minecraft seed.
    """
    verify_noise_world(path,expected_seed)
    level=Path(path)/'level.dat'
    data=nbt.NBTFile(str(level))
    data['Data']['GameRules']['spawnRadius'].value='0'
    data.write_file()
    verify_noise_world(path,expected_seed,require_fixed_spawn=True)


class FreshWorldManager:
    def __init__(self,mcio_dir,run_tag,instance='DemoInstance',*,paired_source_tag=None):
        self.root=Path(mcio_dir)
        self.run_tag=run_tag
        self.instance=instance
        # Opt-in evaluation-only snapshots: same GENERATED world files for
        # both policy arms, never relabel a different seed or touch user saves.
        if paired_source_tag is not None:
            names(paired_source_tag,0)  # exact safe tag validation
            if paired_source_tag==run_tag:
                raise ValueError('Shared source namespace cannot be a live run tag')
        self.paired_source_tag=paired_source_tag
        self.manager=world.WorldManager(mcio_dir=self.root)

    def paths(self,episode):
        source,live=names(self.run_tag,episode)
        if self.paired_source_tag is not None:
            source=names(self.paired_source_tag,episode)[0]
        return (self.root/'world_storage'/source,
                self.root/'instances'/self.instance/'saves'/live)

    def prepare(self,episode,seed,*,resume=False):
        source,live=names(self.run_tag,episode)
        if self.paired_source_tag is not None:
            source=names(self.paired_source_tag,episode)[0]
        src_path,dest_path=self.paths(episode)
        if resume:
            if not dest_path.is_dir():
                raise FileNotFoundError('Cleanly paused seed world missing; do NOT generate replacement')
            verify_noise_world(dest_path,seed,require_fixed_spawn=True)
            return live
        if dest_path.exists():
            raise FileExistsError('World already exists; do not overwrite or silently resume: '+str(dest_path))
        if src_path.exists():
            # An interrupted generation can leave a complete source to reuse
            # but never re-label an incomplete or wrong-seed source.
            verify_noise_world(src_path,seed,
                               require_fixed_spawn=self.paired_source_tag is not None)
        else:
            self.manager.create(source,mc_version='1.21.3',gamemode='survival',
                difficulty='normal',seed=seed,reset_server_properties=True,
                server_properties={'level-type':'minecraft:normal',
                    'generate-structures':'true','spawn-protection':'0',
                    'online-mode':'false','server-ip':'127.0.0.1'})
            verify_noise_world(src_path,seed)
        fix_spawn_radius(src_path,seed)
        self.manager.copy('storage',source,self.instance,live,force=False)
        verify_noise_world(dest_path,seed,require_fixed_spawn=True)
        # Paired source is an immutable disposable natural-world snapshot:
        # preserve so the other frozen eval arm gets IDENTICAL initial bytes.
        # Never change source after creating it; the live copy is independent.
        if self.paired_source_tag is None:
            self.manager.delete('storage',source)
        return live

    def retire(self,episode,*,keep=False):
        source,live=names(self.run_tag,episode)
        src_path,dest_path=self.paths(episode)
        if not keep and dest_path.is_dir():
            self.manager.delete(self.instance,live)
        if src_path.is_dir() and self.paired_source_tag is None:
            self.manager.delete('storage',source)

    def quarantine(self,episode):
        """Preserve a possibly auto-saved, uncommitted world before retry.

        Never delete unfinished files. Move the owned snapshot out of
        Minecraft's saves and repair only the matching MCio registry entry.
        """
        source,live=names(self.run_tag,episode)
        src_path,dest_path=self.paths(episode)
        quarantine_root=self.root/'aborted_fresh_streams'
        quarantine_root.mkdir(parents=True,exist_ok=True)
        moved=[]
        for path in ((dest_path,) if self.paired_source_tag is not None else
                     (dest_path,src_path)):
            if not path.exists():continue
            dest=quarantine_root/path.name
            if dest.exists():raise FileExistsError('Quarantine target already exists: '+str(dest))
            path.rename(dest)
            moved.append(str(dest))
        # Manager deletion after rename only deregisters the known owned names.
        # It cannot delete the moved files.
        if dest_path.parent.exists():
            self.manager.delete(self.instance,live)
        if src_path.parent.exists() and self.paired_source_tag is None:
            self.manager.delete('storage',source)
        return moved
