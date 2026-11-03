"""Scoped temporary changes to disposable MCio instance graphics settings.

Restores the original *bytes*, not a reconstructed options file. A recoverable
backup remains if Python is abruptly killed before entering the finally block.
Only use on the dedicated MCio DemoInstance, never a user's personal game.
"""
from contextlib import contextmanager
from pathlib import Path
import re


@contextmanager
def temporary_game_options(instance_dir, overrides, *, run_tag):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',run_tag):
        raise ValueError('Unsafe run tag')
    if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,50}',str(key)) or
           not re.fullmatch(r'[A-Za-z0-9_.-]{1,30}',str(value))
           for key,value in overrides.items()):
        raise ValueError('Unsafe Minecraft option key/value')
    if not overrides:
        yield
        return
    instance_dir=Path(instance_dir)
    config=instance_dir/'options.txt'
    backup=instance_dir/f'options.gmaincraft_{run_tag}.backup'
    # Exclusive create prevents unknowingly clobbering someone's crash
    # recovery copy or running two override experiments simultaneously.
    if list(instance_dir.glob('options.gmaincraft_*.backup')):
        raise FileExistsError('Outstanding game-options backup: restore it before another override')
    original=config.read_bytes()
    with backup.open('xb') as output:
        output.write(original)
    try:
        lines=original.decode('utf-8').splitlines()
        lines=[line for line in lines if line.split(':',1)[0] not in overrides]
        lines.extend(f'{key}:{value}' for key,value in overrides.items())
        config.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        yield
    finally:
        config.write_bytes(original)
        backup.unlink()


def restore_orphaned_game_options(instance_dir,overrides,*,run_tag):
    """Recover ONLY this run's verified temporary options after a hard crash.

    Call only while holding the actual *run lock* and before starting a new
    MCio JVM. Never force-restore foreign/edited settings. Normal exits still
    use temporary_game_options()'s existing byte-identical finally block.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',run_tag):
        raise ValueError('Unsafe recovery run tag')
    instance_dir=Path(instance_dir)
    config=instance_dir/'options.txt'
    backup=instance_dir/f'options.gmaincraft_{run_tag}.backup'
    others=[p for p in instance_dir.glob('options.gmaincraft_*.backup') if p!=backup]
    if others:
        raise FileExistsError('Another run left options backup; refusing to touch it')
    if not backup.exists():return False
    original=backup.read_bytes()
    edited=original.decode('utf-8').splitlines()
    edited=[line for line in edited if line.split(':',1)[0] not in overrides]
    edited.extend(f'{key}:{value}' for key,value in overrides.items())
    expected=('\n'.join(edited)+'\n').encode('utf-8')
    if config.read_bytes()!=expected:
        raise RuntimeError('Interrupted MCio options differ from expected override; manual inspection required')
    config.write_bytes(original)
    backup.unlink()
    return True
