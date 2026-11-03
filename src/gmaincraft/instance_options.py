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
