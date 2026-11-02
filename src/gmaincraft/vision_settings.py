"""Reproducible clean RGB observation settings for the isolated MCio client.

Minecraft 1.21.3 + MCio can expose admin/reset error messages as a large red
text overlay on an 84x84 policy frame, even if sendCommandFeedback=false.
`chatScale:0.0` removes the overlay while leaving hearts, hotbar, world and
crosshair intact. Setting `chatVisibility:2` is NOT equivalent: on this MCio
bridge it caused reset camera observations to stay at yaw/pitch 0. Use full
visibility with zero scale, in this DEDICATED test instance only.
"""
from pathlib import Path
import os
import tempfile


VISUAL_KEYS={'chatVisibility':'0','chatScale':'0.0'}


def ensure_clean_visual_options(instance_dir):
    path=Path(instance_dir)/'options.txt'
    if not path.is_file():
        raise FileNotFoundError(f'MCio instance options file missing: {path}')
    raw=path.read_text(encoding='utf-8')
    lines=raw.splitlines()
    # Idempotent even if previous manual MCio runs produced duplicate keys.
    if all([x for x in lines if x.startswith(k+':')]==[f'{k}:{v}']
           for k,v in VISUAL_KEYS.items()):
        return False
    cleaned=[line for line in lines if line.split(':',1)[0] not in VISUAL_KEYS]
    cleaned.extend(f'{key}:{value}' for key,value in VISUAL_KEYS.items())
    result='\n'.join(cleaned)+'\n'
    temp=None
    try:
        fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.options.',suffix='.tmp')
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(result);f.flush();os.fsync(f.fileno())
        os.chmod(temp,path.stat().st_mode)
        os.replace(temp,path)
    finally:
        if temp and os.path.exists(temp):os.unlink(temp)
    return True
