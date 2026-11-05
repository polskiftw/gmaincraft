#!/usr/bin/env python3
"""Keep raw Minecraft/compositor logs while showing only useful trainer pulses.

subprocess.Popen reaps the compositor's negative-signal exit itself, so
bash does not misreport a post-child Gamescope SIGSEGV as training failure.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('console',type=Path)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    cmd=args.command[1:] if args.command and args.command[0]=='--' else args.command
    if not cmd: p.error('No command supplied')
    if args.console.exists():p.error(f'Existing console, refusing overwrite: {args.console}')
    env=os.environ.copy()
    env.setdefault('GMAINCRAFT_GAMESCOPE_BACKEND','headless')
    with args.console.open('x',encoding='utf-8') as output:
        with subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,universal_newlines=True,
                              bufsize=1) as proc:
            for line in proc.stdout:
                output.write(line)
                if (line.startswith(('NUMERIC PICKUP READY',
                        'NUMERIC PICKUP PULSE', 'NUMERIC PICKUP SUMMARY',
                        'Traceback', 'RuntimeError:', 'ValueError:',
                        'AssertionError:', 'FileExistsError:', 'FileNotFoundError:'))):
                    print(line,end='',flush=True)
            returncode=proc.wait()
        output.write(f'OUTER PROCESS RETURN CODE: {returncode}\n')
    # Return 139 for a real SIGSEGV so caller can validate independently.
    return 128-returncode if returncode<0 else returncode

if __name__=='__main__':sys.exit(main())
