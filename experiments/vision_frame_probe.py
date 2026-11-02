"""Capture an actual policy-visible MCio RGB frame, no private world state."""
import argparse
from pathlib import Path
from PIL import Image
from joint_log_ppo import launch_world
from gmaincraft.joint_curriculum import LogCase

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    args.frame_width=320;args.frame_height=180
    def probe(arena):
        ob=arena.reset(LogCase(0.,0.,0,2,13.5,29.25))
        rgb=ob.get_frame()
        frame=Image.fromarray(rgb)
        target=ROOT/'artifacts'/f'vision_frame_{args.run_tag}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        frame.save(target)
        print(f'POLICY_RGB_CAPTURE {target} size={frame.size}',flush=True)
    launch_world(f'MDL_VisionFrame_{args.run_tag}',args,probe)

if __name__=='__main__':main()
