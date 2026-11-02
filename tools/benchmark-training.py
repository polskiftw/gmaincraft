"""Reproducible local CPU/CUDA conv-net training throughput microbenchmark.

Measures the pixel network's batched optimizer updates, NOT Minecraft FPS,
trajectory acquisition, full PPO convergence, or general agent performance.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.vision_ppo import make_network, HISTORY, IMAGE_SIZE


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args(argv)
    if not 1<=args.batch_size<=512 or not 1<=args.steps<=1000 or not 1<=args.threads<=64:
        parser.error('Invalid batch size/steps/threads')
    import torch
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else args.device
    if device=='auto': device='cpu'
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run tools/gpu-preflight.py first')
    torch.set_num_threads(args.threads)
    torch.manual_seed(4)
    net=make_network().to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=.00025)
    batch=torch.randint(0,256,(args.batch_size,HISTORY*3,IMAGE_SIZE,IMAGE_SIZE),
                        device=device,dtype=torch.uint8)
    target=torch.randn(args.batch_size,device=device)
    def one_step():
        logits, values=net(batch)
        loss=values.sub(target).square().mean()-.01*torch.distributions.Categorical(
            logits=logits).entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for _ in range(3): one_step()
    if device=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(args.steps): one_step()
    if device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    print(f'MODEL TRAINING ONLY: torch={torch.__version__} device={device} '
          f'params={sum(x.numel() for x in net.parameters())} '
          f'updates={args.steps} batch={args.batch_size} '
          f'total_s={elapsed:.3f} updates_per_s={args.steps/elapsed:.2f} '
          f'images_per_s={args.steps*args.batch_size/elapsed:.1f}')
    if device=='cuda':
        print(f'CUDA peak MB={torch.cuda.max_memory_allocated()/2**20:.1f} '
              f'GPU={torch.cuda.get_device_name(0)}')

if __name__=='__main__':main()
