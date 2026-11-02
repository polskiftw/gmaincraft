"""Test the isolated shell's CUDA device and optional torch backend."""
import ctypes
from pathlib import Path
import subprocess
import sys


def main():
    print('GPU PREFLIGHT (sandbox only)')
    try:
        result = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                 '--format=csv,noheader'], capture_output=True,text=True,
                                timeout=10,check=True)
        print('NVIDIA card / graphics driver:',result.stdout.strip())
    except Exception as e:
        print('FAIL: NVIDIA management API unavailable:', type(e).__name__, str(e)[:100])
    uvm = Path('/dev/nvidia-uvm')
    print('NVIDIA UVM device:', 'present' if uvm.exists() else 'MISSING')
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        code = driver.cuInit(0)
        print('CUDA driver cuInit:',code,('(success)' if code == 0 else '(failure)'))
    except Exception as e:
        print('CUDA driver cuInit: failure',repr(e))
    try:
        import torch
        print('PyTorch build:',torch.__version__,'CUDA runtime:',torch.version.cuda)
        if not torch.cuda.is_available():
            print('FAIL: PyTorch cannot initialize CUDA')
            return 1
        x=torch.ones((1024,1024),device='cuda')
        print('PASS: CUDA tensor on:',torch.cuda.get_device_name(0),
              'sum:',x.sum().item(),
              'allocated MiB:',round(torch.cuda.memory_allocated()/2**20,1))
        return 0
    except Exception as e:
        print('FAIL: PyTorch CUDA test:',type(e).__name__,str(e)[:400])
        return 1


if __name__ == '__main__':
    sys.exit(main())
