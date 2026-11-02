import argparse
import os
from . import list_profiles


def main():
    ap=argparse.ArgumentParser(prog='nvfp4_training')
    cmds=ap.add_subparsers(dest='command',required=True)
    cmds.add_parser('list',help='print the accepted profile names')
    cmds.add_parser('verify',help='run reference parity, gradient and convergence gates on this GPU')
    check=cmds.add_parser('check',help='verify real GPU updates for one or all profiles')
    check.add_argument('--profile',choices=list_profiles())
    check.add_argument('--output',help='optional JSON destination')
    b=cmds.add_parser('bench',help='measure full public-API training steps')
    b.add_argument('--profile',choices=list_profiles(),default='balanced')
    b.add_argument('--precision',choices=('fp4','bf16'),default='fp4')
    b.add_argument('--task',choices=('453m','fineweb'),default='453m')
    b.add_argument('--remat',action='store_true')
    b.add_argument('--batch',type=int,default=2)
    b.add_argument('--seq',type=int)
    b.add_argument('--seed',type=int)
    b.add_argument('--warmup',type=int,default=12)
    b.add_argument('--steps',type=int,default=30,help='timed steps per round')
    b.add_argument('--rounds',type=int,default=5)
    b.add_argument('--output',help='optional JSON destination')
    a=ap.parse_args()
    if a.command=='list':
        print('\n'.join(list_profiles()))
    elif a.command=='verify':
        from ._runtime import load_runtime
        load_runtime()
        from ._verification.verify_fast_v2 import main as verify
        verify()
    elif a.command=='check':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        from .selftest import run
        run((a.profile,) if a.profile else list_profiles(),a.output)
    else:
        if a.seq is None:a.seq=64 if a.task=='453m' else 512
        if a.seed is None:a.seed=1234 if a.task=='453m' else 2031
        if min(a.batch,a.seq,a.warmup,a.steps,a.rounds)<1:ap.error('sizes and iteration counts must be positive')
        if a.precision=='bf16' and a.remat:ap.error('--remat applies only to fp4')
        from .benchmark import run
        run(a)


if __name__=='__main__':main()
