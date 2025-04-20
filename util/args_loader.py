import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="ReAct OOD detection w/ optional FNO1d+RK4"
    )

    # ─── model architecture ────────────────────────────────────────────────
    parser.add_argument(
        '--model-arch',
        choices=['resnet18','resnet50','mobilenet','fno1d'],
        default='resnet50',
        help='architecture to use'
    )
    parser.add_argument(
        '--net-file',
        required=True,
        help='path to checkpoint (.pt for fno1d or .pth.tar for resnets)'
    )

    # ─── FNO1d+RK4 hyperparameters ─────────────────────────────────────────
    parser.add_argument(
        '--time-step', type=float, default=1e-3,
        help='Δt for RK4 integration (only for fno1d)'
    )
    parser.add_argument(
        '--modes',      type=int,   default=256,
        help='number of Fourier modes (only for fno1d)'
    )
    parser.add_argument(
        '--width',      type=int,   default=256,
        help='channel width (only for fno1d)'
    )
    parser.add_argument(
        '--time-history',type=int,  default=1,
        help='history steps (only for fno1d)'
    )
    parser.add_argument(
        '--time-future', type=int,  default=1,
        help='future steps (only for fno1d)'
    )

    # ─── in‑distribution dataset ───────────────────────────────────────────
    parser.add_argument(
        '--in-dataset',
        choices=['CIFAR-10','CIFAR-100','imagenet','rk4'],
        default='CIFAR-10'
    )
    parser.add_argument(
        '--input-file',
        help='path to your RK4 .npy file (only for in-dataset=rk4)'
    )
    parser.add_argument(
        '--skip', type=int, default=1,
        help='subsample factor for RK4 (only for rk4)'
    )

    # ─── out‑of‑distribution & scoring ────────────────────────────────────
    parser.add_argument('--out-datasets', nargs='+', default=[])
    parser.add_argument('--method',
                        choices=['energy','msp','odin'],
                        default='energy')
    parser.add_argument('--threshold', type=float, default=0.0)

    # ─── misc ─────────────────────────────────────────────────────────────
    parser.add_argument('-b','--batch-size', type=int, default=128)
    parser.add_argument('--gpu',         default='0')
    parser.add_argument('--base-dir',    default='output/ood_scores')
    parser.add_argument('--name',        default='experiment')
    parser.add_argument('--epochs',      type=int, default=100)
    parser.add_argument('--lim',         type=int, default=2000)

    return parser.parse_args()
