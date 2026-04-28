from argparse import Namespace

_SIFE_L = None
_SIFE_NMODULI = 3
_SIFE_N = 4096
_TERMS = None
_UNKNOWN = None
CHECKPOINT_PATH = [
    "train/weights/tinet/best_model_cusin_1.pth",
    "train/weights/tinet/best_model_cusin_2.pth",
    "train/weights/tinet/best_model_cusin_3.pth",
    "train/weights/tinet/best_model_cusin_4.pth"
]

CIFAR10_CHECKPOINT_PATH = [
    "train/weights/cifar10/best_cifar10_cusin_1.pth",
    "train/weights/cifar10/best_cifar10_cusin_2.pth",
    "train/weights/cifar10/best_cifar10_cusin_3.pth",
    "train/weights/cifar10/best_cifar10_cusin_4.pth"
]

def set_params(args: Namespace):
    global _SIFE_L, _TERMS, _UNKNOWN
    _SIFE_L = args.sife_l
    _TERMS = args.terms
    _UNKNOWN = args.unknown

    print(f"SIFE_L: {_SIFE_L}")
    print(f"TERMS: {_TERMS}")
    print(f"UNKNOWN: {_UNKNOWN}")

def get_params():
    return Namespace(
        sife_l=_SIFE_L,
        terms=_TERMS,
        unknown=_UNKNOWN,
        sife_nmoduli=_SIFE_NMODULI,
        sife_n=_SIFE_N
    )