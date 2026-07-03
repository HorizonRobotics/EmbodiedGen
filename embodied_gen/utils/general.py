import random
import logging
import warnings

import numpy as np
import torch

__all__ = ["filter_warnings", "set_seed"]


def filter_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message="To copy construct from a tensor.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Unable to load pointnet2_ops cpp extension.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="TORCH_CUDA_ARCH_LIST is not set.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The given buffer is not writable.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Using torch.cross without specifying the dim arg is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release.*",
        category=UserWarning,
    )
    logging.getLogger("grasp_gen.dataset.renderer").setLevel(logging.ERROR)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
