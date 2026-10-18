import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    Make random operations more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(42)
    print("Random seed set to 42")