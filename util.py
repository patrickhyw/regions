import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set global random seed for reproducibility."""
    np.random.seed(seed)
