import numpy as np
import random
import torch
import pytorch_lightning as pl


def setup():
    # seed things for reproducibility
    seed_value = 42
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        cuda_capability = torch.cuda.get_device_capability()
        if cuda_capability >= (8, 0):
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    np.random.seed(seed_value)  # if NumPy is used
    random.seed(seed_value)  # if Python's `random` is used
    pl.seed_everything(seed=seed_value, workers=True)
