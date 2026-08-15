import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch


def set_seed_all(seed: int = 0) -> None:
    """Set seed for reproducibility across random, numpy, and torch."""
    if not isinstance(seed, int):
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


def set_benchmark_mode() -> None:
    """Enable benchmark mode for speed (disables full determinism)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename: Union[str, Path]):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
    }
    time.sleep(2)  # to avoid file writing issues
    torch.save(state, filename)


def save_pred(y_true: np.ndarray, y_pred: np.ndarray, filename: Union[str, Path]):
    """Save true and predicted labels as JSON."""
    pred_to_save = {
        "y_true": np.squeeze(y_true).tolist(),
        "y_pred": np.squeeze(y_pred).tolist(),
    }
    with filename.open(mode="w") as f:
        json.dump(pred_to_save, f)


def set_learning_rate(learning_rate: float, optimizer: torch.optim.Optimizer) -> None:
    """Update learning rate inside optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
