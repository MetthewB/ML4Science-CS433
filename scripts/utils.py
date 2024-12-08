from typing import Union
import numpy as np
import torch

def _zero_mean(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Zero the mean of an array."""
    return x - x.mean()

def _fix_range(gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Adjust the range of an array based on a reference array."""
    a = (gt * x).sum() / (x * x).sum()
    return x * a

def _fix(gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Zero-mean and range-adjust an array."""
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))