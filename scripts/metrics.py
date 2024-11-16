"""
Metrics submodule.

This module contains various metrics and a metrics tracking class for evaluating image quality.
"""

from typing import Callable, Optional, Union
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


# Utility Functions
def zero_mean(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Zero the mean of an array."""
    return x - x.mean()

def fix_range(gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Adjust the range of an array based on a reference array."""
    scale_factor = (gt * x).sum() / (x * x).sum()
    return x * scale_factor

def fix(gt: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Zero-mean and range-adjust an array."""
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


# Peak Signal-to-noise Ratio (PSNR) Metrics
def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """Peak Signal-to-Noise Ratio (PSNR)."""
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)

def scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Scale-invariant PSNR."""
    range_param = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = zero_mean(gt) / np.std(gt)
    return psnr(zero_mean(gt_), fix(gt_, pred), range_param)

def avg_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Average PSNR over a batch of images."""
    return np.mean([psnr(target[i], prediction[i], data_range=(np.max(target[i]) - np.min(target[i]))) for i in range(len(target))])


# Running PSNR Tracking Class
class RunningPSNR:
    """Track PSNR across batches during validation."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the tracking variables."""
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor):
        """Update running PSNR stats for a new batch."""
        if self.max is None:
            self.max, self.min = tar.max().item(), tar.min().item()
        else:
            self.max, self.min = max(self.max, tar.max().item()), min(self.min, tar.min().item())

        mse = torch.mean((rec - tar) ** 2, dim=(1, 2, 3))
        self.mse_sum += mse.sum()
        self.N += len(mse)

    def get(self) -> Optional[torch.Tensor]:
        """Calculate the current PSNR value."""
        if self.N == 0:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)
