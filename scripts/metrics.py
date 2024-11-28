"""
Metrics submodule.

This module contains various metrics and a metrics tracking class for evaluating image quality.
"""

from typing import Callable, Optional, Union
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


# Utility Functions
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

# PSNR and SI-PSNR using NumPy
def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """Peak Signal to Noise Ratio."""
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)

def scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Scale-invariant PSNR."""
    range_param = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = _zero_mean(gt) / np.std(gt)
    return peak_signal_noise_ratio(_zero_mean(gt_), _fix(gt_, pred), data_range=range_param)

# PSNR and SI-PSNR using PyTorch
def compute_psnr(gt, pred, data_range=1):
    """Compute PSNR for a given image and its clean version."""
    gt_np = gt.data.cpu().numpy().astype(np.float32)
    pred_np = pred.data.cpu().numpy().astype(np.float32)
    PSNR = peak_signal_noise_ratio(gt_np, pred_np, data_range=data_range)
    return PSNR

def compute_si_psnr(gt, pred):
    """Compute Scale-Invariant PSNR for a given image and its clean version."""
    gt_np = gt.cpu().numpy().astype(np.float32)
    pred_np = pred.cpu().numpy().astype(np.float32)
    
    # Use the same logic as scale_invariant_psnr
    range_param = (np.max(gt_np) - np.min(gt_np)) / np.std(gt_np)
    gt_ = _zero_mean(gt_np) / np.std(gt_np)
    return peak_signal_noise_ratio(_zero_mean(gt_), _fix(gt_, pred_np), data_range=range_param)

# Average PSNR and SI-PSNR
def avg_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Average PSNR over a batch of images."""
    return np.mean([psnr(target[i], prediction[i], data_range=(np.max(target[i]) - np.min(target[i]))) for i in range(len(target))])

def avg_si_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Average SI-PSNR over a batch of images."""
    return np.mean([scale_invariant_psnr(target[i], prediction[i]) for i in range(len(target))])


# Running PSNR Tracking Class
class RunningPSNR:
    """Compute the running PSNR during validation step in training."""

    def __init__(self):
        """Constructor."""
        self.N = None
        self.mse_sum = None
        self.max = self.min = None
        self.reset()

    def reset(self):
        """Reset the running PSNR computation."""
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor) -> None:
        """Update the running PSNR statistics given a new batch."""
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max is None:
            assert self.min is None
            self.max = ins_max
            self.min = ins_min
        else:
            self.max = max(self.max, ins_max)
            self.min = min(self.min, ins_min)

        mse = (rec - tar) ** 2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum += torch.nansum(elementwise_mse)
        self.N += len(elementwise_mse) - torch.sum(torch.isnan(elementwise_mse))

    def get(self) -> Optional[torch.Tensor]:
        """Get the actual PSNR value given the running statistics."""
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)
