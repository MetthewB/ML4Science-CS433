import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from scripts.utils import _zero_mean, _fix

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