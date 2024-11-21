# @inproceedings{krull2019noise2void,
#   title={Noise2void-learning denoising from single noisy images},
#   author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={2129--2137},
#   year={2019}
# }

import numpy as np
from csbdeep.utils.utils import normalize_minmse


def PSNR(gt, img, range):
    """
    Compute Peak Signal-to-Noise Ratio.

    Parameters:
        gt: np.array
            The ground truth target image.
        img: np.array
            The image of interest.
        range: float
            Intensity range e.g. gt.max() - gt.min() used for the PSNR
            computation.
    """
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(range) - 10 * np.log10(mse)


def best_PSNR(gt, img, range):
    """
    Compute best Peak Signal-to-Noise Ratio by normalizing img such that
    MSE is minimized to the gt image.

    Parameters:
        gt: np.array
            The ground truth target image.
        img: np.array
            The image of interest.
        range: float
            Intensity range e.g. gt.max() - gt.min() used for the PSNR
            computation.
    """
    img_n = normalize_minmse(img, gt)
    return PSNR(gt, img_n, range=range)
