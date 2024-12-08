import numpy as np
from skimage.metrics import structural_similarity
from scripts.psnr_metrics import psnr, scale_invariant_psnr
from scripts.helpers import normalize_image, data_range
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def compute_metrics(denoised_image, ground_truth_image):
    '''Compute PSNR, SI-PSNR, and SSIM for the denoised image.'''
    
    log.info(f"Computing metrics for denoised image.")
    
    denoised_image = normalize_image(denoised_image)

    psnr_denoised = psnr(ground_truth_image, denoised_image, data_range=data_range(ground_truth_image))
    si_psnr_denoised = scale_invariant_psnr(ground_truth_image, denoised_image)
    ssim_denoised = structural_similarity(ground_truth_image, denoised_image, data_range=data_range(ground_truth_image))
    
    return [psnr_denoised, si_psnr_denoised, ssim_denoised]