import torch
import torch.nn as nn
import numpy as np
import logging as log
import pandas as pd
from tqdm import tqdm

from scripts.helpers import *
from scripts.metrics import *
from skimage.metrics import structural_similarity

from Noise2Noise.models.drunet import DRUNet
torch.set_grad_enabled(False)

log.basicConfig(level=log.INFO)

def load_model(model_path):
    """Load a pre-trained model."""
    
    infos = torch.load(model_path, map_location=torch.device('cpu'))
    config = infos['config']

    model = DRUNet(config['net_params']['nb_channels'], config['net_params']['depth'], config['training_options']['color'])
    model.load_state_dict(infos['state_dict'])  # loads the saved model weights into the new model
    model.eval()  # set the model to evaluation mode
    
    return model

def denoiser_n2n(noisy_image, model):
    """Denoise the image using the Noise2Noise model."""
    
    noisy_normalized_image = normalize_image(noisy_image)
    
    noisy_image_reshaped = noisy_normalized_image.reshape(1, 1, 512, 512)  # Reshape the image to (Batch, Channels, Height, Width)
    
    noisy_image_tensor = torch.tensor(noisy_image_reshaped, dtype=torch.float32)
    
    with torch.no_grad():
        denoised_image = model(noisy_image_tensor).view(512, 512).numpy()
    
    denoised_image_normalized = normalize_image(denoised_image)
    
    return denoised_image_normalized


def main(): 
    log.info("Starting the main script")
    
    results = []
    
    for channel in tqdm(range(3), desc="Channels"):
        
        
        for i in tqdm(range(120), desc=f"Images in channel {channel}"):
            
            image_index = str(i + 1).zfill(3)
            log.info(f"Processing image {image_index}, channel {channel}")

            # Load image
            image_path = f'data/channel{channel}/Image{image_index}/wf_channel{channel}.npy'
            image = np.load(image_path)
            log.debug(f"Loaded image from {image_path}")
            
            # Sample a slice from the image (400, 512, 512)
            log.info("Sampling a slice from the image...")
            noisy_image, _ = sample_image(image)
            
            # Load Noise2Noise model
            log.info("Loading Noise2Noise model...")
            model = load_model('Noise2Noise/checkpoint.pth')
            
            # Denoise the image
            log.info("Denoising the image using Noise2Noise denoiser...")
            denoised_img = denoiser_n2n(noisy_image, model)

            # Generate ground truth and sample image
            log.info("Generating ground truth image...")
            ground_truth_img = ground_truth(image)
            
            # Calculate metrics
            log.info("Calculating the metrics...")
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            
            log.info(f"PSNR: {psnr_denoised}, SI-PSNR: {si_psnr_denoised}, SSIM: {ssim_denoised}")

            # Append results
            results.append([
                image_index, f"{channel}", psnr_denoised, si_psnr_denoised, ssim_denoised
            ])
    results_df = pd.DataFrame(results, columns=["Image", "Channel", "PSNR", "SI-PSNR", "SSIM"])
    results_df.to_csv("output/n2n_results.csv", index=False)
    
    
if __name__ == "__main__":
    main()
    