import numpy as np
import os
import logging as log
import matplotlib.pyplot as plt
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from sklearn.model_selection import train_test_split
from scripts.helpers import *
from scripts.metrics import scale_invariant_psnr
from train_n2v import train_and_predict_n2v
from tqdm import tqdm
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



log.basicConfig(level=log.INFO)

def denoiser_n2v(image, patch_size, batch_size, num_epochs): 
    '''Denoise the image using the Noise2Void model.
    
    Args:
    image: np.ndarray, Image to be denoised
    patch_size: tuple, Size of the patches to be extracted from the image
    batch_size: int, Number of patches to be processed in each batch
    num_epochs: int, Number of epochs to train the model
    '''
    # Sample a slice from the image (1, 512, 512)
    log.info("Sampling a slice from the image...")
    noisy_image, noisy_image_index = sample_image(image) 
    
    log.info("Normalizing the noisy image...")
    normalized_noisy_image = normalize_image(noisy_image)
    
    # Prepare the data for training the model
    # Take all the remaining slices to train the model (399)
    log.info("Preparing data for training the model...")
    filtered_image_indices = np.setdiff1d(np.arange(image.shape[0]), noisy_image_index)
    filtered_image = image[filtered_image_indices]
    normalized_filtered_image = [normalize_image(slice) for slice in filtered_image]
    
    # Split the data into training and validation sets
    log.info("Splitting the data into training and validation sets...")
    X_train, X_val = train_test_split(normalized_filtered_image, test_size=0.2, random_state=42)
    
    # Convert lists to NDArray
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    
    # Train the Noise2Void model
    log.info("Training and predict the Noise2Void model...")
    denoised_image = train_and_predict_n2v(normalized_noisy_image, X_train, X_val, batch_size, num_epochs, patch_size)
    log.info("Training completed successfully")
    normalized_denoised_image = normalize_image(denoised_image)
    log.info("Denoising completed successfully")
    
    return normalized_denoised_image

def main(): 
    log.info("Starting the main script")
    
    for channel in tqdm(range(3), desc="Channels"):
        
        
        results = []
        
        for i in tqdm(range(120), desc=f"Images in channel {channel}"):
            image_index = str(i + 1).zfill(3)
            log.info(f"Processing image {image_index}, channel {channel}")

            # Load image
            image_path = f'data/channel{channel}/Image{image_index}/wf_channel{channel}.npy'
            image = np.load(image_path)
            log.debug(f"Loaded image from {image_path}")

            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image)
            
            denoised_img = denoiser_n2v(image, patch_size=(64, 64), batch_size=512, num_epochs=400)

            # Calculate metrics
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            

            # Append results
            results.append([
                image_index, f"{channel}", psnr_denoised, si_psnr_denoised, ssim_denoised
            ])
    results_df = pd.DataFrame(results, columns=["Image", "Channel", "PSNR", "SI-PSNR", "SSIM"])
    results_df.to_csv("output/n2v_results.csv", index=False)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
   