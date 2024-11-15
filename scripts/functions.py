import os
import pandas as pd
from tqdm import tqdm
import time
import psutil
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter, median_filter

from helpers import *


def get_paths():
    """
    Returns the data and output paths relative to the current working directory.
    
    :return: A tuple (data_path, output_path)
    """
    data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data')) + '/'
    output_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'output')) + '/'
    return data_path, output_path


def process_images(data_path, num_images=120, denoiser=None, **denoiser_params):
    """
    Loop through each image and channel, apply the specified denoiser function, 
    and compute PSNR, SSIM, runtime, and RAM usage metrics.
    
    :param data_path: Path to the image data
    :param num_images: Number of images to process
    :param denoiser: Denoiser to apply (e.g., gaussian_filter, median_filter)
    :param denoiser_params: Parameters to pass to the denoiser (e.g., sigma for Gaussian)
    :return: PSNR, SSIM, runtime, and RAM usage metrics
    """
    # Initialize lists to store results for each filter type and channel
    denoiser_results = []

    # Get the current process for tracking RAM usage
    process = psutil.Process()
    
    for i in tqdm(range(num_images), initial=1, total=num_images):
        image_index = str(i + 1).zfill(3)
        for channel in range(3):
            # Load image
            image_channel = load_image(data_path, f'Image{image_index}/wf_channel{channel}.npy')
            
            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image_channel)
            sampled_img = sample_image(image_channel)

            # Measure the start time and RAM before processing
            start_time = time.time()
            ram_before = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            # Apply the denoiser
            denoised_img = denoiser(sampled_img, **denoiser_params)  # Apply the filter here

            # Measure the end time and RAM after processing
            end_time = time.time()
            ram_after = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            runtime = end_time - start_time
            ram_usage = ram_after - ram_before

            # Compute data range
            data_range_img = data_range(ground_truth_img)

            # Calculate PSNR and SSIM
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range_img)

            # Store results including runtime and RAM usage
            denoiser_results.append([image_index, f"{channel}", psnr_denoised, ssim_denoised, runtime, ram_usage])
    
    return denoiser_results


def save_results(denoiser_results, output_path, denoiser_name="Custom Denoiser"):
    """Save results to a CSV file for the specified denoiser."""
    # Create DataFrame
    denoiser_df = pd.DataFrame(denoiser_results, columns=['ImageIndex', 'Channel', 'PSNR', 'SSIM', 'Runtime', 'RAM Usage'])

    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save CSV file with the filter name
    denoiser_df.to_csv(os.path.join(output_path, f'{denoiser_name}_denoiser_results.csv'), index=False)

    return denoiser_df


def compute_averages(denoiser_df, denoiser_name="Custom Denoiser"):
    """Compute average PSNR, SSIM, Runtime, and RAM Usage for each channel and denoiser type."""
    channels = denoiser_df['Channel'].unique()  # Get unique channel values from the DataFrame
    averages = {'DenoiserType': [], 'Channel': [], 'Average PSNR': [], 'Average SSIM': [], 'Average Runtime': [], 'Average RAM Usage': []}

    for channel in channels:
        # Filter DataFrame for the given channel
        channel_df = denoiser_df[denoiser_df['Channel'] == channel]
        
        # Append averages
        averages['DenoiserType'].append(denoiser_name)
        averages['Channel'].append(channel)  # Store channel index
        averages['Average PSNR'].append(channel_df['PSNR'].mean())
        averages['Average SSIM'].append(channel_df['SSIM'].mean())
        averages['Average Runtime'].append(channel_df['Runtime'].mean())
        averages['Average RAM Usage'].append(channel_df['RAM Usage'].mean())
    
    avg_results_df = pd.DataFrame(averages)
    return avg_results_df


def select_denoiser(denoiser_name):
    """
    Select the filter function and its parameters based on the denoiser name.
    
    :param denoiser_name: The name of the denoiser ("Gaussian" or "Median")
    :return: denoiser (denoiser function), denoiser_params (dictionary of parameters)
    """
    if denoiser_name == "Gaussian":
        denoiser = gaussian_filter
        denoiser_params = {'sigma': 2}  # Gaussian filter parameter
    elif denoiser_name == "Median":
        denoiser = median_filter
        denoiser_params = {'size': 2}  # Median filter parameter
    else:
        raise ValueError("Unsupported denoiser.")
    
    return denoiser, denoiser_params
