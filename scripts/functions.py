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

def process_images(data_path, num_images=120, denoiser=None, disable_progress=False, **denoiser_params):
    """
    Loop through each image and channel, apply the specified denoiser function, 
    and compute PSNR, SSIM, runtime, and RAM usage metrics.
    
    :param data_path: Path to the image data
    :param num_images: Number of images to process
    :param denoiser: Denoiser to apply (e.g., gaussian_filter, median_filter)
    :param disable_progress: Whether to disable the progress bar (default: False)
    :param denoiser_params: Parameters to pass to the denoiser (e.g., sigma for Gaussian)
    :return: PSNR, SSIM, runtime, and RAM usage metrics
    """
    denoiser_results = []
    process = psutil.Process()  # For monitoring memory usage

    for i in tqdm(range(num_images), disable=disable_progress):
        image_index = str(i + 1).zfill(3)
        for channel in range(3):
            # Load image
            image_channel = load_image(data_path, f'Image{image_index}/wf_channel{channel}.npy')
            
            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image_channel)
            sampled_img = sample_image(image_channel)

            # Measure runtime and memory usage
            start_time = time.time()
            ram_before = process.memory_info().rss / (1024 ** 2)  # RAM in MB
            denoised_img = denoiser(sampled_img, **denoiser_params)
            runtime = time.time() - start_time
            ram_after = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            # Calculate metrics
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            ram_usage = ram_after - ram_before

            # Append results
            denoiser_results.append([image_index, f"{channel}", psnr_denoised, ssim_denoised, runtime, ram_usage])
    
    return denoiser_results

def compute_averages(results_df):
    """
    Compute averages for each combination of denoiser and parameter.
    
    :param results_df: DataFrame containing all results
    :return: DataFrame of average results
    """
    return (
        results_df.groupby(['DenoiserType', 'Parameter', 'Channel'])
        .mean(numeric_only=True)
        .reset_index()
    )

def display_styled_results(df, output_path, output_file, title):
    """
    Display styled DataFrame and save to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to display and save.
        output_path (str): Output directory path.
        output_file (str): Filename for saving the CSV.
        title (str): Title to display before the DataFrame.
    """
    # Format and style the DataFrame
    styled_df = df.style.format({
        'PSNR': "{:.2f}",
        'SSIM': "{:.4f}",
        'Runtime': "{:.4f} s",
        'RAM Usage': "{:.2f} MB"
    }).background_gradient(subset=['PSNR', 'SSIM', 'Runtime', 'RAM Usage'])

    # Display styled DataFrame
    print(f"\n{title}:")
    display(styled_df)

    # Save DataFrame to CSV
    output_file_path = os.path.join(output_path, output_file)
    df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")


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