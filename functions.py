import os
import pandas as pd
from tqdm import tqdm
import time
import psutil
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, denoise_nl_means
from scipy.ndimage import gaussian_filter, median_filter

import sys, os
from IPython.display import display

from n2v.denoise_n2v import *


from scripts.helpers import *
from scripts.metrics import scale_invariant_psnr


def select_denoiser(denoiser_name):
    """
    Select the filter function and its parameters based on the denoiser name.
    
    :param denoiser_name: The name of the denoiser ("Gaussian", "Median", "TV", "Wavelet", or "NL-Means")
    :return: denoiser (denoiser function), denoiser_params (dictionary of parameters)
    """
    if denoiser_name == "Gaussian":
        denoiser = gaussian_filter
        denoiser_params = {'sigma': 2}  # Gaussian filter parameter
    elif denoiser_name == "Median":
        denoiser = median_filter
        denoiser_params = {'size': 2}  # Median filter parameter
    elif denoiser_name == "TV-Chambolle":
        denoiser = denoise_tv_chambolle
        denoiser_params = {'weight': 0.1}  # TV denoiser parameter
    elif denoiser_name == "Wavelet":
        denoiser = denoise_wavelet
        denoiser_params = {}  # Use default parameters with no changes
    elif denoiser_name == "NL-Means":
        denoiser = denoise_nl_means
        denoiser_params = {}  # Use default parameters with no changes
    elif denoiser_name == "Noise2Void":
        denoiser = denoise_n2v
        denoiser_params = {}  # Use default parameters with no changes
    else:
        raise ValueError("Unsupported denoiser.")
    
    return denoiser, denoiser_params


def process_images(data_path, num_images, denoiser=None, disable_progress=False, **denoiser_params):
    """
    Loop through each image and channel, apply the specified denoiser function, 
    and compute PSNR, SI-PSNR, SSIM, runtime, and RAM usage metrics.
    
    :param data_path: Path to the image data
    :param num_images: Number of images to process
    :param denoiser: Denoiser to apply (e.g., gaussian_filter, median_filter)
    :param disable_progress: Whether to disable the progress bar (default: False)
    :param denoiser_params: Parameters to pass to the denoiser (e.g., sigma for Gaussian)
    :return: Metrics results including PSNR, SI-PSNR, SSIM, runtime, and RAM usage
    """
    denoiser_results = []
    process = psutil.Process()  # For monitoring memory usage

    for i in tqdm(range(num_images), disable=disable_progress):
        for channel in range(1):
            print(f"Nb images : {i}")
            image_index = str(i + 1).zfill(3)
            
            print(f"Nb channel : {channel}")
            # Load image
            image_channel = load_image(data_path, f'channel{channel}/Image{image_index}/wf_channel{channel}.npy')
            
            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image_channel)
            
            sampled_img = normalize_image(sample_image(image_channel))
            
            # Measure runtime and memory usage
            start_time = time.time()
            ram_before = process.memory_info().rss / (1024 ** 2)  # RAM in MB
            
            denoised_img =  normalize_image(denoise_n2v(sampled_img)) #denoiser(sampled_img, **denoiser_params)
            
            runtime = time.time() - start_time
            ram_after = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            # Calculate metrics
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            ram_usage = ram_after - ram_before

            print(f"PSNR : {psnr_denoised}")
            print(f"SI-PSNR : {si_psnr_denoised}")
            print(f"SSIM : {ssim_denoised}")
            
            # Append results
            denoiser_results.append([
                image_index, f"{channel}", psnr_denoised, si_psnr_denoised, ssim_denoised, runtime, ram_usage
            ])
    
    return denoiser_results


def process_with_denoiser(denoiser_name, data_path, num_images, parameter_ranges, disable_progress):
    """
    Processes images with the specified denoiser.
    
    :param denoiser_name: Name of the denoiser ("Gaussian", "Median", etc.)
    :param data_path: Path to the dataset.
    :param num_images: Number of images to process.
    :param parameter_ranges: Dictionary defining parameter ranges for each denoiser.
    :param disable_progress: Whether to disable progress bar.
    :return: Results DataFrame and output filename.
    """
    all_results = []

    # Select denoiser and parameters
    denoiser, denoiser_params = select_denoiser(denoiser_name)
    param_config = parameter_ranges[denoiser_name]

    if param_config["values"]:  # Denoisers with tunable parameters
        param_name = param_config["param_name"]
        for value in tqdm(param_config["values"], disable=disable_progress):
            denoiser_params = {param_name: value}
            denoiser_results = process_images(
                data_path, num_images=num_images, denoiser=denoiser, disable_progress=disable_progress, **denoiser_params
            )
            for result in denoiser_results:
                result.extend([denoiser_name, f"{param_name} = {value}"])
            all_results.extend(denoiser_results)
        result_filename = f"{denoiser_name}_denoiser_results.csv"

    else:  # Denoisers with default parameters
        denoiser_params = {}
        denoiser_results = process_images(
            data_path, num_images=num_images, denoiser=denoiser, disable_progress=disable_progress, **denoiser_params
        )
        for result in denoiser_results:
            result.extend([denoiser_name, "Default parameters"])
        all_results.extend(denoiser_results)
        result_filename = f"{denoiser_name}_denoiser_results_default.csv"

                

    # Convert results to a DataFrame
    results_df = pd.DataFrame(all_results, columns=[
        'ImageIndex', 'Channel', 'PSNR', 'SI-PSNR', 'SSIM', 'Runtime', 'RAM Usage', 'DenoiserType', 'Parameter'
    ])
    results_df = results_df[['DenoiserType', 'Parameter', 'ImageIndex', 'Channel', 
                             'PSNR', 'SI-PSNR', 'SSIM', 'Runtime', 'RAM Usage']]
    return results_df, result_filename


def compute_averages(results_df):
    """
    Compute averages for each combination of denoiser and parameter.
    
    :param results_df: DataFrame containing all results
    :return: DataFrame of average results
    """
    # Ensure we compute averages for all expected numeric columns, including SI-PSNR
    return (
        results_df.groupby(['DenoiserType', 'Parameter', 'Channel'], as_index=False)[
            ['PSNR', 'SI-PSNR', 'SSIM', 'Runtime', 'RAM Usage']
        ]
        .mean()
        .reset_index(drop=True)
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
        'SI-PSNR': "{:.2f}",
        'SSIM': "{:.4f}",
        'Runtime': "{:.4f} s",
        'RAM Usage': "{:.2f} MB"
    }).background_gradient(subset=['PSNR', 'SI-PSNR', 'SSIM', 'Runtime', 'RAM Usage'])

    # Display styled DataFrame
    print(f"\n{title}:")
    display(styled_df)

    # Save DataFrame to CSV
    output_file_path = os.path.join(output_path, output_file)
    df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")
    