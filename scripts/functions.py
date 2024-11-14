import os
import pandas as pd
from tqdm import tqdm
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


def process_images(data_path, num_images=120, filter_fn=None, **filter_params):
    """
    Loop through each image and channel, apply the specified filter function, 
    and compute PSNR and SSIM metrics.
    
    :param data_path: Path to the image data
    :param num_images: Number of images to process
    :param filter_fn: Filter function to apply (e.g., gaussian_filter, median_filter)
    :param filter_params: Parameters to pass to the filter function (e.g., sigma for Gaussian)
    :return: PSNR and SSIM metrics
    """
    # Initialize lists to store results for each filter type and channel
    filter_results = []
    
    for i in tqdm(range(num_images), initial=1, total=num_images):
        image_index = str(i + 1).zfill(3)
        for channel in range(3):
            # Load image
            image_channel = load_image(data_path, f'Image{image_index}/wf_channel{channel}.npy')
            
            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image_channel)
            sampled_img = sample_image(image_channel)

            # Apply the filter (using filter_fn passed in, such as gaussian_filter)
            filtered_img = filter_fn(sampled_img, **filter_params)  # Apply the filter here
            data_range_img = data_range(ground_truth_img)

            # Calculate PSNR and SSIM
            psnr_filtered = peak_signal_noise_ratio(ground_truth_img, filtered_img, data_range=data_range_img)
            ssim_filtered = structural_similarity(ground_truth_img, filtered_img, data_range=data_range_img)

            # Store results
            filter_results.append([image_index, f"Channel {channel}", psnr_filtered, ssim_filtered])
    
    return filter_results


def save_results(filter_results, output_path, filter_name="Custom Filter"):
    """Save results to a CSV file for the specified filter."""
    # Create DataFrame
    filter_df = pd.DataFrame(filter_results, columns=['ImageIndex', 'Channel', 'PSNR', 'SSIM'])

    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save CSV file with the filter name
    filter_df.to_csv(os.path.join(output_path, f'{filter_name}_filter_results.csv'), index=False)

    return filter_df


def compute_averages(filter_df, filter_name="Custom Filter"):
    """Compute average PSNR and SSIM for each channel and filter type."""
    channels = ['Channel 0', 'Channel 1', 'Channel 2']
    averages = {'FilterType': [], 'Channel': [], 'Average PSNR': [], 'Average SSIM': []}

    for channel in channels:
        # Filter averages for the given filter function
        averages['FilterType'].append(filter_name)
        averages['Channel'].append(channel[-1])  # Append channel index only
        averages['Average PSNR'].append(filter_df[filter_df['Channel'] == channel]['PSNR'].mean())
        averages['Average SSIM'].append(filter_df[filter_df['Channel'] == channel]['SSIM'].mean())
    
    avg_results_df = pd.DataFrame(averages)
    return avg_results_df


def select_filter(filter_name):
    """
    Select the filter function and its parameters based on the filter name.
    
    :param filter_name: The name of the filter ("Gaussian" or "Median")
    :return: filter_fn (filter function), filter_params (dictionary of parameters)
    """
    if filter_name == "Gaussian":
        filter_fn = gaussian_filter
        filter_params = {'sigma': 2}  # Gaussian filter parameter
    elif filter_name == "Median":
        filter_fn = median_filter
        filter_params = {'size': 2}  # Median filter parameter
    else:
        raise ValueError("Unsupported filter function.")
    
    return filter_fn, filter_params
