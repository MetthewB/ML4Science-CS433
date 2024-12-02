import os
import pandas as pd
from tqdm import tqdm
import time
import psutil
import logging as log
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, denoise_nl_means
from scipy.ndimage import gaussian_filter, median_filter
from IPython.display import display
from scripts.helpers import *
from scripts.denoise_n2v_careamics import denoise_n2v_careamics
from scripts.metrics import scale_invariant_psnr

log.basicConfig(level=log.INFO)

def select_denoiser(denoiser_name):
    """
    Select the filter function and its parameters based on the denoiser name.
    
    :param denoiser_name: The name of the denoiser ("Gaussian", "Median", "TV", "Wavelet", or "NL-Means")
    :return: denoiser (denoiser function), denoiser_params (dictionary of parameters)
    """
    log.info(f"Selecting denoiser: {denoiser_name}")
    
    if denoiser_name == "Gaussian":
        denoiser = gaussian_filter
        denoiser_params = {'sigma': 2}  # Gaussian filter parameter
        log.info("Selected Gaussian filter with parameters: sigma=2")
    elif denoiser_name == "Median":
        denoiser = median_filter
        denoiser_params = {'size': 2}  # Median filter parameter
        log.info("Selected Median filter with parameters: size=2")
    elif denoiser_name == "TV-Chambolle":
        denoiser = denoise_tv_chambolle
        denoiser_params = {'weight': 0.1}  # TV denoiser parameter
        log.info("Selected TV-Chambolle filter with parameters: weight=0.1")
    elif denoiser_name == "Wavelet":
        denoiser = denoise_wavelet
        denoiser_params = {}  # Use default parameters with no changes
        log.info("Selected Wavelet filter with default parameters")
    elif denoiser_name == "NL-Means":
        denoiser = denoise_nl_means
        denoiser_params = {}  # Use default parameters with no changes
        log.info("Selected NL-Means filter with default parameters")
    elif denoiser_name == "Noise2Void":
        denoiser = denoise_n2v_careamics
        denoiser_params = {}  # Use default parameters with no changes
        log.info("Selected Noise2Void filter with default parameters")
    else:
        log.error("Unsupported denoiser")
        raise ValueError("Unsupported denoiser.")
    
    return denoiser, denoiser_params



def process_with_denoiser(denoiser_name, data_path, nb_images, nb_channels, parameter_ranges, disable_progress):
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

    if denoiser_name == "Noise2Void":
        # Extract parameter names and values
        param_names = list(param_config.keys())
        param_values = [param_config[name] for name in param_names]

        # Use zip to iterate over parameter combinations
        for values in zip(*param_values):
            # Create a dictionary to hold the current parameter configuration
            denoiser_params = dict(zip(param_names, values))

            denoiser_results = process_images(
                data_path, nb_images=nb_images, nb_channels=nb_channels, denoiser=denoise_n2v_careamics, disable_progress=disable_progress, **denoiser_params
            )

            # Evaluate and store the results
            for result in denoiser_results:
                result.extend([denoiser_name, str(denoiser_params)])
            all_results.extend(denoiser_results)
        result_filename = f"{denoiser_name}_denoiser_results.csv"
    
    elif param_config["values"]:  # Denoisers with tunable parameters
        param_name = param_config["param_name"]
        for value in tqdm(param_config["values"], disable=disable_progress):
            denoiser_params = {param_name: value}
            denoiser_results = process_images(
                data_path, nb_images=nb_images, nb_channels=nb_channels, denoiser=denoiser, disable_progress=disable_progress, **denoiser_params
            )
            for result in denoiser_results:
                result.extend([denoiser_name, f"{param_name} = {value}"])
            all_results.extend(denoiser_results)
        result_filename = f"{denoiser_name}_denoiser_results.csv"

    else:  # Denoisers with default parameters
        denoiser_params = {}
        denoiser_results = process_images(
            data_path, nb_images=nb_images, nb_channels=nb_channels, denoiser=denoiser, disable_progress=disable_progress, **denoiser_params
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


def process_images(data_path, nb_images, nb_channels, denoiser=None, disable_progress=False, **denoiser_params):
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
    log.info(f"Processing images from {data_path} with {nb_images} images")
    denoiser_results = []
    process = psutil.Process()  # For monitoring memory usage

    for channel in tqdm(range(nb_channels), desc="Channels", disable=disable_progress):
        for i in tqdm(range(nb_images), desc=f"Images in channel {channel}", disable=disable_progress, leave=False):
            image_index = str(i + 1).zfill(3)
            log.info(f"Processing image {image_index}, channel {channel}")

            # Load image
            image_path = f'channel{channel}/Image{image_index}/wf_channel{channel}.npy'
            full_image_path = os.path.join(data_path, image_path)
            image = np.load(full_image_path)
            log.debug(f"Loaded image from {full_image_path}")

            # Generate ground truth and sample image
            ground_truth_img = ground_truth(image)
            sampled_img = normalize_image(sample_image(image))

            # Measure runtime and memory usage
            ram_before = process.memory_info().rss / (1024 ** 2)  # RAM in MB
            start_time = time.time()
            denoised_img = normalize_image(denoiser(sampled_img, **denoiser_params))
            runtime = time.time() - start_time
            ram_after = process.memory_info().rss / (1024 ** 2)  # RAM in MB
            log.info(f"Denoised image in {runtime:.2f} seconds")

            # Calculate metrics
            psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            ram_usage = ram_after - ram_before
            log.info(f"Computed metrics: PSNR={psnr_denoised:.2f}, SSIM={ssim_denoised:.2f}, SI-PSNR={si_psnr_denoised:.2f}, RAM Usage={ram_usage:.2f} MB")

            # Append results
            denoiser_results.append([
                image_index, f"{channel}", psnr_denoised, si_psnr_denoised, ssim_denoised, runtime, ram_usage
            ])

    return denoiser_results


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
    