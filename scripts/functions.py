import os
import pandas as pd
from tqdm import tqdm
import random
import time
import psutil
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, denoise_nl_means
from scipy.ndimage import gaussian_filter, median_filter

from helpers import *
from metrics import *
from tv import prox_tv_iso

# ----------------
# PART 1: TABLES
# ----------------
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
    elif denoiser_name == "TV-ISO":
        denoiser = prox_tv_iso
        denoiser_params = {'lmbda': 0.08, 'niter': 200}  # TV-ISO denoiser parameters
    else:
        raise ValueError("Unsupported denoiser.")
    
    return denoiser, denoiser_params


def process_images(data_path, num_images=2, denoiser=None, disable_progress=False, **denoiser_params):
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
        image_index = str(i + 1).zfill(3)
        for channel in range(3):
            # Load image
            image_channel = load_image(data_path, f'Image{image_index}/wf_channel{channel}.npy')
            
            # Generate ground truth and sample image
            ground_truth_img = normalize_image(ground_truth(image_channel))
            sampled_img = normalize_image(sample_image(image_channel))

            # Measure runtime and memory usage
            start_time = time.time()
            ram_before = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            if denoiser == prox_tv_iso:
                prox_tv = denoiser(device='cpu')  # Adjust device as needed
                denoised_img = prox_tv.eval(torch.from_numpy(sampled_img).unsqueeze(0).unsqueeze(0).float(), **denoiser_params).squeeze().numpy()
            else:
                denoised_img = normalize_image(denoiser(sampled_img, **denoiser_params))

            runtime = time.time() - start_time
            ram_after = process.memory_info().rss / (1024 ** 2)  # RAM in MB

            # Calculate metrics
            psnr_denoised = psnr(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_img)
            ssim_denoised = structural_similarity(ground_truth_img, denoised_img, data_range=data_range(ground_truth_img))
            ram_usage = ram_after - ram_before

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
            if denoiser_name == "TV-ISO":
                denoiser_params['niter'] = 200  # Ensure niter is included for TV-ISO
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
    # Create the tables directory if it doesn't exist
    tables_path = os.path.join(output_path, 'tables')
    os.makedirs(tables_path, exist_ok=True)

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
    # output_file_path = os.path.join(tables_path, output_file)
    # df.to_csv(output_file_path, index=False)
    # print(f"Results saved to {output_file_path}")


# ---------------
# PART 2: PLOTS
# ---------------
def get_random_image(data_path):
    """
    Get a random image from the available images in the data folder.

    :param data_path: Path to the image data
    :return: Selected image
    """
    available_images = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not available_images:
        raise ValueError("No images found in the data path.")
    return random.choice(sorted(available_images))


def process_and_denoise_image(data_path, selected_image, denoiser_name, denoiser_params, slice_index):
    """
    Process and denoise a single slice of a 3D image using the specified denoiser.

    :param data_path: Path to the image data
    :param selected_image: Name of the selected image
    :param denoiser_name: Name of the denoiser
    :param denoiser_params: Parameters to pass to the denoiser
    :param slice_index: Index of the slice to process
    :return: Lists of images, noisy images, denoised images, SI-PSNR values for noisy and denoised images
    """
    images = []
    noisy_images = []
    denoised_images = []
    si_psnr_noisy_list = []
    si_psnr_denoised_list = []

    for channel in range(3):
        # Load the 3D image
        image = np.load(os.path.join(data_path, f'{selected_image}/wf_channel{channel}.npy'))

        # Compute the ground truth as the average of all slices
        ground_truth = image.mean(axis=0)

        # Select the specific slice from the 3D image
        image_slice = image[slice_index, :, :]

        # Normalize the ground truth and selected slice
        ground_truth_normalized = normalize_image(ground_truth)
        image_slice_normalized = normalize_image(image_slice)

        # Create a noisy version of the normalized slice
        noise_std = np.std(image_slice_normalized)  # Adjust noise level as needed
        noisy_image = normalize_image(image_slice_normalized + np.random.normal(0, noise_std, image_slice_normalized.shape))  # Example noisy image

        x = torch.from_numpy(image_slice_normalized).to('cpu').float()  # ground truth
        y = torch.from_numpy(noisy_image).to('cpu').float()  # noisy measurements

        if denoiser_name == "TV-ISO":
            prox_tv = prox_tv_iso('cpu')  # construct proximal operator of tv
            denoised_tv = prox_tv.eval(y.unsqueeze(0).unsqueeze(0), **denoiser_params)  # you need to tune the lmbda and niter sufficiently large such that the solution does not change
            denoised_image = denoised_tv.squeeze().numpy()
        else:
            denoiser, _ = select_denoiser(denoiser_name)
            denoised_image = denoiser(noisy_image, **denoiser_params)

        # Normalize the denoised image
        denoised_image_normalized = normalize_image(denoised_image)

        # Calculate SI-PSNR
        si_psnr_noisy = scale_invariant_psnr(ground_truth_normalized, noisy_image)
        si_psnr_denoised = scale_invariant_psnr(ground_truth_normalized, denoised_image_normalized)

        images.append(ground_truth_normalized)
        noisy_images.append(noisy_image)
        denoised_images.append(denoised_image_normalized)
        si_psnr_noisy_list.append(si_psnr_noisy)
        si_psnr_denoised_list.append(si_psnr_denoised)

    return images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list


def plot_denoiser_results(images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list, title, output_path):
    """
    Plot the ground truth, noisy, and denoised images with their SI-PSNR values for multiple channels.

    :param images: List of ground truth images (numpy arrays)
    :param noisy_images: List of noisy images (numpy arrays)
    :param denoised_images: List of denoised images (numpy arrays or torch tensors)
    :param si_psnr_noisy_list: List of SI-PSNR values of the noisy images
    :param si_psnr_denoised_list: List of SI-PSNR values of the denoised images
    :param title: Title for the plot
    :param output_path: Path to save the plot image
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    channels = ['Channel 0', 'Channel 1', 'Channel 2']
    
    for i in range(3):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'{channels[i]} Ground Truth')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_images[i])
        axes[i, 1].set_title(f'Noisy, SI-PSNR={np.round(si_psnr_noisy_list[i], 2)}')
        axes[i, 1].axis('off')

        if isinstance(denoised_images[i], torch.Tensor):
            denoised_image = denoised_images[i][0, 0].detach().cpu().numpy()
        else:
            denoised_image = denoised_images[i]
        
        axes[i, 2].imshow(denoised_image)
        axes[i, 2].set_title(f'Denoised, SI-PSNR={np.round(si_psnr_denoised_list[i], 2)}')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    # Create the plots directory if it doesn't exist
    plots_path = os.path.join(output_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)

    # Save the plot to a file
    # plot_file_path = os.path.join(plots_path, f"{title.replace(' ', '_')}.png")
    # plt.savefig(plot_file_path)
    
    print(f"\n{title}:")
    plt.show()
    # print(f"Plot saved to {plot_file_path}")


def run_denoising_pipeline(data_path, output_path, denoiser_name, parameter_ranges, disable_progress):
    """
    Run the denoising pipeline for the specified denoiser.

    :param data_path: Path to the image data
    :param output_path: Path to save the results
    :param denoiser_name: Name of the denoiser
    :param parameter_ranges: Dictionary defining parameter ranges for each denoiser
    :param disable_progress: Whether to disable the progress bar
    """
    # Get a random image from the available images
    selected_image = get_random_image(data_path)

    # Process images with the chosen denoiser
    results_df, result_filename = process_with_denoiser(denoiser_name, data_path, num_images=2, parameter_ranges=parameter_ranges, disable_progress=disable_progress)

    # Save and display results
    display_styled_results(results_df, output_path, result_filename, title=f"{denoiser_name} Denoiser Results")

    # Compute and save averages
    avg_results = compute_averages(results_df)
    display_styled_results(avg_results, output_path, f"average_{result_filename}", title=f"Average {denoiser_name} Results")

    # Get the denoiser parameters for the plot
    denoiser, denoiser_params = select_denoiser(denoiser_name)
    param_config = parameter_ranges[denoiser_name]

    # Choose a random slice index once for all hyperparameter values
    slice_index = random.randint(0, 399)

    if param_config["values"]:
        param_name = param_config["param_name"]
        for param_value in param_config["values"]:
            denoiser_params = {param_name: param_value}
            if denoiser_name == "TV-ISO":
                denoiser_params['niter'] = 200  # Ensure niter is included for TV-ISO

            # Process and denoise a single image slice
            images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list = process_and_denoise_image(data_path, selected_image, denoiser_name, denoiser_params, slice_index)

            # Plot results for the chosen denoiser
            plot_title = f"{selected_image} (Slice {slice_index}) - {denoiser_name} ({param_name}={param_value})"
            plot_denoiser_results(images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list, title=plot_title, output_path=output_path)
    else:
        # Process and denoise a single image slice with default parameters
        images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list = process_and_denoise_image(data_path, selected_image, denoiser_name, denoiser_params, slice_index)

        # Plot results for the chosen denoiser
        plot_title = f"{selected_image} (Slice {slice_index}) - {denoiser_name} (Default parameters)"
        plot_denoiser_results(images, noisy_images, denoised_images, si_psnr_noisy_list, si_psnr_denoised_list, title=plot_title, output_path=output_path)