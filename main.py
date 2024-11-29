import sys
import os
import logging as log

# Configure logging
log.basicConfig(level=log.INFO)


# Add the 'scripts' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

# Now you can import the functions module
from functions import *
from helpers import *


# Define global parameter ranges
PARAMETER_RANGES = {
    "Gaussian": {"param_name": "sigma", "values": [2.5, 3, 3.5]},
    "Median": {"param_name": "size", "values": [2, 3, 5]},
    "TV-Chambolle": {"param_name": "weight", "values": [0.05, 0.1, 0.2]},
    "Wavelet": {"param_name": None, "values": None},  # Use default parameters
    "NL-Means": {"param_name": None, "values": None},  # Use default parameters
    "Noise2Void": {
        "train_steps_per_epoch": [100],  # List of values
        "train_epochs": [400],  # List of values
        "train_batch_size": [512],  # List of values
        "n2v_perc_pix": [0.198],  # List of values
        "n2v_patch_shape": [(64, 64)]  # List of values
    }
}

disable_progress = True  # Set to False to enable progress bar

def main():
    """Main script to evaluate denoisers and save results."""
    log.info("Starting the main script")
    
    # Get paths
    data_path, output_path = get_paths()

    # Choose denoiser
    denoiser_name = "Noise2Void"  # "Gaussian", "Median", "TV-Chambolle", "Wavelet", "NL-Means", or "Noise2Void"
    log.info(f"Chosen denoiser: {denoiser_name}")
    
    # Process images with the chosen denoiser
    log.info("Processing images with the chosen denoiser")
    results_df, result_filename = process_with_denoiser(denoiser_name, data_path, nb_images=120, nb_channels=3, parameter_ranges=PARAMETER_RANGES, disable_progress=disable_progress)

    # Save and display results
    log.info("Saving and displaying results")
    display_styled_results(results_df, output_path, result_filename, title=f"{denoiser_name} Denoiser Results")

    # Compute and save averages
    log.info("Computing and saving averages")
    avg_results = compute_averages(results_df)
    display_styled_results(avg_results, output_path, f"average_{result_filename}", title=f"Average {denoiser_name} Results")

    log.info("Main script finished")

if __name__ == "__main__":
    main()
