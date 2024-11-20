from functions import *

# Define global parameter ranges
PARAMETER_RANGES = {
    "Gaussian": {"param_name": "sigma", "values": [2.5, 3, 3.5]},
    "Median": {"param_name": "size", "values": [5, 8, 10]},
    "TV-Chambolle": {"param_name": "weight", "values": [0.05, 0.1, 0.2]},
    "Wavelet": {"param_name": None, "values": None},  # Use default parameters
    "NL-Means": {"param_name": None, "values": None},  # Use default parameters
    "Noise2Void": {"param_name": None, "values": None}  # Use default parameters
}

disable_progress = True  # Set to False to enable progress bar

def main():
    """Main script to evaluate denoisers and save results."""
    # Get paths
    data_path, output_path = get_paths()

    # Choose denoiser
    denoiser_name = "Noise2Void"  # "Gaussian", "Median", "TV-Chambolle", "Wavelet", "NL-Means", or "Noise2Void"

    # Process images with the chosen denoiser
    results_df, result_filename = process_with_denoiser(denoiser_name, data_path, num_images=1, parameter_ranges=PARAMETER_RANGES, disable_progress=disable_progress)

    # Save and display results
    display_styled_results(results_df, output_path, result_filename, title=f"{denoiser_name} Denoiser Results")

    # Compute and save averages
    avg_results = compute_averages(results_df)
    display_styled_results(avg_results, output_path, f"average_{result_filename}", title=f"Average {denoiser_name} Results")


if __name__ == "__main__":
    main()
