# Image Filtering and Evaluation Project

This project performs image denoising using two different filters: Gaussian and Median. It processes images stored in a specific folder structure, applies the filters, and calculates performance metrics (PSNR and SSIM) for each filter. The results are saved as CSV files, and average results are displayed.

## Project Structure

The project is organized into the following structure:

```
project/
│
├── data/                # Folder containing image data
   ├── channel0/ 
   │   ├── Image001/        # Folder for Image001
   │   │   ├── wf_channel0.npy   # Image data for channel 0
   │   │
   │   ├── Image002/        # Folder for Image002 (and so on up to Image120)
   │   ├── ...
   │   ├── Image120/
   ├── channel1/ 
   │   ├── Image001/        # Folder for Image001
   │   │   ├── wf_channel1.npy   # Image data for channel 1
   │   │
   │   ├── Image002/        # Folder for Image002 (and so on up to Image120)
   │   ├── ...
   │   ├── Image120/
   ── channel2/ 
   │   ├── Image001/        # Folder for Image001
   │   │   ├── wf_channel2.npy   # Image data for channel 2
   │   │
   │   ├── Image002/        # Folder for Image002 (and so on up to Image120)
   │   ├── ...
   │   ├── Image120/
│
├── output/              # Folder for saving output results
│   ├── gaussian_filter_results.csv  # PSNR and SSIM results for Gaussian filter
│   ├── median_filter_results.csv    # PSNR and SSIM results for Median filter
│
├── scripts/             # Folder containing the script files
│   ├── helpers.py       # Helper functions for image loading and filtering
│   ├── functions.py     # Main functions for image processing, saving results, and computing averages
│   ├── main.ipynb       # Jupyter notebook containing the main image processing logic
```

## Overview

1. **Data Folder (`data/`)**:
    - Contains subfolders for each image (Image001 to Image120).
    - Each image folder contains three `.npy` files corresponding to three channels: `wf_channel0.npy`, `wf_channel1.npy`, and `wf_channel2.npy`.

2. **Output Folder (`output/`)**:
    - The results of the filter operations are saved as CSV files:
        - `gaussian_filter_results.csv`: Contains PSNR and SSIM metrics for images processed with the Gaussian filter.
        - `median_filter_results.csv`: Contains PSNR and SSIM metrics for images processed with the Median filter.

3. **Scripts Folder (`scripts/`)**:
    - **`helpers.py`**: Contains helper functions for loading images, applying filters, and calculating metrics.
    - **`functions.py`**: Contains the main logic for processing images, saving results, and calculating averages for the PSNR and SSIM metrics.
    - **`main.ipynb`**: A Jupyter notebook that demonstrates how to use the functions and scripts to process the images and save the results. The notebook also includes the logic to switch between different filters (Gaussian or Median).

## How It Works

### Step 1: Filter Selection and Processing
In the `main.ipynb` or in your main Python script, you can choose between the two filters available:

- **Gaussian filter** (from `scipy.ndimage`)
- **Median filter** (from `scipy.ndimage`)

The filter is applied to each of the three channels of every image in the `data/` folder.

### Step 2: Metrics Calculation
For each processed image, the PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are calculated for each channel and saved to a list.

### Step 3: Saving Results
Once the images are processed and the metrics are computed, the results are saved into CSV files:

- `gaussian_filter_results.csv`
- `median_filter_results.csv`

These CSV files contain the image index, channel, PSNR, and SSIM for each image and filter type.

### Step 4: Averages
The average PSNR and SSIM for each filter (Gaussian or Median) across all images and channels are computed and displayed in the notebook.

## Running the Code

To run the code, follow these steps:

1. **Install dependencies**:
   - Ensure you have the required Python packages installed:
     ```
     pip install numpy pandas scipy scikit-image tqdm
     ```

2. **Prepare the data**:
   - Place the image files (`wf_channel0.npy`, `wf_channel1.npy`, `wf_channel2.npy`) into the appropriate folders under `data/` (Image001 to Image120).

3. **Run the main script or Jupyter notebook**:
   - You can run the Jupyter notebook (`main.ipynb`) to interactively process the images and see the results.
   - Alternatively, you can execute the functions in your Python script (e.g., `main.py`) to process the images.

4. **View Results**:
   - After processing, you will find the results saved as CSV files in the `output/` folder.
   - The average PSNR and SSIM values for each filter are printed to the console.

## Example Usage in `main.ipynb`

```python
# Example of running the filter for Gaussian filter
filter_name = "Gaussian"  # or "Median"
output_path, data_path = get_paths()

# Select the filter function and parameters
filter_fn, filter_params = select_filter(filter_name)

# Process images and get results
filter_results = process_images(data_path, num_images=120, filter_fn=filter_fn, **filter_params)

# Save results
save_results(filter_results, output_path, filter_name=filter_name)

# Compute averages
filter_df = pd.DataFrame(filter_results, columns=['ImageIndex', 'Channel', 'PSNR', 'SSIM'])
avg_results = compute_averages(filter_df, filter_name=filter_name)

# Display results
print(f"Average PSNR and SSIM for {filter_name} filter:")
print(avg_results)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.