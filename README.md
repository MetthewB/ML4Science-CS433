# ML4Science

## Project Description

ML4Science is a project focused on denoising images using various techniques and models. The project includes raw and processed data, models for denoising, and scripts for training and evaluating the models. The primary goal is to improve the quality of images by reducing noise using state-of-the-art methods.

## Requirements

To run this project, you need to install the dependencies listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

```
ML4Science/
├── data/
│   ├── raw/                # Raw data (original images)
│   │   ├── readme.txt                        # description of the dataset
│   │   ├── Image%03d/                        # folders for different FOVs, ID range from 001 to 120
│   │   │   ├── wf_channel{0,1,2}.npy         # 400 shots of the FOV captured using widefield, each file contains a 400x512x512 uint16 tensor
│   │   │   ├── sim_input_channel{0,1,2}.npy  # SIM inputs for getting the SIM reconstructed results, each file contains a 1536x2560 tensor
│   │   │   └── sim_channel{0,1,2}.npy        # SIM reconstructed results of the 3 channels, each file contains a 1024x1024 tensor
│   ├── processed/          # Processed data (denoised images)
│   │   ├── Gaussian/               # Contains .npy files for all parameters of Image001, channel0
│   │   ├── Median/ 
│   │   ├── NL-Means/ 
│   │   ├── TV-Chambolle/ 
│   │   ├── TV-ISO/ 
│   │   ├── Wavelet/ 
│   │   ├── BM3D/ 
│   │   └── Noise2Noise/ 
│   ├── output/             # CSV files (results of denoised images)
│   │   ├── Gaussian/               # Contains channel by channel and average .csv files of results (metrics)
│   │   ├── Median/ 
│   │   ├── NL-Means/ 
│   │   ├── TV-Chambolle/ 
│   │   ├── TV-ISO/ 
│   │   ├── Wavelet/ 
│   │   ├── BM3D/ 
│   │   └── Noise2Noise/ 
├── models/                 # Models used for TV-ISO and Noise2Noise
│   ├── __init__.py                 # Makes models a package
│   ├── prox_tv_iso.py              
│   ├── tv_iso.py                   # Proximal TV-ISO and TV-ISO model definition
│   ├── drunet.py                   
│   ├── resblock.py                 # DRUNet and ResBlocks model definition
│   ├── noise2noise_weights/        
│   │   └── noise2noise_weights.pth # File store Noise2Noise weights
├── Noise2Noise/            # Files related to Noise2Noise
│   ├── config.json                 # Configuration file
│   ├── convert_dataset_to_h5.py    # Converts dataset to h5 format
│   ├── export_plots_tensorboard.py # Exports plots to TensorBoard
│   ├── metrics.py                  # Computes metrics
│   ├── models/                     # Models directory
│   ├── tensorboard_logs/           
│   ├── tensorboard_plots/          # TensorBoard logs and plots directories
│   ├── train.py                    # Training script
│   ├── trainer.py                  
│   └── w2s_dataset.py              # Trainer and Dataset class
├── Noise2Void/             # Files related to Noise2Void
│   ├── train_n2v.ipynb             # Jupyter notebook for training Noise2Void
│   └── train_n2v.py                # Training script
├── scripts/                # Various utility scripts
│   ├── __init__.py                 # Makes scripts a package
│   ├── utils.py                    # Utility functions
│   ├── helpers.py                  # Helper functions
│   ├── metrics.py                  # Metrics computation script
│   ├── denoiser_selection.py       # Denoiser selection script
│   ├── denoising_pipeline.py       # Denoising pipeline script
│   └── explorations.ipynb          # Plot generation script
├── run.ipynb               # Jupyter notebook to run the project
├── run.py                  # Script to run the project
├── requirements.txt        # Project dependencies
├── LICENSE.txt             # Project license
└── README.md               # Project description and instructions
```

## Running TensorBoard

To launch TensorBoard, follow these steps:

1. Open a terminal and navigate to the `Noise2Noise` directory.
2. Run the following command:

```bash
tensorboard --logdir=tensorboard_logs
```