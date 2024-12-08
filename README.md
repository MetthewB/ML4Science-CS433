my_denoising_project/
├── data/
│   ├── raw/                # Raw data (original images)
│   |   ├── channel0/
│   |   |   ├── Image001/ 
│   |   |   |    ├── wf_channel0.npy 
│   |   ├── channel1/ 
│   |   |   ├── Image001/ 
│   |   |   |    ├── wf_channel1.npy
│   |   ├── channel2/ 
│   |   |   ├── Image001/ 
│   |   |   |    ├── wf_channel2.npy
│   ├── processed/          # Processed data (denoised images)
│   |   ├── Gaussian/ 
│   |   ├── Median/ 
│   |   ├── NL-Means/ 
│   |   ├── TV-Chambolle/ 
│   |   ├── Wavelet/ 
│   ├── output/ 
│   |   ├── Gaussian/ 
│   |   ├── Median/ 
│   |   ├── NL-Means/ 
│   |   ├── TV-Chambolle/ 
│   |   ├── Wavelet/ 
│   └── README.md           # Description of the data
├── models/
│   ├── __init__.py         # Makes models a package
│   ├── drunet.py           # DRUNet model definition
│   ├── resblock.py         # ResBlocks definition
│   └── other_models.py     # Other model definitions
│   ├── noise2noise_weights/ # Directory to store Noise2Noise weights
│   └── noise2void_weights/  # Directory to store Noise2Void weights
├── scripts/
│   ├── __init__.py         # Makes scripts a package
│   ├── helpers.py          # Helper functions
│   ├── denoiser_selection.py 
│   ├── denoiser_pipeline.py 
│   ├── metric_computation.py 
│   ├── psnr_metrics.py 
│   ├── running_psnr.py 
│   ├── utils.py         
│   ├── train.py            # Training script
│   ├── train_noise2noise.py # Training script for Noise2Noise
│   └── train_noise2void.py  # Training script for Noise2Void
├── tests/
│   ├── __init__.py         # Makes tests a package
├── .gitignore              # Git ignore file
├── requirements.txt        # Project dependencies
├── results.py
├── results.ipynb
├── README.md               # Project description and instructions
└── setup.py                # Setup script for installing the package