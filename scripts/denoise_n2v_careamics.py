from matplotlib import pyplot as plt
import numpy as np
import logging as log
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.utils.metrics import psnr
from scripts.helpers import *

log.basicConfig(level=log.INFO)

def denoise_n2v_careamics(single_image,
                          unet_kern_size=3,
                          train_steps_per_epoch=100,  # Number of steps (batches) per epoch
                          train_epochs=5,  # Number of epochs to train the model
                          train_loss='mse',  # Loss function to use during training
                          batch_norm=True,  # Whether to use batch normalization
                          train_batch_size=4,  # Batch size for training
                          n2v_perc_pix=0.198,  # Percentage of pixels to manipulate for Noise2Void training
                          n2v_patch_shape=(64, 64),  # Shape of the patches to extract from the image
                          n2v_manipulator='uniform_withCP',  # Method to manipulate pixels
                          n2v_neighborhood_radius=5):  # Radius of the neighborhood for Noise2Void training
    """
    Noise2Void filter for denoising a single image using careamics.

    Parameters
    ----------
    single_image : np.ndarray
        The input image to be denoised.
    unet_kern_size : int, optional
        Size of the convolutional kernels in the U-Net architecture. Default is 3.
    train_steps_per_epoch : int, optional
        Number of steps (batches) per epoch. Default is 100.
    train_epochs : int, optional
        Number of epochs to train the model. Default is 5.
    train_loss : str, optional
        Loss function to use during training. Default is 'mse' (Mean Squared Error).
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    train_batch_size : int, optional
        Batch size for training. Default is 4.
    n2v_perc_pix : float, optional
        Percentage of pixels to manipulate for Noise2Void training. Default is 0.198.
    n2v_patch_shape : tuple of int, optional
        Shape of the patches to extract from the image. Default is (64, 64).
    n2v_manipulator : str, optional
        Method to manipulate pixels. Default is 'uniform_withCP'.
    n2v_neighborhood_radius : int, optional
        Radius of the neighborhood for Noise2Void training. Default is 5.

    Returns
    -------
    np.ndarray
        The denoised image.
    """

    log.info("Starting Noise2Void denoising using careamics")

    # Normalize the image to [0, 1]
    log.info("Normalizing the image")
    single_image_normalized = normalize_image(single_image)
    
    # Expand dimensions to simulate batch and channel
    log.info("Expanding dimensions to simulate batch and channel")
    #single_image_expanded = np.expand_dims(single_image_normalized, axis=(0, -1))  # Shape: (1, height, width, 1)

    # Create a configuration for Noise2Void using careamics
    log.info("Creating Noise2Void configuration")
    config = create_n2v_configuration(
        experiment_name="n2v_single_image",
        data_type='array',
        axes='YX',
        n_channels=1,
        patch_size=n2v_patch_shape,
        batch_size=train_batch_size,
        num_epochs=train_epochs
    )

    # Create the CAREamist model
    log.info("Creating the CAREamist model")
    careamist = CAREamist(source=config, work_dir="models")

    # Train the model
    log.info("Training the model")
    careamist.train(train_source=single_image_normalized, val_source=single_image_normalized)
    log.info("Training finished")

    # Apply the model to denoise the image
    log.info("Applying the model to denoise the image")
    denoised_image = careamist.predict(single_image_normalized, axes='YX')  # Predicts denoised image
    log.info("Prediction finished")
    
    return denoised_image