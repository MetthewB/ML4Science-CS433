import os 
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import logging as log
import tensorflow as tf
from csbdeep.utils import plot_history
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from scripts.helpers import *

log.basicConfig(level=log.INFO)

# Configure TensorFlow to use the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info(f"Using GPU: {gpus}")
    except RuntimeError as e:
        log.error(f"Error setting up GPU: {e}")
else:
    log.warning("No GPU found. Using CPU instead.")


def denoise_n2v(single_image,
                unet_kern_size=3,
                train_steps_per_epoch=100, # Number of steps (batches) per epoch
                train_epochs=5, # Number of epochs to train the model
                train_loss='mse', # Loss function to use during training
                batch_norm=True, # Whether to use batch normalization
                train_batch_size=4, # Batch size for training
                n2v_perc_pix=0.198, # Percentage of pixels to manipulate for Noise2Void training
                n2v_patch_shape=(64, 64), # Shape of the patches to extract from the image
                n2v_manipulator='uniform_withCP', # Method to manipulate pixels
                n2v_neighborhood_radius=5): # Radius of the neighborhood for Noise2Void training
    """
    Noise2Void filter for denoising a single image.

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
    
    log.info("Starting Noise2Void denoising")

    # Normalize the image to [0, 1]
    log.info("Normalizing the image")
    single_image_normalized = normalize_image(single_image)
    
    # Create a data generator
    log.info("Creating data generator")
    datagen = N2V_DataGenerator()
    
    # Add dimension to single image to simulate batch and channel
    log.info("Expanding dimensions to simulate batch and channel")
    single_image_expanded = np.expand_dims(single_image_normalized, axis=(0, -1))  # Shape: (1, 512, 512, 1)

    # Generate patches (Shape: (64, 64))
    log.info("Generating patches")
    patches = datagen.generate_patches_from_list([single_image_expanded], shape=(64, 64))

    # Dynamically split into training and validation sets (e.g., 80% training, 20% validation)
    log.info("Splitting patches into training and validation sets")
    split_idx = int(0.8 * len(patches))
    train_patches = patches[:split_idx]
    val_patches = patches[split_idx:]

    # Define model configuration
    log.info("Defining model configuration")
    config = N2VConfig(
        train_patches,
        unet_kern_size=unet_kern_size,
        train_steps_per_epoch=train_steps_per_epoch, # see more data during each epoch, amount of data the model "sees" during one epoch
        train_epochs=train_epochs, # train model for longer period, number of times the model trains over the dataset
        train_loss=train_loss,
        batch_norm=batch_norm,
        train_batch_size=train_batch_size, # change it, larger batch size can stabilize training
        n2v_perc_pix=n2v_perc_pix,  # Suggested value
        n2v_patch_shape=n2v_patch_shape, # can change the patch sahpe if images are large, can help model learn better features 
        n2v_manipulator=n2v_manipulator,
        n2v_neighborhood_radius=n2v_neighborhood_radius,
    )
    
    # config = N2VConfig(
    #     train_patches,
    #     unet_kern_size=3,
    #     train_steps_per_epoch=100, # see more data during each epoch, amount of data the model "sees" during one epoch
    #     train_epochs=400, # train model for longer period, number of times the model trains over the dataset
    #     train_loss='mse',
    #     batch_norm=True,
    #     train_batch_size=512, # change it, larger batch size can stabilize training
    #     n2v_perc_pix=0.198,  # Suggested value
    #     n2v_patch_shape=(64, 64), # can change the patch sahpe if images are large, can help model learn better features 
    #     n2v_manipulator='uniform_withCP',
    #     n2v_neighborhood_radius=5,
    # )
    
    # config = N2VConfig(
    #     train_patches,
    #     unet_kern_size=3,
    #     train_steps_per_epoch=200, # see more data during each epoch
    #     train_epochs=100, # train model for longer period 
    #     train_loss='mse',
    #     batch_norm=True,
    #     train_batch_size=128, # change it, larger batch size can stabilize training
    #     n2v_perc_pix=0.198,  # Suggested value
    #     n2v_patch_shape=(64, 64), # can change the patch sahpe if images are large, can help model learn better features 
    #     n2v_manipulator='uniform_withCP',
    #     n2v_neighborhood_radius=5,
    # )

    # Create the Noise2Void model
    log.info("Creating the Noise2Void model")
    model = N2V(config, "n2v_single_image", basedir="models")

    # Train the model
    log.info("Training the model")
    history = model.train(X=train_patches, validation_X=val_patches)
    log.info('Training finished')
    
    # Create the output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # TODO : fix this 
    # Save the plot as an image file
    log.info("Saving the training history plot")
    fig = plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'])
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path)
    matplotlib.use('Agg') # to close the png image 
    log.info(f"Training history plot saved to {plot_path}")

    # Apply the model to denoise the image
    log.info("Applying the model to denoise the image")
    denoised_image = model.predict(single_image_expanded.reshape(512, 512), axes='YX')  # Predicts denoised image
    log.info("Prediction finished")
    
    return denoised_image
