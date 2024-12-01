import numpy as np
from random import randint
from n2v.models import N2V, N2VConfig
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import logging as log 
from tensorflow.keras.optimizers import Adam

log.basicConfig(level=log.INFO)


def get_random_slice(image):
    """Randomly samples a 2D slice from a 3D image."""
    slice_idx = randint(0, image.shape[0] - 1)
    return image[slice_idx, :, :]

def train_n2v_careamics(image,
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
    Train a Noise2Void model using careamics.

    Parameters
    ----------
    images : list of np.ndarray
        List of input images to train the Noise2Void model.
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
    N2V
        The trained Noise2Void model.
    """

    log.info("Starting Noise2Void training using careamics")

    # Prepare the data generator for Noise2Void
    datagen = N2V_DataGenerator()
    
    # Prepare training data for Noise2Void
    training_data = []
    for _ in range(10):  # Sample 10 random slices per image #TODO : change it if we want more data for training
        training_data.append(get_random_slice(image))
    
    # Convert slices into patches for training
    patches = datagen.generate_patches_from_list(
        [np.expand_dims(s, axis=(0,-1)) for s in training_data], 
        shape=(64, 64)  # Patch size
    )
    
    log.info(f"Generated {len(patches)} patches for training")
    
    # Step 3: Noise2Void Training
    # Create and configure the Noise2Void model
    model_name = "n2v_denoiser"
    basedir = "models"


    config = N2VConfig(
            patches,
            unet_kern_size=3,
            train_steps_per_epoch=50, # see more data during each epoch, amount of data the model "sees" during one epoch
            train_epochs=100, # train model for longer period, number of times the model trains over the dataset
            n2v_perc_pix=0.198 # percentage of pixels to mask
        )


    model = N2V(config, name=model_name, basedir=basedir)


    log.info(f"Training configuration: {config}")
    model.prepare_for_training(config)

    # Train the model
    print("Starting training...")
    history = model.train(patches[:90], patches[90:])
    print("Training completed!")

    # Save the model
    model.export_TF()
    print(f"Model saved to: {basedir}/{model_name}")