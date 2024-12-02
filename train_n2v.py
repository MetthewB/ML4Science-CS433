import numpy as np
import os
import logging as log
import matplotlib.pyplot as plt
from careamics import CAREamist
from careamics.config import create_n2v_configuration

log.basicConfig(level=log.INFO)

def train_and_predict_n2v(noisy_image, X_train, X_val, batch_size, num_epochs, patch_size):
    '''Train the Noise2Void model.
    
    Args:
    X_train: np.ndarray, Training data 
    X_val: np.ndarray, Validation data
    patch_size: tuple, Size of the patches to be extracted from the images
    batch_size: int, Number of patches to be processed in each batch
    num_epochs: int, Number of epochs to train the model
    '''

    output_root = 'models/'

    # Create the Noise2Void configuration
    log.info("Creating Noise2Void configuration...")
    config = create_n2v_configuration(
        experiment_name="n2v_experiment",
        data_type='array',
        axes='SYX',
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs
    )

    log.info("Creating CAREamist model...")
    careamist = CAREamist(source=config, work_dir=os.path.join(output_root, "n2v_experiment"))

    # Train model
    log.info("Training CAREamist model...")
    careamist.train(train_source=X_train, val_source=X_val)
    log.info("Training completed successfully")
    
    # Denoise the image
    denoised_image = np.array(careamist.predict(noisy_image)).squeeze()
    
    return denoised_image