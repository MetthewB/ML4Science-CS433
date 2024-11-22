import numpy as np
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from scripts.helpers import *

def denoise_n2v(single_image):
    """ Noise2Void filter for denoising a single image. """

    # Normalize the image to [0, 1]
    single_image_normalized = normalize_image(single_image)
    
    # Create a data generator
    datagen = N2V_DataGenerator()
    
    # Add dimension to single image to simulate batch and channel
    single_image_expanded = np.expand_dims(single_image_normalized, axis=(0, -1))  # Shape: (1, 512, 512, 1)

    # Generate patches (Shape: (64, 64))
    patches = datagen.generate_patches_from_list([single_image_expanded], shape=(64, 64))

    # Dynamically split into training and validation sets (e.g., 80% training, 20% validation)
    split_idx = int(0.8 * len(patches))
    train_patches = patches[:split_idx]
    val_patches = patches[split_idx:]

    # Define model configuration
    config = N2VConfig(
        train_patches,
        unet_kern_size=3,
        train_steps_per_epoch=100, # see more data during each epoch
        train_epochs=5, # train model for longer period 
        train_loss='mse',
        batch_norm=True,
        train_batch_size=4, # change it, larger batch size can stabilize training
        n2v_perc_pix=0.198,  # Suggested value
        n2v_patch_shape=(64, 64), # can change the patch sahpe if images are large, can help model learn better features 
        n2v_manipulator='uniform_withCP',
        n2v_neighborhood_radius=5,
    )

    # Create the Noise2Void model
    model = N2V(config, "n2v_single_image", basedir="models")

    # Train the model
    model.train(X=train_patches, validation_X=val_patches)
    print('Training finished')

    # Apply the model to denoise the image
    denoised_image = model.predict(single_image_expanded.reshape(512, 512), axes='YX')  # Predicts denoised image
    print("Prediction finished")
    
    return denoised_image
