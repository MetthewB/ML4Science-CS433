import numpy as np
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

def denoise_n2v(single_image):
    """ Noise2Void filter."""
    
    single_image_normalized = (single_image - single_image.min()) / (single_image.max() - single_image.min())
     
    # Create a data generator
    datagen = N2V_DataGenerator()

    # Add dimension to single image to simulate batch and channel
    single_image_expanded = np.expand_dims(single_image_normalized, axis=(0, -1))  # Shape: (1, 512, 512, 1)

    # Generate patches for training
    patches = datagen.generate_patches_from_list([single_image_expanded], shape=(64, 64))  # Patch size: (64, 64)

    # Define model configuration
    config = N2VConfig(
        single_image_expanded,
        unet_kern_size=3,
        train_steps_per_epoch=100,
        train_epochs=5,
        train_loss='mse',
        batch_norm=True,
        train_batch_size=4,
        n2v_perc_pix=1.6,
        n2v_patch_shape=(64, 64),
        n2v_manipulator='uniform_withCP',
        n2v_neighborhood_radius=5,
    )

    # Create the Noise2Void model
    model = N2V(config, "n2v_single_image", basedir="models")

    # Train the model
    model.train(X=patches[:410], validation_X=patches[410:])  # Using 20% of the data for validation

    print('training finished')
    
    # Apply the model to denoise the image
    denoised_image = model.predict(single_image_expanded.reshape(512, 512), axes='YX')  # Predicts denoised image

    print("PREDICTION FINISHED")
    
    return denoised_image