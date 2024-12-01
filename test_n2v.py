import numpy as np
import os
import logging as log
from skimage.util import random_noise
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from scripts.helpers import *
import matplotlib.pyplot as plt

from careamics import CAREamist
from careamics.config import create_n2v_configuration



log.basicConfig(level=log.INFO)

def sample_slices(image, n_slices_per_image):
    '''Sample random slices from the images'''
    slices = []
    indices = []
    
    for _ in range(n_slices_per_image):
        idx = np.random.randint(0, image.shape[0])  # Random slice along the first axis
        slices.append(image[idx])
        indices.append(idx)
        
    filtered_indices = np.setdiff1d(np.arange(image.shape[0]), indices)
    
    filtered_image = image[filtered_indices]  # Filtered image
    log.info(f"Filtered image shape: {filtered_image.shape}")
    
    return np.array(slices), filtered_image

log.info("Loading data...")
image1 = np.load('data/channel0/Image001/wf_channel0.npy')
log.info("Data loaded successfully")

log.info("Sampling slices...")
slices, filtered_image1 = sample_slices(image1, 10)

log.info("Normalizing slices...")
normalize_slices = [normalize_image(slice) for slice in slices]

log.info("Splitting data into training and validation sets...")
train_slices, val_slices = train_test_split(normalize_slices, test_size=0.2, random_state=42)

# Convert lists to NDArray
train_slices = np.array(train_slices)
val_slices = np.array(val_slices)

log.info(f"Train slices: {len(train_slices)}")
log.info(f"Validation slices: {len(val_slices)}")


output_root='models/'


# Create the Noise2Void configuration
log.info("Creating Noise2Void configuration...")
config = create_n2v_configuration(
    experiment_name="n2v_experiment",
    data_type='array',
    axes='SYX',
    patch_size=(64, 64),
    batch_size=4,
    num_epochs=2
)

log.info("Creating CAREamist model...")
careamist = CAREamist(source=config, work_dir=os.path.join(output_root, "n2v_experiment"))

# train model
log.info("Training CAREamist model...")
careamist.train(train_source=train_slices, val_source=val_slices)

log.info("Normalizing noisy image...")
noisy_image = normalize_image(sample_image(filtered_image1))

log.info("Denoising noisy image...")

# Denoise the image
denoised_image = np.array(careamist.predict(noisy_image)).squeeze()

log.info("Denoising completed successfully")

# Save the denoised image
output_path = "output/denoised_image.npy"
log.info(f"Saving denoised image to {output_path}")
np.save(output_path, denoised_image)


# Display the noisy and denoised images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image.squeeze(), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap="gray")

plt.show()


