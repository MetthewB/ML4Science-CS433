import torch
import logging as log
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Noise2Noise.models.drunet import DRUNet
import numpy as np
from tqdm import tqdm
from skimage import io
from scripts.helpers import *
from scripts.metrics import scale_invariant_psnr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

log.basicConfig(level=log.INFO)

# Create a custom dataset class
class NoisyImageDataset(Dataset):
    def __init__(self, noisy_images, target_images, data_augmentation=False):
        self.noisy_images = noisy_images
        self.target_images = target_images
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image = self.noisy_images[idx]
        target_image = self.target_images[idx]
        if self.data_augmentation:
            # Apply data augmentation here if needed
            pass
        return torch.tensor(noisy_image, dtype=torch.float32), torch.tensor(target_image, dtype=torch.float32)
    
# Define training options
training_options = {
    'color': False,
    'batch_size': 16,
    'patch_size': 128,
    'data_augmentation': True,
    'gamma_scheduler': 0.5,
    'lr': 0.0001,
    'lr_decay_step': 10000,
    'num_workers': 4,
    'testing_step': 10000,
    'total_steps': 100000
}

# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")

# Initialize the DRUNet model
log.info("Initializing the DRUNet model...")
nb_channels = 64 # Number of channels in the first convolutional layer of the network (how many features network can learn at each layer)
depth = 5 # number of residual blocks in each stage of the network (how many layers the network has to learn complex features)
color = False  # Set to True for color images, False for grayscale images
model = DRUNet(nb_channels, depth, color)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load your noisy and target images (example with random data)
log.info("Loading the noisy and target images...")
image = np.load('data/channel0/Image001/wf_channel0.npy')

# Sample a slice from the image (1, 512, 512)
log.info("Sampling a slice from the image...")
noisy_image, noisy_image_index = sample_image(image) 

log.info("Normalizing the noisy image...")
normalized_noisy_image = normalize_image(noisy_image)

# Prepare the data for training the model
# Take all the remaining slices to train the model (399)
log.info("Preparing data for training the model...")
filtered_image_indices = np.setdiff1d(np.arange(image.shape[0]), noisy_image_index)
filtered_image = image[filtered_image_indices]
normalized_filtered_image = [normalize_image(slice) for slice in filtered_image]

noisy_images = np.array(normalized_noisy_image)[0:3, :, :] # 199 gray scale images of size 512x512
target_images = np.array(normalized_filtered_image)[3:7, :, :] # 199 gray scale images of size 512x512

# Create the dataset and dataloader
log.info("Creating the dataset and dataloader...")
dataset = NoisyImageDataset(noisy_images, target_images, data_augmentation=training_options['data_augmentation'])
dataloader = DataLoader(dataset, batch_size=training_options['batch_size'], shuffle=True, num_workers=training_options['num_workers'])

# Training loop
num_epochs = training_options['total_steps'] // len(dataloader)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_options['lr_decay_step'], gamma=training_options['gamma_scheduler'])

log.info("Training the model...")
for epoch in range(num_epochs):
    for step, (noisy_image, target_image) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(noisy_image)
        loss = criterion(output, target_image)
        loss.backward()
        optimizer.step()

        if step % training_options['testing_step'] == 0:
            log.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}")

    scheduler.step()

# Denoise the image
log.info("Denoising the image...")
noisy_image_tensor = torch.tensor(normalized_noisy_image, dtype=torch.float32)
with torch.no_grad():
    denoised_image = model(noisy_image_tensor).numpy().squeeze()
    
# Generate ground truth and sample image
log.info("Generating ground truth and sample image...")
ground_truth_img = ground_truth(image)

log.info(f'Ground truth image shape: {ground_truth_img.shape}') 
log.info(f'Denoised image shape: {denoised_image.shape}')  

log.info("Calculating the metrics...")
psnr_denoised = peak_signal_noise_ratio(ground_truth_img, denoised_image, data_range=data_range(ground_truth_img))
si_psnr_denoised = scale_invariant_psnr(ground_truth_img, denoised_image)
ssim_denoised = structural_similarity(ground_truth_img, denoised_image, data_range=data_range(ground_truth_img))
log.info(f"PSNR: {psnr_denoised:.2f}, SI-PSNR: {si_psnr_denoised:.2f}, SSIM: {ssim_denoised:.2f}")