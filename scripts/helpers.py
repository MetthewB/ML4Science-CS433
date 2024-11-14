import numpy as np

def load_image(path_to_data, image_path):
    """Load a .npy image file from the specified path."""
    return np.load(path_to_data + image_path)

def ground_truth(image):
    """Calculate ground truth as the mean projection of image along the first axis."""
    return image.mean(axis=0)

def sample_image(image):
    """Randomly sample an image slice."""
    sampled_image_index = np.random.randint(0, image.shape[0])
    return image[sampled_image_index]

def data_range(ground_truth_image):
    """Calculate data range for PSNR and SSIM calculations."""
    return ground_truth_image.max() - ground_truth_image.min()