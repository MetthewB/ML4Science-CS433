import os
import numpy as np

def get_paths():
    """
    Returns the data and output paths relative to the current working directory.
    
    :return: A tuple (data_path, output_path)
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data' directory
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'data')) + '/'
    output_path = os.path.abspath(os.path.join(script_dir, '..', 'output')) + '/'
    
    return data_path, output_path

def normalize_image(image): 
    """Normalizes an image if pixels range are not between 0 and 1."""
    if (not (0 <= image.min() and image.max() <=1 )): 
        return (image - image.min()) / (image.max() - image.min()) 
    else : 
        return image 

def load_image(path_to_data, image_path):
    """Load a .npy image file from the specified path."""
    return np.load(path_to_data + image_path)

def ground_truth(image):
    """Calculate ground truth as the mean projection of image along the first axis."""
    return normalize_image(image.mean(axis=0))

def sample_image(image):
    """Randomly sample an image slice."""
    sampled_image_index = np.random.randint(0, image.shape[0])
    return image[sampled_image_index].reshape(1, image.shape[1], image.shape[2]), sampled_image_index

def data_range(ground_truth_image):
    """Calculate data range for PSNR and SSIM calculations."""
    return ground_truth_image.max() - ground_truth_image.min()
