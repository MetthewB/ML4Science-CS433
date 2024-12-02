import numpy as np
import os
import logging as log
import matplotlib.pyplot as plt
from careamics import CAREamist
from careamics.config import create_n2v_configuration

log.basicConfig(level=log.INFO)

def predict_n2v(noisy_image, model_path):
    '''Predict the denoised image using the Noise2Void model.
    
    Args:
    noisy_image: np.ndarray, Image to be denoised
    model_path: str, Path to the trained Noise2Void model
    '''
    
    # Load the trained model
    log.info("Loading the trained model...")
    model = CAREamist.load(model_path)
    
    # Predict the denoised image
    log.info("Predicting the denoised image...")
    denoised_image = np.array(model.predict(noisy_image)).squeeze()
    
    return denoised_image