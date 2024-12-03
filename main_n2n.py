import numpy as np
import argparse
import logging as log
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from noise2noise.model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from sklearn.model_selection import train_test_split
from scripts.helpers import get_paths, load_image, ground_truth, sample_image, data_range, normalize_image

log.basicConfig(level=log.INFO)

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125
    

def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=512,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    args = parser.parse_args()

    return args

# create unique random pairs of images from normalized_image using 398 images you get 149 pairs
# TODO : can be changed 
def create_random_pairs(image_slices):
    pairs = []
    num_slices = len(image_slices)
    
    for i in range(num_slices - 1):
        pairs.append((image_slices[i], image_slices[i + 1]))
    
    return pairs

image_dir = "data/Image001/wf_channel0.npy"
image_size = 512
batch_size = 16
nb_epochs = 60
lr = 0.01
steps = 1000
loss_type = "mse"
output_path = "output/noise2noise"
model = "srresnet"

log.info("Loading image from {}".format(image_dir))
image = np.load(image_dir)

noisy_image, noisy_index = sample_image(image)
normalize_noisy_image = normalize_image(noisy_image)

filtered_image_indices = np.setdiff1d(np.arange(image.shape[0]), noisy_index)
filtered_image = image[filtered_image_indices]
normalized_image = [normalize_image(slice) for slice in filtered_image]


log.info("Creating random pairs of images")
train_data = create_random_pairs(normalized_image)

# Divide the train data into x and y
x, y = zip(*train_data)

# Convert to numpy arrays if needed
x = np.array(x)
y = np.array(y)

# Define the model
log.info("Initializing model")
#args = get_args() # get the arguments
model = get_model(model_name="srresnet") # initialize the model

opt = Adam(learning_rate=lr)
callbacks = []

if loss_type == "l0":
    l0 = L0Loss()
    callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
    loss_type = l0()

model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])

#output_path.mkdir(parents=True, exist_ok=True)
callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.keras",
                                    monitor="val_PSNR",
                                    verbose=1,
                                    mode="max",
                                    save_best_only=True))

log.info("Training model")
hist = model.fit(x,
                 y,
                 steps_per_epoch=steps,
                 epochs=nb_epochs,
                 verbose=1,
                 callbacks=callbacks, 
                 validation_split=0.2)

log.info("Saving model")
np.savez(str(output_path.joinpath("history.npz")), history=hist.history)

