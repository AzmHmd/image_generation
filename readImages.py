import os
import time
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image


# Hyperparameters
IMAGE_SIZE = 128
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 300
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5


# Paths
INPUT_DATA_DIR = "/home/azh2/Downloads/simpsons-faces/cropped/" # Path to the folder with input images. For more info check simspons_dataset.txt


# Training
input_images = np.asarray([np.asarray(Image.open(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in glob(INPUT_DATA_DIR + '*.png')])
print ("Input: " + str(input_images.shape))


np.save('input_images.npy', input_images)    # .npy extension is added if not given
d = np.load('input_images.npy')
print ("saved data: " + str(d.shape))
