#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:48:23 2019

@author: myidispg
"""

from scipy import io as spio
import numpy as np
import pandas as pd
import gc

emnist = spio.loadmat('../MNIST_Dataset/matlab/emnist-digits.mat')
# load training dataset
x_train = emnist['dataset'][0][0][0][0][0][0]
x_train = x_train.astype(np.int64)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1].astype(np.int64)

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

del emnist
gc.collect()

# store labels for visualization
train_labels = y_train
test_labels = y_test

# normalize
x_train = np.divide(x_train, 255)
x_test = np.divide(x_test, 255)

# Reshape the train images for CNN
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)

import keras

# Labels should be one-hot encoded
y_train = keras.utils.to_categorical(y_train, 62)


# -------------------Verify that the data is import correctly----------------
import matplotlib.pyplot as plt
img = X_train[995]
img = 255- img
img = np.invert(img)

# visualize the image
plt.imshow(img[0], cmap='gray')
y_train[995]

# -----Define the model-------------
mean_px = x_train.mean().astype(np.float32)

gc.collect()

std_px = x_train.std().astype(np.float32)


def norm_input(x): return (x-mean_px)/std_px

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

# Batchnorm + dropout + data augmentation
def create_model():
    model = Sequential([
            Lambda(norm_input, input_shape = (1, 28, 28), output_shape = (1, 28, 28)),
            Conv2D(32, (3, 3)),
            LeakyReLU(),
            BatchNormalization(axis=1),
            Conv2D(32, (3,3)),
            LeakyReLU(),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Conv2D(64, (3,3)),
            LeakyReLU(),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Conv2D(64, (3,3)),
            LeakyReLU(),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(),
            Dense(512),
            LeakyReLU(),
            BatchNormalization(),
            Dropout(0.2),
            Dense(62, activation='softmax')
            ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data Augmentation
    
batch_size = 512

from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                         height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')
batches = gen.flow(x_train, y_train, batch_size=batch_size)
steps_per_epoch = int(np.ceil(batches.n/batch_size))


# ---- Ensembling-----

# Create TEN models from scratch
models = []
weights_epoch = 0

for i in range(10):
    m = create_model()
    models.append(m)

