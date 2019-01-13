#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:12:55 2019

@author: myidispg
"""

import keras
import numpy as np

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

import gc

from scipy import io as spio

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
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="A")
x_test = x_test.reshape(x_test.shape[0], 28, 28,  1, order="A")

# labels should be onehot encoded
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# calculate mean and standard deviation
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

# function to normalize input data
def norm_input(x): return (x-mean_px)/std_px

def create_model():
#    model = Sequential()
#    model.add(Lambda(norm_input, input_shape=(1, 28,28), output_shape=(1,28,28)))
#    model.add(Conv2D(32, kernel_size=3, activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(32, kernel_size=3, activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.4))
#    
#    model.add(Conv2D(64,kernel_size=3,activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(64,kernel_size=3,activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.4))
#    
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.4))
#    model.add(Dense(10, activation='softmax'))
    
    model = Sequential([
        Lambda(norm_input, input_shape=(28,28,1), output_shape=(28,28,1)),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(32, (3,3)),
        LeakyReLU(),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Conv2D(64, (3,3)),
        LeakyReLU(),
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
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# Data augmentation

batch_size = 512

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                        height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')
#batches = gen.flow(x_train, y_train, batch_size=batch_size)
#test_batches = gen.flow(x_test, y_test, batch_size=batch_size)
#steps_per_epoch = int(np.ceil(batches.n/batch_size))
#validation_steps = int(np.ceil(test_batches.n/batch_size))
gen.fit(x_train)


import matplotlib.pyplot as plt

# load ONE image from training set to display on screen
img = x_train[1]

# visualize original image
plt.imshow(img[0], cmap='gray')

# trick our generator into believing img has enough dimensions
# and get some augmented images for our single test image
img = np.expand_dims(img, axis=0)
aug_iter = gen.flow(img)

# show augmented images
f = plt.figure(figsize=(12,6))
for i in range(8):
    sp = f.add_subplot(2, 26//3, i+1)
    sp.axis('Off')
    aug_img = next(aug_iter)[0].astype(np.float32)
    plt.imshow(aug_img[0], cmap='gray')
    
# Train the model
    
model = create_model()

history = model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=3, verbose=1, validation_data=(x_test, y_test),
                              shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')