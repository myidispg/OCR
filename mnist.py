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

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator


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
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A")

# labels should be onehot encoded
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# -------------------Verify that the data is import correctly----------------
import matplotlib.pyplot as plt

samplenum = 5437

img = x_train[samplenum]
img = 255- img

# visualize the image
plt.imshow(img[0], cmap='gray')
train_labels[samplenum][0]

# -----Define the model-------------
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)


def norm_input(x): return (x-mean_px)/std_px

# Batchnorm + dropout + data augmentation
def create_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28), output_shape=(1,28,28)),
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
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data Augmentation
    
batch_size = 512

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                         height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')
batches = gen.flow(x_train, y_train, batch_size=batch_size)
test_batches = gen.flow(x_test, y_test, batch_size=batch_size)
steps_per_epoch = int(np.ceil(batches.n/batch_size))
validation_steps = int(np.ceil(test_batches.n/batch_size))

# load ONE image from training set to display on screen
img = x_train[1]
img = 255-img

# trick our generator into believing img has enough dimensions
# and get some augmented images for our single test image
img = np.expand_dims(img, axis=0)
aug_iter = gen.flow(img)

# visualize original image
plt.imshow(img[0], cmap='gray')

# ---- Ensembling-----

# Create TEN models from scratch
models = []
weights_epoch = 0

for i in range(10):
    m = create_model()
    models.append(m)
    

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model
    
model = create_model()
print(model.summary())
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=512, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

# Save the model
model.save('cnn-digits.h5')

#--------Load the saved model and check how the input is working.---------
from PIL import Image
image = Image.open('GUI/image_0.png')
image = image.resize((28, 28))
image_arr = np.asarray(image)
image_arr = np.subtract(255, image_arr)
image = np.resize(image, (1, 28, 28, 1))

from keras.models import load_model

model = load_model('../cnn-digits.h5')

predict = model.predict(image)

