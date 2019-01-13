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
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
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

#----------------------------------------------------------------------------

# --------Define and train the model-----------------------------------------    

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


# --------Preprocess the image to look more like MNIST---------------------
import math
from scipy import ndimage
import cv2

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

# read the image
gray = cv2.imread('GUI/image_0.png', 0) # 0 is for grayscale read. # 1 for color image without any transparency. -1 for image without any changes
gray = 255 - gray

# better black and white version
# 128 is the threshhold value. Above it the value will be 255 and below will be 0. Controlles by THRESH_BINARY
# cv2.THRESH_OTSU = 
(thresh, gray) = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Remove empty rows and columns.
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)
    
rows,cols = gray.shape

# resize to 20 by 20.
if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))
    
# Add a padding on all sides to turn into 28 * 28
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted

cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()

#--------Load the saved model and check how the input is working.---------
from PIL import Image
from convert_mnist_format import ConvertMNISTFormat

image = Image.open('GUI/image_0.png')
image = image.resize((28, 28))
image = np.asarray(image)
preprocess = ConvertMNISTFormat(image)
image = preprocess.process_image()

image = np.resize(image, (1, 28, 28, 1))
image = np.divide(image, 255)

from keras.models import load_model

model = load_model('cnn-digits.h5')

predict = np.argmax(model.predict(image))

