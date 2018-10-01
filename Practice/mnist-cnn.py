#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:50:22 2018

@author: myidispg
"""

from scipy import io as spio
import numpy as np
import pandas as pd
import gc

emnist = spio.loadmat("../Datasets/matlab/emnist-byclass.mat")
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.int64)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1].astype(np.int64)

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.int64)

test = x_test[0]

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

y_detect_test = []

for i in range(x_test.shape[0]):
    y_detect_test.append([1])


del emnist
gc.collect()

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 1

# Convert x_test to a 4D matrix of dimensions- 116323x28x28x1 for convolution layer.
# Divide by 255 for feature scaling
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

# Encoding y_train to categorical
import keras
y_detect_test.reshape(x_test.shape[0],1)
y_detect_test = keras.utils.to_categorical(y_detect_test, num_classes)

# CNN Model
from keras.models import Sequential
from keras.layers import Convolution2D
# COnvolution 2D is for images. For videos, we have a third dimension that is time. 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 2, activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

classifier.fit(x_test, y_detect_test,batch_size=32, epochs=1)
