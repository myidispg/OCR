#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:50:22 2018

@author: myidispg
"""
#----------------------------------------------------------------------------------------------------

from scipy import io as spio
import numpy as np
import pandas as pd
import gc
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


emnist = spio.loadmat("../Datasets/matlab/emnist-byclass.mat")
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.int64)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1].astype(np.int64)

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.int64)

# load test labels
y_test_2d = emnist["dataset"][0][0][1][0][0][1]

# Convert y_test from 2-d to 1-d

y_test = []

for label in y_test_2d:
    y_test.append(label[0])

y_detect_test = []

for i in range(x_test.shape[0]):
    y_detect_test.append([1])


del emnist,i
gc.collect()

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 2

# Convert x_test to a 4D matrix of dimensions- 116323x28x28x1 for convolution layer.
# Divide by 255 for feature scaling
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

# Encoding y_train to categorical
y_detect_test = np.asarray(y_detect_test, dtype='float32')
y_detect_test.reshape(x_test.shape[0],1)
y_detect_test = keras.utils.to_categorical(y_detect_test, num_classes)

##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))

model.summary()
#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 2
#model training
model.fit(x_test, y_detect_test,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1)
#          validation_data=(X_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

# ------------------Convert the train and test set to csv with labels appended----------
df_test = pd.DataFrame(x_test)
del x_test
df_y_test = pd.DataFrame(y_test_2d)
del y_test_2d

df_test = pd.concat([df_test, df_y_test], axis=1)
del df_y_test

df_test.to_csv('Test_Images_with labels', index=False)
del df_test
gc.collect()


df_train = pd.DataFrame(x_train)
del x_train
df_y_train = pd.DataFrame(y_train)
del y_train

gc.collect()

df_train = pd.concat([df_train, df_y_train], axis=1)
del df_y_train

df_train.to_csv('Train_Images_with_labels', index=False)
del df_train
gc.collect()

#------------------------------------------------------------------------------------

# Testing CNN with a dummy image
# Testing an image.
from PIL import Image, ImageOps
# Open the image and convert to grayscale
pil_im = Image.open('flower_test_image.jpeg').convert('L')
# Invert image colors
pil_im = ImageOps.invert(pil_im)
pil_im.save('flower_test_image_grayscale.jpeg')
pil_im = pil_im.resize((28,28))
pil_im.save('flower_test_image_grayscale28px.jpeg')

# get pixels values
pix_val = list(pil_im.getdata())

# pixel vaues in a list
pix_val_flat = []
pix_val_flat.append([value for value in pix_val])

pix_val = np.asarray(pix_val)
pix_val = pix_val.reshape(1,28,28,1)

test_detect = model.predict(pix_val)