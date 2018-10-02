#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:00:52 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
import keras
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Importing the datasets
text_dataset = pd.read_csv('../Datasets/Test_Images_with_labels.csv')
img_dataset = pd.read_csv('../Datasets/Non-text-images.csv')

# Drop the label column because it is not useful
text_dataset = text_dataset.drop(['0.1'], axis=1)

# Create a list of labels with 1 for text and 0 for non-text
labels_list = []

for i in range(text_dataset.shape[0]):
    labels_list.append(1)
for i in range(img_dataset.shape[0]):
    labels_list.append(0)
# Convert the list to numpy array. Will be concatenated later.
labels_list = np.asarray(labels_list)

# Concatenate the dataframes and the concatenate the labels list.
df_train = pd.concat([text_dataset, img_dataset])

# Delete unnecessary data
del i, img_dataset, text_dataset
gc.collect()

# Get numpy arrays.
X_train = df_train.iloc[:,:].values
y_train = labels_list

# Concatenating the data and labels to shuffle
y_train = y_train.reshape(y_train.shape[0], 1)
concat_array = np.concatenate([X_train, y_train], axis=1)

X_train = concat_array[:, :-1]
y_train = concat_array[:, 784]

# Delete unnecessary data
del labels_list, df_train
gc.collect()

# Some dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 2

# Convert X_train to a 4D matrix of dimensions- (no.of rows)x28x28x1 for convolution layer.
# Divide by 255 for feature scaling
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')/255

# Convert y to categorical
y_train = y_train.reshape(y_train.shape[0], 1)
y_train = keras.utils.to_categorical(y_train, num_classes)

# Building CNN
model = Sequential()
# Convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# ^^ 32 convolution filters used each of size 3x3

model.add(Conv2D(64, (3, 3), activation='relu'))
# ^^ 64 convolution filters used each of size 3x3
# Choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
# Flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# Fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
# One more dropout for convergence' sake :) 
model.add(Dropout(0.5))
# Output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))

model.summary()
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# Binary Cross Entrpy since we have only 2 classes
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 2
#model training
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=100)

del concat_array, batch_size, img_cols, img_rows, input_shape, num_classes, num_epoch
gc.collect()

#---------------------------------------------------------------------------------
# Testing CNN with a dummy image
# Testing an image.
from PIL import Image, ImageOps
# Open the image and convert to grayscale
image_r = Image.open('r-text-test.jpeg').convert('L')
image_r = image_r.resize((28,28))

image_g = Image.open('g-text-test.jpeg').convert('L')
image_g = image_g.resize((28,28))


# get pixels values
pix_val_r = list(image_r.getdata())
pix_val_g = list(image_g.getdata())

# Invert image colors only if amount of white is more than 60.71 %
count_white = 0
for i in range(len(pix_val)):
    

image_r = ImageOps.invert(image_r)
image_g = ImageOps.invert(image_g)
# pil_im.save('r-text-test-grayscale.jpeg')

# pil_im.save('r-text-test-grayscale28px.jpeg')

image_r.save('r-text-test-bw-28.jpeg')
image_g.save('g-text-test-bw-28.jpeg')

# Replace all pixel values less than 100 with 0.
count = 0
for i in range(len(pix_val_r)):
    if pix_val_r[i] < 100:
        count += 1
        pix_val_r[i] = 0
        
count = 0
for i in range(len(pix_val_g)):
    if pix_val_g[i] < 100:
        count += 1
        pix_val_g[i] = 0

del i, count
gc.collect()

pix_val_g = np.asarray(pix_val_g)
pix_val_g = pix_val_g.reshape(1,28,28,1)

test_detect_g = model.predict(pix_val_g)

#---------- Just a script to test text detection in images----------------

img = Image.open('n-text-test.jpeg').convert('L')
img = img.resize((28,28))
img = ImageOps.invert(img)
pix_val = list(img.getdata())
count = 0
for i in range(len(pix_val)):
    if pix_val[i] < 170:
        count += 1
        pix_val[i] = 0

del i, count
gc.collect()

pix_val = np.asarray(pix_val)
pix_val = pix_val.reshape(1, 28,28,1)

test_detect = model.predict(pix_val)


# Points- Make sure the images have background as 0. 
# Images with white background and colored text are easilly classified. They are inverted so text is made white and background is black.
# Therefore, make sure that the background has pixel values of 0 and text has more than 160 or 170.