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
text_dataset = pd.read_csv('../Datasets/Test_Images_with_labels_invert.csv')
img_dataset = pd.read_csv('../Datasets/Non-text-images.csv')

# Drop the label column because it is not useful
text_dataset = text_dataset.drop(['784'], axis=1)

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
del labels_list, df_train, concat_array
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

# save the model to disk
model.save('text_detection_model.h5')

del batch_size, img_cols, img_rows, input_shape, num_classes, num_epoch, model
gc.collect()

# load the model from disk
from keras.models import load_model

detection_model = load_model('text_detection_model.h5')
detection_model.summary()

#---------------------------------------------------------------------------------

#---------- Just a script to test text detection in images----------------

img = Image.open('g-text-test.jpeg').convert('L')
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

test_detect = detection_model.predict(pix_val)


# Points- Make sure the images have background as 0. 
# Images with white background and colored text are easily classified. They are inverted so text is made white and background is black.
# Therefore, make sure that the background has pixel values of 0 and text has more than 160 or 170.

def preprocess_image(pix_val):
    count_white = 0
    count_black = 0
    for pix in pix_val:
        if pix >= 127:
            count_white += 1
        else:
            count_black += 1
    print('count_white- ' + str(count_white))
    print('count_black- ' + str(count_black))
    
    # if count_black<count_white, invert the image and then return the numpy array in proper shape.
    if count_black < count_white:
        pix_val = list(invert_img_pix_list(pix_val))
        
    # Now that the image inversion is done, we will change any pixel values less that 160 to 0.
    for i in range(len(pix_val)):
        pix_val[i] = 0 if pix_val[i] < 170 else pix_val[i]
        
    # Convert to numpy array and then reshape to 1x28x28x1 as required by Conv Net.
    pix_val = np.asarray(pix_val)
    pix_val = pix_val.reshape(1, 28, 28, 1)
    
    return pix_val

def invert_img_pix_list(pix_list):
    print('yes')
    for i in range(len(pix_list)):
        pix_list[i] = 255-pix_list[i]
    return pix_list

# Final pipeline for opening, processing the image and then text_detection.
from PIL import Image

# Open the image and convert to grayscale.
img = Image.open('g-text-test.jpeg').convert('L')
img = img.resize((28,28))
# Get list of pixel values
pix_val = list(img.getdata())

pix_val = preprocess_image(pix_val)
test_detect = detection_model.predict(pix_val)

# The current approach is not working, let us try with background of mnist images inverted.
    
