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
img_dataset = pd.read_csv('../Datasets/Non-text-images-2.csv')

# Drop the label column which tells the class out of 62 classes. Not useful here.
text_dataset = text_dataset.drop(['784'], axis=1)
gc.collect()

# Create a list of labels with 1 for text and 0 for non-text
labels_list = []

# for i in range(text_dataset.shape[0]):
for i in range(text_dataset.shape[0]):
    labels_list.append(1)
for i in range(img_dataset.shape[0]):
    labels_list.append(0)
# Convert the list to numpy array. Will be concatenated later.
labels_list = np.asarray(labels_list)

# Concatenate the dataframes and then concatenate the labels list.
df_train = pd.concat([text_dataset, img_dataset])

# Delete unnecessary data
del i, img_dataset, text_dataset
gc.collect()

# Get numpy arrays.
X_train = df_train.iloc[:,:].values
y_train = labels_list

del df_train, labels_list
gc.collect()

# Concatenating the data and labels to shuffle
y_train = y_train.reshape(y_train.shape[0], 1)
concat_array = np.concatenate([X_train, y_train], axis=1)

np.random.shuffle(concat_array)

X_train = concat_array[:, :-1]
y_train = concat_array[:, 784]

# Delete unnecessary data
del concat_array
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
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 5

#model training
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=2)

# save the model to disk
model.save('text_detection_model-1.h5')

#-----------TEST MODELS----------------------------------------------------

model = Sequential()
model.add(Conv2D(5, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(25, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

batch_size = 128
num_epoch = 2
#model training
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=2)

model.save('text_detection_model-2.h5')

# Random Forest
from sklearn.ensemble import IsolationForest

X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')/255

text_detector = IsolationForest(n_estimators=10, verbose=100)
text_detector.fit(X_train, y_train)
#-----------------------------TEST MODELS END--------------------------------------
del batch_size, img_cols, img_rows, input_shape, num_classes, num_epoch,model, X_train, y_train
gc.collect()

# load the model from disk
from keras.models import load_model

detection_model = load_model('text_detection_model-1.h5')
detection_model.summary()

#---------------------------------------------------------------------------------

#---------- Just a script to test text detection in images----------------
#----------------Process Image v2.0--------------------------
def preprocess_image(pix_val):
        
#    for i in range(len(pix_val)):
#        if pix_val[i] <= 100:
#            pix_val[i] = 0
#        else:
#            pix_val[i] /= 255
#          
    pix_val = np.asarray(pix_val)
    pix_val = pix_val.reshape(1, 28, 28, 1)
    # Convert to numpy array and then reshape to 1x28x28x1 as required by Conv Net.
    
    return pix_val
#------------------------------------------------------------

# Final pipeline for opening, processing the image and then text_detection.
from PIL import Image, ImageOps

# Open the image and convert to grayscale.
img = Image.open('../Test Images/pro-g-text-test.jpeg').convert('L')
img = img.resize((28,28))
img = ImageOps.invert(img)
# Get list of pixel values
pix_val = list(img.getdata())

pix_val = preprocess_image(pix_val)/255
test_detect = detection_model.predict(pix_val)

def find_text_presence(file_path):
    """
    Function to find whether the given image consists of a text or not.
    This is the most basic level of text detection.
    input- Takes teh file path as input
    Returns 0 for no text, 1 for text.
    """
    from PIL import Image
    img = Image.open(file_path).convert('L')
    img = img.resize((28,28))
    # Get list of pixel values
    pix_val = list(img.getdata())
    pix_val = preprocess_image(pix_val)/255
    test_detect = detection_model.predict(pix_val)
    
    return np.argmax(test_detect)
    
print(find_text_presence('../Test Images/3-non-text-test.jpeg'))

def text_detect(image_path, detector_model):
    from PIL import Image, ImageOps
    img = Image.open(image_path)# .convert('L')
    # img = ImageOps.invert(img)
    pix_val = list(img.getdata())
    pix_val = np.asarray(pix_val)/255
    pix_val = pix_val.reshape(1,28,28,1)
    
    test_detect = detector_model.predict(pix_val)
    
    return True if np.argmax(test_detect) == 1 else False

text_detect('../Test Images/pro-5-non-text-test.jpeg', detection_model)    
    

