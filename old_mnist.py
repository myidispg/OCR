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


#--------Load the saved model and check how the input is working.---------
from PIL import Image
from convert_mnist_format import ConvertMNISTFormat
import numpy as np
import cv2

## Solve over-segmentation(Prevelant in open loop chars like W, U, M etc.)
#threshold = 9
#segments = []
#
#for x in range(1, len(psc)):
#    if (psc[x] - psc[x-1]) < threshold:
#        segments.append(psc[x])
        
# Draw a line over all psc
from keras.models import load_model

labels = [0,1,2,3,4,5,6,7,8,9,
          'A','B','C','D','E','F','G','H','I', 'J', 'K','L','M','N','O','P',
          'Q','R','S','T','U','V','W','X','Y','Z', 'a','b','c','d','e','f',
          'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
          'w','x','y','z']

model = load_model('cnn-by-class-2.h5')

from character_segment import SegmentCharacters

def brighten_image(image):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] > 0:
                image[x][y] = (image[x][y] + 0.2) if (image[x][y] + 0.2) < 1 else image[x][y]
    return image
    

test_images = ['image_1.png', 'image_2.png', 'image_3.png', 'image_6.png', 'image_7.png', 
               'image_8.png', 'image_9.png', 'image_10.png', 'image_11.png', 'image_12.png', 
               'image_13.png', 'image_15.png']


for image in test_images:
    image = Image.open(image)
    image = np.asarray(image)
    image = np.divide(image, 255)
    image = (1-image)
#    cv2.imshow('image', image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    char_segment = SegmentCharacters(image)
    coords = char_segment.find_char_images()
    
    sep_images = []
    current_index = 0
    kernel = np.ones((2,2),np.uint8)
    for coord in coords:
        if coord == 0:
            pass
        else:
            copy = image[0: 500, current_index: coord+1]
            print(copy.shape)
            mnist_convert = ConvertMNISTFormat(copy)
            copy = mnist_convert.process_image()
            copy = cv2.dilate(copy,kernel,iterations = 1)
            copy = brighten_image(copy)
            cv2.imshow('image', copy)
            cv2.waitKey()
            cv2.destroyAllWindows()
            sep_images.append(copy)
            current_index = coord
    new_image = np.resize(sep_images[0], (1, 28, 28, 1))
    predict = np.argmax(model.predict(new_image))
    print('The image has - {}'.format(labels[predict]))


