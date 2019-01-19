a#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:56:08 2019

@author: myidispg
"""

import numpy as np
import keras
import os
import gc
import cv2
import pandas as pd

from random import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

data_dir_base = '../MNIST_Dataset/images_by_classes'

# Get a list of all the files in the test and train datasets.
test_dir = 'test'
train_dir = 'train'

train_dataset = os.listdir(os.path.join(data_dir_base, train_dir))
test_dataset = os.listdir(os.path.join(data_dir_base, test_dir))

# Shuffle the lists.
shuffle(train_dataset)
shuffle(test_dataset)

# Get some basic dimensions of the image
(h, w) = (28, 28)
center = (w/2, h/2)

# To ratate the image, we need imutils
import imutils
scale = 1.0
rotation_mat = cv2.getRotationMatrix2D(center, 180, scale)


# Create a list of labels with corresponding image names.
label_dict = {}

labels = []
for i in range(62):
    labels.append(i)
    
for label in labels:
    label_dict[label] = []

for data in test_dataset:
    label = int(data.split('_')[1].split('.')[0])
    label_dict[label].append(data)

# Visualize 5 images of each category
for i in range(15):
    img = cv2.imread(os.path.join(data_dir_base, test_dir, label_dict[34][i]), 0)
#    img = cv2.warpAffine(img, rotation_mat, (h, w))
    img = imutils.rotate(img, 270)
    img = cv2.flip(img, 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define a batch generator
def batch_generator(image_paths, batch_size, isTraining):
    while True:
        batch_imgs = []
        batch_labels = []
        
        type_dir = 'train' if isTraining else 'test'
        
        for i in range(len(image_paths)):
#            print(i)
#            print(os.path.join(data_dir_base, type_dir, image_paths[i]))
            img = cv2.imread(os.path.join(data_dir_base, type_dir, image_paths[i]), 0)
            img = imutils.rotate(img, 270)
    img = cv2.flip(img, 1)
            img  = np.divide(img, 255)
            img = img.reshape(28, 28, 1)
            batch_imgs.append(img)
            label = image_paths[i].split('_')[1].split('.')[0]
            batch_labels.append(label)
            category_labels = keras.utils.to_categorical(batch_labels, 62)
            print(category_labels)
            if len(batch_imgs) == batch_size:
                yield (np.asarray(batch_imgs), np.asarray(category_labels))
                batch_imgs = []
                batch_labels = []
        if batch_imgs:
            yield batch_imgs
        
gen = next(batch_generator(test_dataset, 10, False))


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
    model.add(Dense(62, activation='softmax'))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model
    
model = create_model()
print(model.summary())

history = model.fit_generator(batch_generator(train_dataset, 512, True),
                              epochs=5,
                              steps_per_epoch = 1363,
                              validation_data=batch_generator(test_dataset, 512, False),
                              validation_steps = 227,
                              verbose=1,
                              shuffle=1)

model.save('cnn-by-class.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

# For checking the categories
from PIL import Image
from convert_mnist_format import ConvertMNISTFormat
import numpy as np
import cv2

image = Image.open('image_7.png')
image = image.resize((28, 28), Image.ANTIALIAS)
image = np.asarray(image)
preprocess = ConvertMNISTFormat(image)
image = preprocess.process_image()
image = np.divide(image, 255)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

from keras.models import load_model

model = load_model('cnn-by-class.h5')
image = np.resize(image, (1, 28, 28, 1))
predict = np.argmax(model.predict(image))

for data in test_dataset:
    label = data.split('_')[1].split('.')[0]
    if label == str(36):
        print(data)

import cv2
image = cv2.imread(os.path.join(data_dir_base, test_dir, '110274_36.png'), 0)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

labels_completed = []

for data in train_dataset:
    label = data.split('_')[1].split('.')[0]
    if label not in labels_completed:
        print(label)
        labels_completed.append(label)
        image = cv2.imread(os.path.join(data_dir_base, test_dir, data), 0)
        cv2.imshow('image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
