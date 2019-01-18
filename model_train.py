#!/usr/bin/env python3
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

# Define a batch generator

def batch_generator(image_paths, batch_size, isTraining):
    while True:
        batch_imgs = []
        batch_labels = []
        
        type_dir = 'train' if isTraining else 'test'
        
        for i in range(len(image_paths)):
            print(i)
            print(os.path.join(data_dir_base, type_dir, image_paths[i]))
            img = cv2.imread(os.path.join(data_dir_base, type_dir, image_paths[i]), 0)
            img  = np.divide(img, 255)
            img = img.reshape(28, 28, 1)
            batch_imgs.append(img)
            label = image_paths[i].split('_')[1].split('.')[0]
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                yield (np.asarray(batch_imgs), np.asarray(batch_labels))
                batch_imgs = []
        if batch_imgs:
            yield batch_imgs
        
index = next(batch_generator(train_dataset, 10, True))

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