# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:48:33 2018

@author: Prashant Goyal
"""
#------Code on python-mnist website----------------------
from mnist import MNIST

# Path to the directory containing the datasets
mndata = MNIST('../Datasets/gzip')

# Select the datasets. Can be of type- 
# balanced
# byclass - chose this because it has 26 classes for lower case, 26 for upper case and 10 for digits.
# bymerge
# digits
# letters
# mnist
mndata.select_emnist('byclass')
# Set this true to extract data from the .gz files directly.
mndata.gz = True

images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

df_images_train = pd.DataFrame(images_train)
#-----------------------------------------------------------

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

# store labels for visualization
train_labels = y_train
test_labels = y_test

# For character detection, a list with all ones
y_detect_train = []

for i in range(x_train.shape[0]):
    y_detect_train.append([1])

y_detect_test = []

for i in range(x_test.shape[0]):
    y_detect_test.append([1])

# reshape using matlab order
x_train_reshape = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
x_test_reshape = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

y_train_reshape = y_train.reshape(y_test.shape[0], 1)
# labels should be onehot encoded
import keras
y_train = keras.utils.to_categorical(y_train, 62)
y_test = keras.utils.to_categorical(y_test, 62)


# --------------------all this part is useless, just for finding the labels------------------
samplenum = 0
import matplotlib.pyplot as plt


img = x_train_reshape[samplenum]

# visualize image
plt.imshow(img[0], cmap='gray')

y_demo = train_labels[samplenum]
y_demo1 = y_train[samplenum]

# Loop to find the index of a label
index = 0

for i in range(len(train_labels)):
    if train_labels[i][0] == 34:
        index = i
        break
    
del samplenum, img, y_demo, y_demo1, index, i
gc.collect()
#-----------------------------------------------------------------------------------------
        
# Train IsolationForest to detect whether an image contains a character or not
from sklearn.ensemble import IsolationForest

# Split train data into 2 parts.

x_train_4l = []

for i in range(150000):
    x_train_4l.append(x_train[i])
    
y_detect_train = []

for i in range(150000):
    y_detect_train.append([1])

del x_train,i, emnist
gc.collect()

text_detector = IsolationForest(n_estimators=5, verbose=100)
text_detector.fit(x_train_4l, y_detect_train)

y_train_predict = text_detector.predict(x_train)

# Based on a quick calculation, achieved an accuracy of 88% in text detection. Will try again with train set. More accuracy expected there with grid search.

# Testing an image.
from PIL import Image, ImageOps
# Open the image and convert to grayscale
pil_im = Image.open('r_test_image.jpeg').convert('L')
# Invert image colors
pil_im = ImageOps.invert(pil_im)
pil_im.save('r_grayscale.jpeg')
pil_im = pil_im.resize((28,28))
pil_im.save('r_grayscale28px.jpeg')

# get pixels values
pix_val = list(pil_im.getdata())

# pixel vaues in a list
pix_val_flat = []
pix_val_flat.append([value for value in pix_val])

test_detect = text_detector.predict(pix_val_flat)

# Write a function to convert background to white. Check if max values are more than 100. If max are more than 100, replace those by 0.
# Some notes- 0 for black, 255 for white. If there are more than 60.71%(found by calculating white percentage in train set), replace all more than 0 and less than 100 by 0. 

# -----------------------Try to find the percentage of white and black in an image------------
white_count = 0
black_count = 0
percentages_white = [] 

for i in range(len(x_train_4l)):
    for pix in x_train_4l[0]:
        if pix == 0:
            white_count += 1
        else:
            black_count += 1
    percentages_white.append(white_count/784*100)
    white_count = 0
    black_count = 0
    
del white_count, black_count, percentages_white, i, pix
gc.collect()
#----Found %of0-60.71428571428571------------------------------------------------------------    
#-----------Now, we can replace all less that 100 by 0 in test image------------------------
for i in range(len(pix_val_flat[0])):
    pix_val_flat[0][i] = 0 if (pix_val_flat[0][i] < 100) else pix_val_flat[0][i]
    
del i
gc.collect()

white_bg_image = Image.new('L', (28,28))
pix = white_bg_image.load()

# Reshape to a matrix of 28x28
count = 0
for x in range(28):
    for y in range(28):
        pix[x,y] = test[count]
        count += 1
        
white_bg_image.save("test.png", "PNG")