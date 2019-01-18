#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:33:42 2019

@author: myidispg
"""
from scipy import io as spio
import os
import numpy as np
import gc

emnist = spio.loadmat('../MNIST_Dataset/matlab/emnist-byclass.mat')
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

# -----Resize each image and save to a directory--------
# The test and train images will be stored in seperate directories.
# The corresponding label incorporated in filename.
# Filename format- "label_counter.png"
base_dir = '../MNIST_Dataset/images_by_classes/'

import cv2

test_dir = 'test/'
# Loop over the test dataset and save each image.
for x in range(len(x_test)):
    print('Working on image {}'.format(x+1))
    image = x_test[x].reshape(28,28)
    label = str(y_train[x][0])
    counter = x + 1
    filename = '{}_{}.png'.format(counter, label)
    # Check if the dir exists
    if not os.path.isdir(os.path.join(base_dir, test_dir)):
        os.makedirs(os.path.join(base_dir, test_dir))
    # Create the full image path
    path = os.path.join(base_dir, test_dir, filename)
    cv2.imwrite(path, image)
    
image = x_train[0].reshape(28, 28)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('image.png', image)