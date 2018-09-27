# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:48:33 2018

@author: Prashant Goyal
"""
#------Code on python-mnist website----------------------
from mnist import MNIST
import numpy as np
import pandas as pd

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

emnist = spio.loadmat("../datasets/matlab/emnist-byclass.mat")
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]    

# store labels for visualization
train_labels = y_train
test_labels = y_test

# reshape using matlab order
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

y_train.shape

# labels should be onehot encoded
import keras
y_train = keras.utils.to_categorical(y_train, 62)
y_test = keras.utils.to_categorical(y_test, 62)

samplenum = 5437
import matplotlib.pyplot as plt

img = x_train[samplenum]

# visualize image
plt.imshow(img[0], cmap='gray')