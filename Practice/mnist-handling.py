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
y_test = emnist["dataset"][0][0][1][0][0][1]    

# store labels for visualization
train_labels = y_train
test_labels = y_test

# For character detection, a list with all ones
y_detect_train = []

for i in range(len(x_train)):
    y_detect_train.append([1])

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
import gc

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
        
# Train OneClassSVM to detect whether an image contains a character or not
from sklearn.svm import OneClassSVM

split_size = int(x_train.shape[0]/100)

x_train_split1 = []

for i in range(split_size):
    x_train_split1.append(x_train[i])


text_detector = OneClassSVM(kernel='rbf')
text_detector.fit(x_train_split1, train_labels)


    