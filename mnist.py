#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:48:23 2019

@author: myidispg
"""

from mnist import MNIST
import gzip
import numpy as np

image_size = 28
num_images = 60000

# Open the images
f = gzip.open('../MNIST_Dataset/train-images-idx3-ubyte.gz', 'r')
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

# Open the labels
f = gzip.open('../MNIST_Dataset/train-labels-idx1-ubyte.gz', 'r')
buf = f.read(32 * num_images * 8)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)



#mndata = MNIST('../MNIST_Dataset/')
#mndata.gz = True
#images, labels = mndata.load_training()
#
#images = np.asarray(images)
#labels = np.asarray(labels)

from sklearn.preprocessing import OneHotEncoder

import cv2
image = np.reshape(data[0], (28,28))
cv2.imshow('image', data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


