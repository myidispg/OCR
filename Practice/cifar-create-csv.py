#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:43:55 2018

@author: myidispg
"""

import pickle

with open('../Datasets/cifar-100-python/train', 'rb') as fo:
    dict_train = pickle.load(fo, encoding='bytes')  
    
train_images = dict_train[b'data'].reshape(50000, 32,32,3)

from PIL import Image


for i in range(train_images.shape[0]):
    image = Image.fromarray(train_images[i], 'RGB').convert('L')
    image = image.resize((28,28))
    pix_val = list(image.getdata())
    pix_value.append(0)
    print(len(pix_value))
    

