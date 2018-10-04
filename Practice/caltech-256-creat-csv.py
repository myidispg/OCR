#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:39:26 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import gc

import os

BASE_URL = '../Datasets/256_ObjectCategories/'

list_contents = os.listdir(BASE_URL)

img_dict = {}

for direc in list_contents:
    img_dict[direc] = len(os.listdir(BASE_URL + direc))
    
del list_contents, direc
gc.collect()

# Fixing count anomalies
img_dict['056.dog'] = 102
img_dict['198.spider'] = 108

pixel_list = []

for folder in img_dict:
    URL = BASE_URL + folder + '/'
    print(URL)
    for i in range(1, img_dict[folder] + 1):
        prefix_file = folder.split('.')
        if i <= 9:
            URL = URL + prefix_file[0] + '_000' + str(i) + '.jpg'
        elif i > 9 and i < 100:
            URL = URL + prefix_file[0] + '_00' + str(i) + '.jpg'
        else:
            URL = URL + prefix_file[0] + '_0' + str(i) + '.jpg'
        # Open the image in grayscale mode
        im = Image.open(URL).convert('L')
        # Resize the image to 28x28 pixels
        im = im.resize((28,28))
        # Invert the image to increase dataset
        im_invert = ImageOps.invert(im)
        
        # Get pixel value of non-inverted image
        pix_val_1 = list(im.getdata())
        # Get pixel value of iameg inverted image
        pix_val_2 = list(im_invert.getdata())
        
        # Append both the image's pixel values with 0 label to a big list.
        pixel_list.append(pix_val_1)
        pixel_list.append(pix_val_2)        
        # Reset URL otherwise it kept appending.
        URL = BASE_URL + folder + '/'

del folder, i, im, im_invert, img_dict, pix_val_1, pix_val_2, prefix_file
gc.collect()

df_images = pd.DataFrame(pixel_list)

df_images.to_csv('Non-text-images-256.csv', index=False)