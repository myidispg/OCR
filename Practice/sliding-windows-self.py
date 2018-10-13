#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:31:18 2018

@author: myidispg
"""

import cv2
import numpy as np

def img_pyramid(image, scale):
    if img.shape[0] > 28*4 and img.shape[1] > 28*4:
        print('inside image pyramid- ')
        print(int(img.shape[0]/scale))
        print(int(img.shape[1]/scale))
        image = cv2.resize(image, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    else:
        return image
    
    return image

def img_pyramid(image, scale):
    if img.shape[0] > 28*4 and img.shape[1] > 28*4:
        return (int(image.shape[1]/scale), int(image.shape[0]/scale))
    else:
        return False
    
img = cv2.imread('../Test Images/Wireless Drivers.png', 0)

# Convert the image to the nearest multiple of windows size i.e. 28
(imgH, imgW) = (img.shape[0] - (img.shape[0]%28), img.shape[1] - img.shape[1]%28)
img = cv2.resize(img, (imgW, imgH))
del imgW, imgH
# Image dimensions are rows x columns(height X width)
img = img_pyramid(img, 1.25)

def sliding_windows(img, windowSize = 28, stepSize= 4):
    text_pixels = []
    img_shape = img.shape
    img = img_pyramid(img, 1.25)
    while img.shape != img_shape:
        print(img.shape)
        img = img_pyramid(img, 1.25)
        img_shape = img.shape
        
def sliding_windows(img, windowSize = 28, stepSize= 4):
    # loop to generate image pyramid with image size reduced by scale
    while img.shape[0] > 28*4 and img.shape[1] > 28*4:
        img = cv2.resize(img, img_pyramid(img, scale = 5))
        print(img.shape)
        # loops to iterate over each column in wach row and run text detection model.
        for x in range(0, img.shape[0]-windowSize, stepSize):
            for y in range(0, img.shape[1]-windowSize, stepSize):
                print(str(x) + ',' + str(y))
                winX = x+windowSize
                winY = y+windowSize
                new_img = img_numpy_array(img, x, y, windowSize)
                
                
        
def img_numpy_array(img, row, column, windowSize):
    array = []
    for x in range(row, row+windowSize):
        row = []
        for y in range(column, column+windowSize):
            row.append(img[x][y])
        array.append(row)
    return np.asarray(array)

window = img_numpy_array(img, 10, 10, 5)

sliding_windows(img)

