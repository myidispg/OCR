#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:31:18 2018

@author: myidispg
"""

import cv2

def img_pyramid(image, scale):
    if img.shape[0] > 28*4 and img.shape[1] > 28*4:
        print('inside image pyramid- ')
        print(int(img.shape[0]/scale))
        print(int(img.shape[1]/scale))
        image = cv2.resize(image, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    else:
        return image
    
    return image

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

sliding_windows(img)

