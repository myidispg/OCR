#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:31:18 2018

@author: myidispg
"""

import cv2

def img_pyramid(image, scale):
    image = cv2.resize(image, (int(img.shape[0]/scale), int(img.shape[1]/scale)))
    
    return image

img = cv2.imread('../Test Images/Wireless Drivers.png', 0)

# Convert the image to the nearest multiple of windows size i.e. 28
(imgH, imgW) = (img.shape[0] - (img.shape[0]%28), img.shape[1] - img.shape[1]%28)

img = cv2.resize(img, (imgW, imgH))
# Image dimensions are rowsxcolumns

(winW, winH) = (28,28)

img = img_pyramid(img, 1.5)
    