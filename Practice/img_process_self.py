#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:30:25 2018

@author: myidispg
"""

import cv2
import numpy as np
import math
from scipy import ndimage

img_g = cv2.imread('../Test Images/g-text-test.jpeg', 0)
img_g = cv2.resize(255- img_g, (28,28))

img_c = cv2.imread('../Test Images/c-text-test.jpeg', 0)
img_c = cv2.resize(255- img_c, (28,28))

img = cv2.imread('../Test Images/i-text-test.jpeg', 0)

img = cv2.resize(255- img, (28,28))

(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Fit the image in a 20 by 20 pixel box. For this, remove all rows and columns of 0's

# Remove top 0s
while np.sum(img[0]) == 0:
    img = img[1:]
# Remove left 0s
while np.sum(img[:,0]) == 0:
    img = np.delete(img, 0, 1)
# Remove bottom 0s
while np.sum(img[-1]) == 0:
    img = img[:-1]
# Remove right 0s
while np.sum(img[:, -1]) == 0:
    img = np.delete(img, -1, 1)
    
rows, cols = img.shape 

# Resize the resultant image to a 20x20 dimensions
if rows>cols:
    factor = 20.0/rows
    rows = 20
    cols = int(cols*factor)
    img = cv2.resize(img, (cols, rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(rows*factor)
    img = cv2.resize(img, (cols, rows))
                           
# Get the padding to be added on all sides. Top Bottom will be same, Left Right will same.
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))

up_pad = int((28-rows)/2)
bottom_pad = 28-up_pad-rows
left_pad = int((28-cols)/2)
right_pad = 28-left_pad-cols

img = np.lib.pad(img, ((bottom_pad, up_pad), (right_pad, left_pad)), 'constant')

cv2.imwrite("../Test Images/pro-5-non-text-test.jpeg", img)



