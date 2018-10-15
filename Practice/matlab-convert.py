#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:33:00 2018

@author: myidispg
"""

import cv2
import skimage
import numpy as np
import gc

#img = cv2.imread('../Test Images/ReceiptSwiss.jpg',0)
img = cv2.imread('../Test Images/Wireless Drivers.png',0)

mser = cv2.MSER_create()
regions, boxes = mser.detectRegions(img)

# Use regionprops to measure certain properties
mserStats = skimage.measure.regionprops(boxes)

# Vertically stack bbox from mserStats
bbox = []

for stat in mserStats:
    bbox.append(stat['bbox'])
    
bbox = np.asarray(bbox)

# Compute aspect ratio using bounding box data
w = bbox[:, 2] - bbox[:, 0]
h = bbox[:, 3] - bbox[:, 1]
aspectRatio = np.divide(w,h)

del w, h
gc.collect()

# Get some useful regionprops for the image to Threshhold.
eccentricity = []
for stat in mserStats:
    eccentricity.append(stat['eccentricity'])
    
eccentricity = np.asarray(eccentricity)

Solidity = []
for stat in mserStats:
    Solidity.append(stat['Solidity'])
    
Solidity = np.asarray(Solidity)

Extent = []
for stat in mserStats:
    Extent.append(stat['Extent'])
    
Extent = np.asarray(Extent)

EulerNumber = []
for stat in mserStats:
    EulerNumber.append(stat['EulerNumber'])
    
EulerNumber = np.asarray(EulerNumber)

# Threshold the data to determine which regions to remove
aspectRatio = np.transpose(aspectRatio)

filterIdx = [prop for prop in aspectRatio if prop > 3]

new_filterIdx = []

for (x,y) in (filterIdx, )

filterIdx = list(np.where(filterIdx or eccentricity > .995))

filterIdx = list(np.where(filterIdx or Solidity > .3))

filterIdx = np.where(filterIdx or eccentricity > .995)