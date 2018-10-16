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
from opencv_regionprops import Contour

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

del mserStats
gc.collect()

# Threshold the data to determine which regions to remove

# loop to count true in filteridx
filterIdx = []

trueCount = 0
for filters in filterIdx:
    if filters == True:
        trueCount += 1

for prop in aspectRatio:
    filterIdx.append(True if prop > 3 else False) # True count 127

for i in range(len(filterIdx)):
    if filterIdx[i] or eccentricity[i] > .995: # True count 151
        filterIdx[i] = True
        
for i in range(len(filterIdx)):
    if filterIdx[i] or Solidity[i] < .3: # True count 151
        filterIdx[i] = True

for i in range(len(filterIdx)):
    if filterIdx[i]:
        if Extent[i] > 0.9 or Extent[i] < 0.2: # True count 151
          filterIdx[i] = True  
          
for i in range(len(filterIdx)):
    if filterIdx[i] or EulerNumber[i] < -4: # True count 151
        filterIdx[i] = True

#-------------
import cv2
import skimage
import numpy as np
import gc
from opencv_regionprops import Contour

img = cv2.imread('../Test Images/Wireless Drivers.png',0)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Filetr contours with less than 5 points

contoursList = [contour for contour in contours[1] if contour.shape[0] >= 5]
    
bbox = []
for cnt in contoursList:
    c = Contour(img, cnt)
    bbox.append(c.bounding_box)
    

    
#---------------------------
import cv2
import numpy as np

#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path
img = cv2.imread('../Test Images/book-cover-abstract-art.jpeg')
#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh,gray = cv2.threshold(gray,127,255,0)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)
 