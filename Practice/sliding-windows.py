#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:30:37 2018

@author: myidispg
"""

import argparse
import time
import cv2
import imutils

def pyramid(image, scale=1.5):
    yield image
    
    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width=w)
        
        yield image

# Suggested step size is 4-8. Examinign each pixel with computationally expensive.
def sliding_window(image, stepSize, windowSize):
    # Slide the windows Horizontaly
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x,y, image[y:y + windowSize[1], x:x + windowSize[0]])
      
img = cv2.imread('../Test Images/Wireless Drivers.png')
(winW, winH) = (28,28)

# loop over the image pyramid
for resized in pyramid(img, scale = 1.5):
    # loop over each resized window of the image pyramid
    for (x,y, window) in sliding_window(resized, stepSize = 4, windowSize = (winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            print('x- ' + str(x))
            print('y- ' + str(y))
            print('window' + str(window))
            continue
            
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
 
		# since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x,y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
        
# Now we can work on saving each windows and running detection on it. 
