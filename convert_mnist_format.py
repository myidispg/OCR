#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:12:21 2019

@author: myidispg
"""

import numpy as np
from scipy import ndimage
import cv2
import math

class ConvertMNISTFormat():
    
    def __init__(self, image):
        self.image = image
    
    def process_image(self):
#        self.image = np.subtract(255, self.image)
        # better black and white version
        # 128 is the threshhold value. Above it the value will be 255 and below will be 0. Controlles by THRESH_BINARY
        # cv2.THRESH_OTSU = 
#        (thresh, self.image) = cv2.threshold(self.image, 5, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#        print(self.image.shape)
        # Remove empty rows and columns.
        while np.sum(self.image[0]) == 0:
            self.image = self.image[1:]
        
        while np.sum(self.image[:,0]) == 0:
            self.image = np.delete(self.image,0,1)
        
        while np.sum(self.image[-1]) == 0:
            self.image = self.image[:-1]
        
        while np.sum(self.image[:,-1]) == 0:
            self.image = np.delete(self.image,-1,1)
            
        rows,cols = self.image.shape
        
            # resize to 20 by 20.
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            # first cols than rows
            self.image = cv2.resize(self.image, (cols,rows), interpolation = cv2.INTER_AREA)
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            # first cols than rows
            self.image = cv2.resize(self.image, (cols, rows), interpolation = cv2.INTER_AREA)
            
        # Add a padding on all sides to turn into 28 * 28
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        self.image = np.lib.pad(self.image,(rowsPadding,colsPadding),'constant')
        
        shiftx,shifty = self.get_best_shift()
        shifted = self.shift(shiftx,shifty)
        self.image = shifted
        
        return self.image
    
    def get_best_shift(self):
        cy,cx = ndimage.measurements.center_of_mass(self.image)

        rows,cols = self.image.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)
    
        return shiftx,shifty
    
    def shift(self, sx,sy):
        rows,cols = self.image.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(self.image,M,(cols,rows))
        return shifted
