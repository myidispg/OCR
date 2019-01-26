#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:27:49 2019

@author: myidispg
"""
import numpy as np

class SegmentCharacters:
    
    def __init__(self, image):
        self.image = image
        # Empty list of columns
        self.psc = []
        
    def find_char_images(self):
        segments = self.segment()
        coords = [0, segments[0] + 6]
        for x in range(1, len(segments)):
            if segments[x] - segments[x-1] == 1:
                pass
            else:
                coords.append(segments[x] + 6)
                
        return coords
        
        
    def segment(self):
        width, height = self.image.shape
#        print('width- {}, height- {}'.format(width, height))
        # Find the sum of pixels in all columns
        col_sum = np.sum(self.image, axis=0)
        # Find the first column with a pixel
        first_pix_col = None
        for x in range(len(col_sum)):
            if col_sum[x] != 0:
                first_pix_col = x
                break
        # Find columns with sum either 0 or 1
        psc = []
        for x in range(first_pix_col, len(col_sum)):
            if col_sum[x] == 0:# or col_sum[x] == 1:
                psc.append(x)
        return psc