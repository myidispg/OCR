#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:27:49 2019

@author: myidispg
"""

class SegmentCharacters:
    
    def __init__(self, image):
        self.image = image
        # Empty list of columns
        self.psc = []
        
    def segment(self):
        for x in range(self.image.shape[0]):
            total = 0
            for y in range(self.image.shape[1]):
                total += self.image[x][y]
            if total == 1:
                self.psc.append(x)
            return self.join_segment(5)
                
    # Join the segments that are within a threshhold        
    def join_segment(self, threshold):
        final_segments = []
        for x in self.psc:
            segments_to_join = []
            for y in range(x, len(self.psc)):
                if y < (x + threshold):
                    if self.psc[y] <= threshold:
                        segments_to_join.append(self.psc[y])
            final_segments.append(self.merge_group(segments_to_join))
            
        return final_segments
    
    def merge_group(self, segments):
        segments = segments.sort()
        return segments[int(len(segments))]