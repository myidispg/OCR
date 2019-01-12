#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:56:35 2018

@author: myidispg
"""
import numpy as np
import cv2
import math
from PIL import Image
import pillowfight

img = cv2.imread('../Test Images/Wireless Drivers.png', 0)
edges = cv2.Canny(img, 100, 200, apertureSize=3)

# Create gradient map using Sobel
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

theta = np.arctan2(sobely64f, sobelx64f)

def swt(theta, edges, sobelx64f, sobely64f):
    # Create an empty image with each pixel value as infinity
    swt = np.empty(theta.shape)
    swt[:] = np.Infinity
    rays = []
    
    # Now iterate over pixels of image, checking canny to see if we are on an edge.
    # If we are, follow a ray normal to the pixel to either the next edge, or the image edge.
    step_x_g = -1 * sobelx64f
    step_y_g = -1 * sobely64f
    mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g)
    grad_x_g = step_x_g/mag_g
    grad_y_g = step_y_g = mag_g
    
    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            if edges[y,x] > 0:
                step_x = step_x_g[y,x] # Step for each pixel
                step_y = step_y_g[y,x]
                mag = mag_g[y,x] # magnitude of that pixel
                grad_x = grad_x_g[y,x] # gradient of that pixel
                grad_y = grad_y_g[y,x]
                ray = []
                ray.append((x,y))
                prev_x, prev_y, i = x, y, 0
                
                while True:
                    i += 1
                    # Move along the direction of the gradient
                    cur_x = math.floor(x + grad_x * i)
                    cur_y = math.floor(y + grad_y * i)
                    if cur_x != prev_x or cur_y != prev_y:
                        #^^ We are at the next pixel now
                        try:
                            if edges[cur_y, cur_x] > 0:
                                # ^^ Found an edge
                                ray.append((cur_x, cur_y))
                                theta_point = theta[y,x]
                                alpha = theta[cur_y, cur_x]
                                if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    thickness = math.sqrt((cur_x - x)^2 - (cur_y - y)^2)
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                    rays.append(ray)
                                break
                            # This is positioned at end to ensure we don't add a point beyond image boundary.
                            ray.append((cur_x, cur_y))
                        except IndexError:
                            # reached image boundary
                            break
                        prev_x = cur_x
                        prev_y = cur_y
    # Compute median SWT
    for ray in rays:
        median = np.median([swt[y,x] for (x,y) in ray])
        for (x,y) in ray:
            swt[y,x] = min(median, swt[y,x])
    # Write SWT image
    cv2.imwrite('../new_img.jpeg', swt)

image = Image.open('../Test Images/Wireless Drivers.png').convert('L')
swt_image = pillowfight.swt(image)
swt_image.save('../swt.png')