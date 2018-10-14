#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:50:14 2018

@author: myidispg
"""

import cv2
import numpy as np
import math

img = cv2.imread('../Test Images/Wireless Drivers.png', 0)
edges = cv2.Canny(img, 100, 200)

# Gradient of each pixel using Sobel Filter
sobelX = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
# direction of gradient
theta = np.arctan(sobelX, sobelY)

# Create empty SWT image with each pixel inititalized to infinity
swt = np.empty(theta.shape)
swt[:] = np.Infinity

rays = []

#step_x_g = -1 * sobelX
#step_y_g = -1 * sobelY
#mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g)
#try:
#    grad_x_g = step_x_g/mag_g
#    grad_y_g = step_y_g/mag_g
#except ZeroDivisionError:
#    grad_x_g = 0
#    grad_y_g = 0
    
#grad_x_g = np.nan_to_num(grad_x_g)
#grad_y_g = np.nan_to_num(grad_y_g)

for x in range(edges.shape[0]): # Move along rows
    for y in range(edges.shape[1]): # Move along columns
        print(str(x) + ',' + str(y))
        if edges[y,x] > 0: # We are at an edge in the image
            # Now we trace a ray in the direction of the gradient.
            #(r = p+n.dp), p-> edge pixel, dp-> gradient
#            grad_x = grad_x_g[y,x]
#            grad_y = grad_y_g[y,x]
            ray = []
            ray.append((x,y))
            prev_x, prev_y, i = x, y, 0
            
            while True:
                i += 1
                cur_x = math.floor(x + i*theta[y,x])
                cur_y = math.floor(y + i*theta[y,x])
                
                if cur_x != prev_x or cur_y != prev_y: # We are at the next pixel
                    print(str(cur_x) + ',' + str(cur_y))
                    try:
                        if edges[cur_y, cur_x] > 0: # Found another edge
                            ray.append((cur_x, cur_y))
                            if math.acos((grad_x * -grad_x_g[cur_y, cur_x]) + (grad_y * -grad_y_g[cur_y, cur_x])) < np.pi/2.0:
                                thickness = math.sqrt((cur_x - x)^2 - (cur_y - y)^2) # Find thickness of ray
                                for (rp_x, rp_y) in ray:
                                    swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                rays.append(ray)
                            break
                        ray.append((cur_x, cur_y))
                    except IndexError:
                        break
                    prev_x = cur_x
                    prev_y = cur_y
# Compute Median of SWT to ensure true stroke widths at the corner
for ray in rays:
    
#----------------------------------------
import cv2
import numpy as np
import math
import os

img = cv2.imread('../Test Images/ReceiptSwiss.jpg', 0)
edg_mtg = cv2.Canny(img, 100, 200)    

#-------------------------------------------------------+
# Find stroke derivates
#-------------------------------------------------------+
dx = cv2.Sobel(edg_mtg, cv2.CV_64F, 1, 0, ksize=5)
dy = cv2.Sobel(edg_mtg, cv2.CV_64F, 0, 1, ksize=5)

cwd = os.getcwd() # Current working directory

cv2.imwrite(os.path.join(cwd, "sobelx.jpg"), dx)
cv2.imwrite(os.path.join(cwd, "sobely.jpg"), dy)

theta = np.arctan2(dy, dx) # Derivates for each pixel

cv2.imwrite(os.path.join(cwd, "theta.jpg"), (theta + np.pi)*255/(2*np.pi))

#-------------------------------------------------------+
# Perform Stroke Width Transform
#-------------------------------------------------------+
swt = np.empty(theta.shape)
swt[:] = np.Infinity
rays = []

step_x_g = -dx
step_y_g = -dy

magnitudes = np.sqrt(np.square(dx) + np.square(dy))

grad_x_g = step_x_g/magnitudes
grad_y_g = step_y_g/magnitudes

edges_locations = np.argwhere(edg_mtg != 0) # indices of all pixels that are edges based on canny

for edg in edges_locations:
    step_x = step_x_g[edg[0], edg[1]]
    step_y = step_y_g[edg[0], edg[1]]
    mag = magnitudes[edg[0], edg[1]]
    grad_x = grad_x_g[edg[0], edg[1]]
    grad_y = grad_y_g[edg[0], edg[1]]

    ray = []
    ray.append((edg[0], edg[1]))
    prev_x, prev_y, i = edg[0], edg[1], 0

    while True:
        i += 1
        cur_x = np.floor(edg[0] + grad_x * i)
        cur_y = np.floor(edg[1] + grad_y * i)

        if cur_x != prev_x or cur_y != prev_y:
            # we have moved to the next pixel!
            try:
                if edg_mtg[cur_y, cur_x] > 0:
                    # found edge,
                    ray.append((cur_x, cur_y))
                    theta_point = theta[edg[1], edg[0]]
                    alpha = theta[cur_y, cur_x]
                    if np.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                        thickness = np.sqrt( (cur_x - edg[0]) * (cur_x - edg[0]) + (cur_y - edg[1]) * (cur_y - edg[1]) )
                        for (rp_x, rp_y) in ray:
                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                        rays.append(ray)
                    break
                # this is positioned at end to ensure we don't add a point beyond image boundary
                ray.append((cur_x, cur_y))
            except IndexError:
                # reached image boundary
                break
            prev_x = cur_x
            prev_y = cur_y

# Compute median SWT
for ray in rays:
    median = np.median([swt[y, x] for (x, y) in ray])
    for (x, y) in ray:
        swt[y, x] = min(median, swt[y, x])                            
                            
cv2.imwrite('swt.jpeg', swt)                    

for x in range(swt.shape[0]):
    for y in range(swt.shape[1]):
        if swt[x,y] == np.Infinity:
            print(True)
        else:
            print(False)
            
import cv2
img = cv2.imread('../Test Images/scene-text-1.jpeg', 0)
mser = cv2.MSER_create()
gray_img = img.copy()

regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
cv2.imwrite('../text.jpg', gray_img) #Saving
