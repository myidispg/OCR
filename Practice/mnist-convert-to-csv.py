# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:27:18 2018

@author: Prashant Goyal
"""

import os
import scipy.misc
import numpy as np

def get_images(imgf, n):
    f = open(imgf, "rb")
    f.read(16)
    images = []
    
    i_count = 0
    j_count = 0
    for i in range(n):
        image = []
        i_count += 1
        for j in range(28*28):
            j_count += 1
            image.append(ord(f.read(1)))
        images.append(image)
    return images

def get_labels(labelf, n):
    l = open(labelf, "rb")
    l.read(8)
    labels = []
    for i in range(n):
        labels.append(ord(l.read(1)))
    return labels

def output_csv(images, labels, outf):
    o = open(outf, "wb")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()

def output_png(images, labels, prefix):
    for i in range(len(images)):
        out = os.path.join(prefix, "%06d-num%d.png"%(i,labels[i]))
        scipy.misc.imsave(out, np.array(images[i]).reshape(28,28))

def csv_and_png(imgf, labelf, prefix, n):
    images = get_images(imgf, n)
    labels = get_labels(labelf, n)
    output_csv(images, labels, "mnist_%s.csv"%prefix)
    # output_png(images, labels, prefix)

csv_and_png("emnist-balanced-test-images-idx3-ubyte", "emnist-balanced-test-labels-idx1-ubyte", "test", 116323)
csv_and_png("t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte",  "test",  116323)


#-------------------------------------------
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []
    
    for i in range(n):
        image = [ord(l.read(1))]
        print(i)
        #print(l)
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()
    
convert("emnist-balanced-test-images-idx3-ubyte", "emnist-balanced-test-labels-idx1-ubyte",
        "mnist_test.csv", 18798)