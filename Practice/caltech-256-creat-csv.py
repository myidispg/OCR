#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:39:26 2018
@author: myidispg
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import gc

import os

# ----------Bring the image to MNSIT format--------------------
def preprocess_image(pix_val):
        
    for i in range(len(pix_val)):
       pix_val[i] = 0 if pix_val[i] <= 140 else pix_val[i]
        
    return pix_val

# ----------CSV for Caltech101 Database------------------------------------
images_dict = {
        'accordion': 55,
        'airplanes': 800,
        'anchor': 42,
        'ant': 42,
        'BACKGROUND_Google': 467,
        'barrel': 47,
        'bass': 54,
        'beaver': 46,
        'binocular': 33,
        'bonsai': 128,
        'brain': 98,
        'brontosaurus':43,
        'buddha': 85,
        'butterfly': 91,
        'camera': 50,
        'cannon':43,
        'car_side': 123,
        'ceiling_fan': 47,
        'cellphone': 59,
        'chair': 62,
        'chandelier': 107,
        'cougar_body': 47,
        'cougar_face': 69,
        'crab': 73,
        'crayfish': 70,
        'crocodile': 50,
        'crocodile_head': 51,
        'cup': 57,
        'dalmatian': 67,
        'dollar_bill': 52,
        'dolphin': 65,
        'dragonfly': 68,
        'electric_guitar': 75,
        'elephant': 64,
        'emu': 53,
        'euphonium': 64,
        'ewer': 85,
        'Faces': 435,
        'Faces_easy': 435,
        'ferry': 67,
        'flamingo': 67,
        'flamingo_head': 43,
        'garfield': 34,
        'gerenuk': 34,
        'gramophone': 51,
        'grand_piano': 99,
        'hawksbill':100,
        'headphone': 42,
        'hedgehog': 54,
        'helicopter': 88,
        'ibis':80,
        'inline_skate': 31,
        'joshua_tree': 64,
        'kangaroo': 86,
        'ketch': 114,
        'lamp': 61,
        'laptop': 81,
        'Leopards': 200,
        'llama': 78,
        'lobster': 41,
        'lotus': 66,
        'mandolin': 43,
        'mayfly': 40,
        'menorah': 87,
        'metronome': 32,
        'minaret': 76,
        'Motorbikes': 798,
        'nautilus': 55,
        'octopus': 35,
        'okapi': 39,
        'pagoda': 47,
        'panda': 38,
        'pigeon': 45,
        'pizza': 53,
        'platypus': 34,
        'pyramid': 57,
        'revolver': 82,
        'rhino': 59,
        'rooster': 49,
        'saxophone': 40,
        'schooner': 63,
        'scissors':39,
        'scorpion': 84,
        'sea_horse': 57,
        'snoopy': 35,
        'soccer_ball':64,
        'stapler': 45,
        'starfish': 86,
        'stegosaurus': 59,
        'stop_sign': 64,
        'strawberry': 35,
        'sunflower': 85,
        'tick': 49,
        'trilobite': 86,
        'umbrella': 75,
        'watch': 239,
        'water_lilly': 37,
        'wheelchair': 57,
        'wild_cat': 34,
        'windsor_chair': 56,
        'wrench': 39,
        'yin_yang': 60
        }

total = 0
for folder in images_dict:
    total += images_dict[folder]
    
base_URL = '../Datasets/101_ObjectCategories/' # foldername/image_0000.jpg

pixel_list = []

for folder in images_dict:
    URL = base_URL + folder + '/'
    print(URL)
    for i in range(1, images_dict[folder] + 1):
        if i <= 9:
            URL = URL + 'image_000' + str(i) + '.jpg'
        elif i > 9 and i < 100:
            URL = URL + 'image_00' + str(i) + '.jpg'
        else:
            URL = URL + 'image_0' + str(i) + '.jpg'
        # Open the image in grayscale mode
        im = Image.open(URL).convert('L')
        # Resize the image to 28x28 pixels
        im = im.resize((28,28))
        # Invert the image to increase dataset
        im_invert = ImageOps.invert(im)
        
        # Get pixel value of non-inverted image
        pix_val_1 = list(im.getdata())
        pix_val_1_1= preprocess_image(pix_val_1)
        # Get pixel value of iameg inverted image
        pix_val_2 = list(im_invert.getdata())
        pix_val_2_1 = preprocess_image(pix_val_2)
        
        # Append both the image's pixel values with 0 label to a big list.
        pixel_list.append(pix_val_1)
        pixel_list.append(pix_val_2)
        pixel_list.append(pix_val_1_1)
        pixel_list.append(pix_val_2_1)       
        # Reset URL otherwise it kept appending.
        URL = base_URL + folder + '/'



del i, base_URL, folder, im, im_invert, images_dict, pix_val_1, pix_val_2, total
gc.collect()


df_images = pd.DataFrame(pixel_list)

df_images.to_csv('../Datasets/Non-text-images-101.csv', index=False)

del df_images, pixel_list
gc.collect()

#------------------CSV for Caltech 256 Databse----------------------------

BASE_URL = '../Datasets/256_ObjectCategories/'

list_contents = os.listdir(BASE_URL)

img_dict = {}

for direc in list_contents:
    img_dict[direc] = len(os.listdir(BASE_URL + direc))
    
del list_contents, direc
gc.collect()

# Fixing count anomalies
img_dict['056.dog'] = 102
img_dict['198.spider'] = 108

pixel_list = []

for folder in img_dict:
    URL = BASE_URL + folder + '/'
    print(URL)
    for i in range(1, img_dict[folder] + 1):
        prefix_file = folder.split('.')
        if i <= 9:
            URL = URL + prefix_file[0] + '_000' + str(i) + '.jpg'
        elif i > 9 and i < 100:
            URL = URL + prefix_file[0] + '_00' + str(i) + '.jpg'
        else:
            URL = URL + prefix_file[0] + '_0' + str(i) + '.jpg'
        # Open the image in grayscale mode
        im = Image.open(URL).convert('L')
        # Resize the image to 28x28 pixels
        im = im.resize((28,28))
        # Invert the image to increase dataset
        im_invert = ImageOps.invert(im)
        
        pix_val_1 = list(im.getdata())
        pix_val_1_1= preprocess_image(pix_val_1)
        # Get pixel value of iameg inverted image
        pix_val_2 = list(im_invert.getdata())
        pix_val_2_1 = preprocess_image(pix_val_2)
        
        # Append both the image's pixel values with 0 label to a big list.
        pixel_list.append(pix_val_1)
        pixel_list.append(pix_val_2)
        pixel_list.append(pix_val_1_1)
        pixel_list.append(pix_val_2_1)            
        # Reset URL otherwise it kept appending.
        URL = BASE_URL + folder + '/'

del folder, i, im, im_invert, img_dict, pix_val_1, pix_val_2, prefix_file
gc.collect()

df_images = pd.DataFrame(pixel_list)

df_images.to_csv('../Datasets/Non-text-images-256.csv', index=False)

del pixel_list, df_images
gc.collect()

#----------------Join both Caltech datasets into 1.----------
img_101 = pd.read_csv('../Datasets/Non-text-images-101.csv')
img_256 = pd.read_csv('../Datasets/Non-text-images-256.csv')

df_final = pd.concat([img_101, img_256], axis=0)

del img_101, img_256
gc.collect()

df_final.to_csv('../Datasets/Non-text-images-1.csv', index=False)

del df_final
gc.collect()

# Delete the two remaining files
if os.path.exists('../Datasets/Non-text-images-101.csv'):
    os.remove("../Datasets/Non-text-images-101.csv")
if os.path.exists('../Datasets/Non-text-images-256.csv'):
    os.remove('../Datasets/Non-text-images-256.csv')
