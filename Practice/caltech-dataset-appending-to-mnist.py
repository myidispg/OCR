#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:11:22 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
from PIL import Image

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
        'dalmation': 67,
        'dollar_bill': 52,
        'dolphin': 65,
        'dragonfly': 68,
        'electric_guitar': 75,
        'elephant': 64,
        'emu': 53,
        'euphonism': 64,
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
        'wild_cat': 67,
        'widsor_chair': 56,
        'wrench': 39,
        'yin_yang': 60
        }

total = 0
for folder in images_dict:
    total += images_dict[folder]
    
base_URL = '../Datasets/101_ObjectCategories/' # foldername/image_0000.jpg

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
  
        im = Image.open(URL).convert('L')
        URL = base_URL + folder + '/'
        
          
