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

BASE_URL = '../Datasets/256_ObjectCategories'

list_contents = os.listdir(BASE_URL)

img_dict = {}

for direc in list_contents:
    img_dict[direc] = len(os.listdir(BASE_URL + '/' + direc))
    
