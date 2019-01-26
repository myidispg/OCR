#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:02:41 2019

@author: myidispg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('../MNIST_Dataset/emnist-byclass-train.csv')
df_test = pd.read_csv('../MNIST_Dataset/emnist-byclass-test.csv')