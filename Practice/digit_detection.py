#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:55:37 2018

@author: myidispg
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, :].values

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)