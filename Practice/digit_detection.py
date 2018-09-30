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

# Generate positive labels for digits
y_train_pos = []

for char in y_train:
    y_train_pos.append(1)


# Train our OneClassSVM model
from sklearn.svm import OneClassSVM
digit_detect = OneClassSVM()
digit_detect.fit(X_train, y_train_pos)

from sklearn.ensemble import IsolationForest
digit_detect = IsolationForest(n_estimators=10, verbose=100, n_jobs=-1)
digit_detect.fit(X_train, y_train_pos)

y_pred = digit_detect.predict(X_test)