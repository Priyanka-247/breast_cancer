#!/usr/bin/env python3

#####################################################
##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
#####################################################
#
# Project by Raul Eulogio
#
# Project found at: https://www.inertia7.com/projects/3
#

# Import Packages -----------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from urllib.request import urlopen

import os

UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean',
         'concavity_mean','concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se',
         'area_se', 'smoothness_se', 'compactness_se',
         'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst',
         'concavity_worst', 'concave_points_worst',
         'symmetry_worst', 'fractal_dimension_worst']

dx = ['Malignant', 'Benign']

from sklearn.datasets import load_breast_cancer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOCAL_DATA_PATH = os.path.join(BASE_DIR, 'src', 'pyspark', 'data', 'data.txt')

try:
    breast_cancer = pd.read_csv(urlopen(UCI_data_URL, timeout=2), names=names)
    breast_cancer.set_index(['id_number'], inplace=True)
    breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
except Exception:
    if os.path.exists(LOCAL_DATA_PATH):
        df_raw = pd.read_csv(LOCAL_DATA_PATH, header=None, delim_whitespace=True)
        df_raw['id_number'] = range(1, len(df_raw) + 1)
        df_raw.rename(columns={0: 'diagnosis'}, inplace=True)
        feature_cols = names[2:]
        for idx, col in enumerate(feature_cols):
            df_raw.rename(columns={idx + 1: col}, inplace=True)
        breast_cancer = df_raw[['id_number', 'diagnosis'] + feature_cols]
        breast_cancer.set_index(['id_number'], inplace=True)
    else:
        sk_data = load_breast_cancer(as_frame=True)
        df_sk = sk_data.frame
        feature_cols = names[2:]
        df_sk.columns = feature_cols + ['diagnosis']
        # Map target: 0 (Malignant) -> 1, 1 (Benign) -> 0
        df_sk['diagnosis'] = df_sk['diagnosis'].map({0: 1, 1: 0})
        df_sk['id_number'] = range(1, len(df_sk) + 1)
        breast_cancer = df_sk[['id_number', 'diagnosis'] + feature_cols].set_index('id_number')



for col in breast_cancer:
	pd.to_numeric(col, errors='coerce')

# For later use in CART models
names_index = names[2:]

# Create Training and Test Set ----------------------------------
feature_space = breast_cancer.iloc[:,
                                   breast_cancer.columns != 'diagnosis']
feature_class = breast_cancer.iloc[:,
                                   breast_cancer.columns == 'diagnosis']


training_set, test_set, class_set, test_class_set = train_test_split(feature_space,
                                                                    feature_class,
                                                                    test_size = 0.20,
                                                                    random_state = 42)

# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

# Scaling dataframe
scaler = MinMaxScaler()

scaler.fit(training_set)

training_set_scaled = scaler.fit_transform(training_set)
test_set_scaled = scaler.transform(test_set)
