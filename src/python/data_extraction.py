#!/usr/bin/env python3

#####################################################
##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
#####################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

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

# Load Wisconsin Breast Cancer Dataset natively via scikit-learn
sk_data = load_breast_cancer(as_frame=True)
df = sk_data.frame

feature_cols = names[2:]
df.columns = feature_cols + ['target']

# Target mapping: sklearn 0=Malignant, 1=Benign -> UCI M=1, B=0
df['diagnosis'] = df['target'].map({0: 1, 1: 0})
df['id_number'] = range(842302, 842302 + len(df))

breast_cancer = df[['id_number', 'diagnosis'] + feature_cols].set_index('id_number')

for col in breast_cancer:
	pd.to_numeric(col, errors='coerce')

# For later use in CART models
names_index = names[2:]

# Create Training and Test Set ----------------------------------
feature_space = breast_cancer.iloc[:, breast_cancer.columns != 'diagnosis']
feature_class = breast_cancer.iloc[:, breast_cancer.columns == 'diagnosis']

training_set, test_set, class_set, test_class_set = train_test_split(
    feature_space,
    feature_class,
    test_size=0.20,
    random_state=42
)

# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

# Scaling dataframe
scaler = MinMaxScaler()
scaler.fit(training_set)

training_set_scaled = scaler.fit_transform(training_set)
test_set_scaled = scaler.transform(test_set)
