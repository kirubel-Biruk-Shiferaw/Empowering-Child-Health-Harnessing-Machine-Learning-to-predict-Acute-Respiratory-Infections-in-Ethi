#!/usr/bin/env python
# coding: utf-8
#This module mainly does the comparision of resampling methods as the data was imbalanced.

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
os.chdir('..')
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import glob

#set the path/data location
data = pd.read_excel("your path")

#we drope 3 variables for very low correlation and high imbalance
data= data.drop('wasting', axis=1)
data= data.drop('water', axis=1)
data= data.drop('vitamin', axis=1)

#define the independent and outcome variables
X= data.drop(["outcomevar"], axis = 1)
Y= data["outcomevar"]

# Train/Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#instal imblearn
get_ipython().system('pip install imblearn')

#see the imbalance
sns.countplot(x ='outcomevar', data=data) 

# Baseline Model training with the raw dataset
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 4, max_features = 3, bootstrap = True, random_state = 18).fit(x_train, y_train)

from sklearn.metrics import classification_report
baseline_prediction = clf.predict(x_test)

# Check the model performance
print(classification_report(y_test, baseline_prediction))

#we will try 3 methods and choose the best one
#option one- Random oversampler/ Randomly over sample the minority class
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros= ros.fit_resample(x_train, y_train)

# Check the number of records after over sampling
from collections import Counter
print(sorted(Counter(y_train_ros).items()))

# Convert the resampled data to a pandas DataFrame
resampled_df = pd.DataFrame(X_train_ros, columns=X.columns)
resampled_df['outcomevar'] = y_train_ros

#fit the resampled data
ros_model = clf.fit(X_train_ros, y_train_ros)
ros_prediction = ros_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, ros_prediction)) # 40 % increase in recall

print("X_train_ros shape:", X_train_ros.shape)
print("x_test shape:", x_test.shape)

# SMOTE: Randomly over sample the minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote= smote.fit_resample(x_train, y_train)

# Check the number of records after over sampling
print(sorted(Counter(y_train_smote).items()))

smote_model = clf.fit(X_train_smote, y_train_smote)
smote_prediction = smote_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, smote_prediction)) # recall increase to 61% but also accuracy down to 64%

#now try nearmiss undersampling
from imblearn.under_sampling import NearMiss
nearmiss = NearMiss()
X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(x_train, y_train)

# Check the number of records after over sampling
print(sorted(Counter(y_train_nearmiss).items()))

nearmiss_model = clf.fit(X_train_nearmiss, y_train_nearmiss)
nearmiss_prediction = nearmiss_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, nearmiss_prediction)) #recall 80%

# Randomly under sample the majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus= rus.fit_resample(x_train, y_train)

# Check the number of records after under sampling
print(sorted(Counter(y_train_rus).items()))

rus_model = clf.fit(X_train_rus, y_train_rus)
rus_prediction = rus_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, rus_prediction))


