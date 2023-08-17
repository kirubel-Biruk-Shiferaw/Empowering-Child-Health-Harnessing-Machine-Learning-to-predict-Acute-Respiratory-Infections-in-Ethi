#!/usr/bin/env python
# coding: utf-8
# This module is about the comparision of the tested models before choosing the main model

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

data = pd.read_excel("your path to the dataset")

#summary of the data
data.info()

#see the missing values
data.isnull().sum()

#replacing unecessary signs and symbols
data = data.replace(r'^\s*$', np.nan, regex=True)

#see the heatmap of missing values
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# see the nature of outcome variable's class imbalance
sns.countplot(x='outcomevar', data=data) #The imbalance is unfair in our case

#calculate the correlation 
data.corr() # we want to see the correlation because we dont want features with no correlation with the outcome variable and still imbalanced feature.

#see the correlation heatmap
sns.heatmap(data.corr(), cmap='coolwarm') #Very good correlation in our case

# Plot all the variables using histogram subplots
fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, figsize=(10,len(data.columns)*4))
for i, col in enumerate(data.columns):
    sns.histplot(data[col], ax=axes[i], kde=False, color='b')
    axes[i].set_title(f"Distribution of {col}")

# Show the plot
plt.tight_layout()
plt.show()

#we drope 3 variables for very low correlation and high imbalance
data= data.drop('wasting', axis=1)
data= data.drop('water', axis=1)
data= data.drop('vitamin', axis=1)

#define the independent and dependent variable
X= data.drop(["outcomevar"], axis = 1)
Y= data["outcomevar"]

# Train/Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Model training
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 4, max_features = 3, bootstrap = True, random_state = 18).fit(x_train, y_train)

from sklearn.metrics import classification_report
baseline_prediction = clf.predict(x_test)

# Check the model performance
print(classification_report(y_test, baseline_prediction))

#Baseline Model hyperparameter tuning was done but no significant difference observed.
# Define the Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4,6,8, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'criterion' :['gini', 'entropy'],
    'random_state' : [42]
}


# Define the K-Fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the Grid Search with K-Fold cross-validation
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=kf)

# Fit the Grid Search to the diabetes data
grid_search.fit(X, Y)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


rf_tuned = RandomForestClassifier(criterion = 'gini', max_depth = 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators = 150, random_state = 42).fit(x_train, y_train)
tuned_baseline_prediction = rf_tuned.predict(x_test)

# Check the model performance
print(classification_report(y_test, tuned_baseline_prediction))

# as we can see from the recall metrics, the outcome variable imbalance highly suppressed the model's performance in identifing minority class (1= ARI yes) and we want our model to identify more yes cases and to resample the data is appropriate to improve model performance.
#please refer to the resampling comarision file to reproduce the resampling technique
#We found that nearmiss undersampling improved the model recall performance to 80 % so we will use this resampling method to train all the models. 

from imblearn.under_sampling import NearMiss
nearmiss = NearMiss()
X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(x_train, y_train)

# Check the number of records after over sampling
from collections import Counter
print(sorted(Counter(y_train_nearmiss).items()))

nearmiss_model = clf.fit(X_train_nearmiss, y_train_nearmiss)
nearmiss_prediction = nearmiss_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, nearmiss_prediction)) #recall 80%


# Model 1: DT (Decision Tree)

x_train, x_test, y_train, y_test = train_test_split(X_train_nearmiss, y_train_nearmiss, test_size = 0.2, random_state = 42)

from sklearn.tree import DecisionTreeClassifier

#Grid search (already done)  and training
dtc_tuned = DecisionTreeClassifier(criterion= 'gini', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2,  random_state= 42)
#fit the model
dtc_tuned.fit(x_train, y_train)
#predict
dtc_pred= dtc_tuned.predict(x_test)
print(classification_report(y_test, dtc_pred))

#see the metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
cm = confusion_matrix(y_test, dtc_pred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, dtc_pred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

#see the roc curve
from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(dtc_tuned, x_test, y_test)
plt.show()


# Model 2: RF (Random Forest)

#Grid search (done already) and Train RF
from sklearn.ensemble import RandomForestClassifier
rfc_tuned = RandomForestClassifier(criterion = 'gini', max_depth = 4, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators = 50, random_state = 42)
rfc_tuned.fit(x_train, y_train)

rfc_pred= rfc_tuned.predict(x_test)
print(classification_report(y_test, rfc_pred))

cm = confusion_matrix(y_test, rfc_pred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, rfc_pred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(rfc_tuned, x_test, y_test)
plt.show()


# Model 3: KNN (K nearest Neighbors)

#KNN Model grid search (done) and training
# Create a K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
KNN_tuned = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
KNN_tuned.fit(x_train, y_train)

# Predict the output for the test data
KNNpred = KNN_tuned.predict(x_test)

print(classification_report(y_test, KNNpred))

cm = confusion_matrix(y_test, KNNpred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, KNNpred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(KNN_tuned, x_test, y_test)
plt.show()


# Model 4: SVM (Support vector machine)

#SVM gride search (done) and model training
from sklearn.svm import SVC
svc_tuned = SVC(probability=True, C= 1, gamma= 1, kernel= 'rbf')
svc_tuned.fit(x_train, y_train)

svc_pred= svc_tuned.predict(x_test)
print(classification_report(y_test, svc_pred))

cm = confusion_matrix(y_test, svc_pred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, svc_pred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(svc_tuned, x_test, y_test)
plt.show()


# Model 5: NB (Nieve bays)

#NB grid search (done) and model training
from sklearn.naive_bayes import GaussianNB
nb_tuned = GaussianNB(var_smoothing= 0.008111308307896872)
nb_tuned.fit(x_train, y_train)

Nbpred = nb_tuned.predict(x_test)

print(classification_report(y_test, Nbpred))

cm = confusion_matrix(y_test, Nbpred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, Nbpred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(nb_tuned, x_test, y_test)
plt.show()


# Model 6: LR (Logistic regression)

#LR grid search (done) and model training
from sklearn.linear_model import LogisticRegression
lr_tuned = LogisticRegression(C=10, penalty = 'l2')
lr_tuned.fit(x_train, y_train)

# Predict the output for the test data
LRpred = lr_tuned.predict(x_test)
print(classification_report(y_test, LRpred))

cm = confusion_matrix(y_test, LRpred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, LRpred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(lr_tuned, x_test, y_test)
plt.show()


# Model 7: GB (Gradiant boosting)

#GB grid search (done) and model training
from sklearn.ensemble import GradientBoostingClassifier
gb_tuned = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=5)
gb_tuned.fit(x_train, y_train)

# Make predictions on the test set
GBpred = gb_tuned.predict(x_test)

# Print the classification report
print(classification_report(y_test, GBpred))

cm = confusion_matrix(y_test, GBpred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, GBpred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(gb_tuned, x_test, y_test)
plt.show()


# Model 8: XGB (eXtreme gradiant boosting)

#XGB grid search(done) and model training
get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
xgb_tuned=XGBClassifier(learning_rate= 0.01, max_depth= 3, n_estimators= 200)
xgb_tuned.fit(x_train, y_train)

XGBpred= xgb_tuned.predict(x_test)
print(classification_report(y_test, XGBpred))

cm = confusion_matrix(y_test, XGBpred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, XGBpred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(xgb_tuned, x_test, y_test)
plt.show()


# Model 9: LAssoR (Lasso regression)

#LassoR grid search (done) and model training
from sklearn.linear_model import LassoCV
lasso_tuned = LassoCV(cv=5, eps=0.01, n_alphas= 50)
lasso_tuned.fit(x_train, y_train)

# Make predictions on the test set
LAsopred = lasso_tuned.predict(x_test).round()
print(classification_report(y_test, LAsopred))

cm = confusion_matrix(y_test, LAsopred)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, LAsopred)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds_rand = roc_curve(y_test, LAsopred)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.show()

# Model 10: Ensemble Learning 

# Import required libraries
from sklearn.ensemble import VotingClassifier

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('svm',svc_tuned ), ('gb', gb_tuned), ('xgb', xgb_tuned)], voting='soft')

# Fit the ensemble model to the data
ensemble_model.fit(x_train, y_train)

# Predict the target variable
y_pred_ensemble = ensemble_model.predict(x_train)

print(classification_report(y_train, y_pred_ensemble))

from sklearn.metrics import RocCurveDisplay
rfc_disp = RocCurveDisplay.from_estimator(ensemble_model, x_test, y_test)
plt.show()

# Ensure the predictions have the same number of samples
min_samples = min(len(svc_pred), len(GBpred), len(XGBpred))
svm_predictions = svc_pred[:min_samples]
xg_predictions = GBpred[:min_samples]
xgb_predictions = XGBpred[:min_samples]
y_test = y_test[:min_samples]

ensemble_predictions = []
for i in range(len(y_test)):
    # Majority voting
    votes = svc_pred[i] + GBpred[i] + XGBpred[i]
    ensemble_predictions.append(1 if votes >= 2 else 0)

cm = confusion_matrix(y_test, ensemble_predictions)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, ensemble_predictions)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

#So, the ensemble prediction provided the best model.
#plot all the roc curves in one plot
# Compute the ROC curve and AUC for each model
dt_probablities = dtc_tuned.predict_proba(x_test)[:, 1]
rf_probablities = rfc_tuned.predict_proba(x_test)[:, 1]
knn_probablities = KNN_tuned.predict_proba(x_test)[:, 1]
nb_probablities = nb_tuned.predict_proba(x_test)[:, 1]
lr_probablities = lr_tuned.predict_proba(x_test)[:, 1]
svm_probabilities = svc_tuned.predict_proba(x_test)[:, 1]
xg_probabilities = gb_tuned.predict_proba(x_test)[:, 1]
xgb_probabilities = xgb_tuned.predict_proba(x_test)[:, 1]
ensemble_probablities = ensemble_model.predict_proba(x_test)[:, 1]
#lasso_probablities = lasso_tuned.predict_proba(x_test) [:, 1]

y_pred = svc_tuned.predict(x_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Compute the ROC curve and AUC for each model
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probabilities)
svm_auc = auc(svm_fpr, svm_tpr)

dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probablities )
dt_auc = auc(dt_fpr, dt_tpr)

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probablities)
rf_auc = auc(rf_fpr, rf_tpr)

knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probablities)
knn_auc = auc(knn_fpr, knn_tpr)

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probablities)
nb_auc = auc(nb_fpr, nb_tpr)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probablities)
lr_auc = auc(lr_fpr, lr_tpr)

xg_fpr, xg_tpr, _ = roc_curve(y_test, xg_probabilities)
xg_auc = auc(xg_fpr, xg_tpr)

xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probabilities)
xgb_auc = auc(xgb_fpr, xgb_tpr)

ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test, ensemble_probablities)
ensemble_auc = auc(ensemble_fpr, ensemble_tpr)

# Plot the ROC curves
plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label='calibrated_classifier (AUC = {:.2f})'.format(svm_auc))
plt.plot(xg_fpr, xg_tpr, label='Gradient Boosting (AUC = {:.2f})'.format(xg_auc))
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost (AUC = {:.2f})'.format(xgb_auc))
plt.plot(dt_fpr, dt_tpr, label='Decision tree classifier (AUC = {:.2f})'.format(dt_auc))
plt.plot(rf_fpr, rf_tpr, label='Random forest classifier (AUC = {:.2f})'.format(rf_auc))
plt.plot(knn_fpr, knn_tpr, label='K nearest neighbor (AUC = {:.2f})'.format(knn_auc))
plt.plot(nb_fpr, nb_tpr, label='Naive bayes (AUC = {:.2f})'.format(nb_auc))
plt.plot(lr_fpr, lr_tpr, label='Logistic regression (AUC = {:.2f})'.format(lr_auc))
plt.plot(ensemble_fpr, ensemble_tpr, label='Ensemble model (AUC = {:.2f})'.format(ensemble_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# List of model names and their corresponding accuracy scores
model_names = ['dtc_tuned', 'rfc_tuned', 'KNN_tuned', 'svc_tuned', 'nb_tuned', 'lr_tuned', 'gb_tuned', 'xgb_tuned', 'lasso_tuned', 'ensemble_model']
accuracy_scores = [0.79, 0.78, 0.76, 0.81, 0.74, 0.80, 0.81, 0.81, 0.79, 0.86]

# Set colors for the bars
colors = ['lightyellow']

# Create a horizontal bar plot
plt.figure(figsize=(8, 6))
plt.barh(model_names, accuracy_scores, color=colors)
# Add labels and title
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.title('Machine Learning Models Comparison')

# Add data values to the bars
for i, score in enumerate(accuracy_scores):
    plt.text(score, i, str(score), ha='left', va='center')
# Add a cool background gradient
gradient = np.linspace(0, 1, 10).reshape(1, 10).T
plt.imshow([gradient], aspect='auto', cmap='winter', extent=(0, 1, -1, len(model_names)))
# Customize the appearance
plt.grid(False)
plt.box(False)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Add labels and title
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.title('Machine Learning Models Comparison')

# Add data values to the bars
for i, score in enumerate(accuracy_scores):
    plt.text(score, i, str(score), ha='left', va='center')
# Add a cool background gradient
gradient = np.linspace(0, 1, 10).reshape(1, 10).T
plt.imshow([gradient], aspect='auto', cmap='viridis', extent=(0, 1, -1, len(model_names)))

# Customize the appearance
plt.grid(False)
plt.box(False)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# List of model names and their predicted probabilities or decision scores
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
model_names = ['dtc_tuned', 'rfc_tuned', 'KNN_tuned', 'svc_tuned', 'nb_tuned', 'lr_tuned', 'gb_tuned', 'xgb_tuned', 'ensemble_model']

# Generate random predictions for the random (no skill) model
y_pred_random = np.random.choice([0, 1], size=len(y_train_nearmiss))

# Calculate precision and recall for the random model
precision_random, recall_random, _ = precision_recall_curve(y_train_nearmiss, y_pred_random)

#plot the PR curve
models= [dtc_tuned, rfc_tuned, KNN_tuned, calibrated_classifier, nb_tuned, lr_tuned, gb_tuned, xgb_tuned, ensemble_model]
# Calculate the Random (No Skill) PR curve
positive_rate = np.sum(y_train_nearmiss) / len(y_train_nearmiss)
recall_random = np.linspace(0, 1, num=100)
precision_random = np.repeat(positive_rate, 100)

# Plot PR curves for each model
plt.figure(figsize=(8, 6))
for model_name, model in zip(model_names, models):
    # Generate model predictions or decision scores for the test set
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # Plot the PR curve for each model
    plt.plot(recall, precision, label=model_name)

# Plot the Random (No Skill) PR curve
plt.plot(recall_random, precision_random, linestyle='--', color='grey', label='Random (No Skill)')

# Add labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')

# Customize the appearance
plt.legend()
plt.grid(True)
plt.xlim([0.5, 1])
plt.ylim([0.5, 1])

# Show the plot
plt.tight_layout()
plt.show()

# Get predicted probabilities for the positive class and binary prediction based on a 0.5 treshold (class 1)
dt_probablities = (dt_probablities >= 0.5).astype(int)
rf_probablities = (rf_probablities >= 0.5).astype(int)
knn_probablities = (knn_probablities >= 0.5).astype(int)
nb_probablities = (nb_probablities >= 0.5).astype(int)
lr_probablities = (lr_probablities >= 0.5).astype(int)
svm_probabilities = (svm_probabilities >= 0.5).astype(int)
xg_probabilities = (xg_probabilities >= 0.5).astype(int)
xgb_probabilities= (xgb_probabilities >= 0.5).astype(int)
ensemble_probablities = (ensemble_probablities >= 0.5).astype(int)

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, dt_probablities)

# Plot the PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Model PR Curve')
plt.plot([0, 1], [positive_rate, positive_rate], linestyle='--', color='grey', label='Random (No Skill)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

#the cvm needs calibration
# Train the SVM model with probability=True to enable Platt scaling
from sklearn.svm import SVC
svc_new = SVC(probability=True)
svc_new.fit(x_train, y_train)

# Calibrate the classifier to obtain probability estimates
from sklearn.calibration import CalibratedClassifierCV
calibrated_classifier = CalibratedClassifierCV(svc_new, method='sigmoid', cv='prefit')
calibrated_classifier.fit(x_train, y_train)

# Get predicted probabilities for the positive class (class 1)
svm_new_pred = calibrated_classifier.predict_proba(x_test)[:, 1]

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, svm_new_pred)

# Calculate AUC-PR
auc_pr = auc(recall, precision)

# Plot the PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

