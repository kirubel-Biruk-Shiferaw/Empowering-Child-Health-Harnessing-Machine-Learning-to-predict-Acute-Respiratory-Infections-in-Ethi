#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[20]:


data = pd.read_excel("C:/Users/kirubel/Documents/Mule Project/Raw final.xlsx")


# In[21]:


#we drope 3 variables for very low correlation and high imbalance
data= data.drop('wasting', axis=1)
data= data.drop('water', axis=1)
data= data.drop('vitamin', axis=1)


# In[22]:


data.corr()


# In[23]:


sns.heatmap(data.corr(), cmap='inferno') #Very good correlation 


# In[24]:


corr= data.corr()

# Getting the Upper Triangle of the co-relation matrix
matrix = np.triu(np.ones_like(corr))

# using the upper triangle matrix as mask 
sns.heatmap(data.corr(), mask=matrix)


# In[25]:


X= data.drop(["outcomevar"], axis = 1)
Y= data["outcomevar"]


# In[26]:


# Train/Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[27]:


#Nearmiss undersampling
from imblearn.under_sampling import NearMiss
nearmiss = NearMiss()
X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(x_train, y_train)


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(X_train_nearmiss, y_train_nearmiss, test_size = 0.2, random_state = 42)


# In[120]:


print("x_train : ",x_train.shape)

print("x_test : ",x_test.shape)

print("y_train : ",y_train.shape)

print("y_test : ",y_test.shape)


# In[122]:


# Convert the arrays or DataFrames to DataFrames if needed
x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)


# In[129]:


# Save x_train to Excel
x_train_df.to_excel("C:/Users/kirubel/Documents/MuleProject/Code/Docker/Dockerfile/x_train.xlsx", index=False)

# Save x_test to Excel
x_test_df.to_excel("C:/Users/kirubel/Documents/MuleProject/Code/Docker/Dockerfile/x_test.xlsx", index=False)

# Save y_train to Excel
y_train_df.to_excel("C:/Users/kirubel/Documents/MuleProject/Code/Docker/Dockerfile/y_train.xlsx", index=False)

# Save y_test to Excel
y_test_df.to_excel("C:/Users/kirubel/Documents/MuleProject/Code/Docker/Dockerfile/y_test.xlsx", index=False)


# In[ ]:





# In[29]:


# Check the number of records after over sampling
from collections import Counter
print(sorted(Counter(y_train_nearmiss).items()))


# In[30]:


#Training the first model, SVM
#SVM gride search (done) and model training
from sklearn.svm import SVC
svc_tuned = SVC(probability=True, C= 1, gamma= 1, kernel= 'linear')
svc_tuned.fit(x_train, y_train)


# In[31]:


svc_pred= svc_tuned.predict(x_test)


# In[32]:


#Training the second model, GB
from sklearn.ensemble import GradientBoostingClassifier
gb_tuned = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=5)
gb_tuned.fit(x_train, y_train)

# Make predictions on the test set
GBpred = gb_tuned.predict(x_test)


# In[33]:


#Training the third model, XGB
#XGB grid search(done) and model training
from xgboost import XGBClassifier
xgb_tuned=XGBClassifier(learning_rate= 0.01, max_depth= 3, n_estimators= 200)
xgb_tuned.fit(x_train, y_train)


# In[34]:


XGBpred= xgb_tuned.predict(x_test)


# In[35]:


# Ensemling the models
# Import required libraries
from sklearn.ensemble import VotingClassifier


# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('svm',svc_tuned ), ('gb', gb_tuned), ('xgb', xgb_tuned)], voting='soft')

# Fit the ensemble model to the data
ensemble_model.fit(x_train, y_train)


# In[36]:


# Predict the target variable
y_pred_ensemble = ensemble_model.predict(x_train)


# In[37]:


from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_ensemble))


# In[38]:


# Ensure the predictions have the same number of samples
min_samples = min(len(svc_pred), len(GBpred), len(XGBpred))
svm_predictions = svc_pred[:min_samples]
xg_predictions = GBpred[:min_samples]
xgb_predictions = XGBpred[:min_samples]
y_test = y_test[:min_samples]


# In[39]:


ensemble_predictions = []
for i in range(len(y_test)):
    # Majority voting
    votes = svc_pred[i] + GBpred[i] + XGBpred[i]
    ensemble_predictions.append(1 if votes >= 2 else 0)


# In[40]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
cm = confusion_matrix(y_test, ensemble_predictions)
# calculate sensitivity/recall
sensitivity = recall_score(y_test, ensemble_predictions)

# calculate specificity
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

# print the results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)


# In[41]:


# Use learning_curve to generate the training and test accuracy scores
from sklearn.model_selection import learning_curve, train_test_split
train_sizes, train_scores, test_scores = learning_curve(ensemble_model, x_train, y_train, cv=10)

# Calculate the mean and standard deviation of the training and test accuracy scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()


# In[42]:


# Extract feature importances from the base classifiers
svm_feature_importances = np.abs(svc_tuned.coef_[0])
gb_feature_importances = gb_tuned.feature_importances_
xgb_feature_importances = xgb_tuned.feature_importances_


# In[43]:


# Calculate average feature importance across the Gradient Boosting and XGBoost classifiers
ensemble_feature_importances = (gb_feature_importances + xgb_feature_importances + svm_feature_importances) / 3


# In[44]:


# Sort feature importances in descending order
sorted_indices = np.argsort(ensemble_feature_importances)[::-1]
sorted_importances = ensemble_feature_importances[sorted_indices]
sorted_features = np.arange(X.shape[1])[sorted_indices]


# In[82]:


sorted_features


# In[67]:


ensemble_feature_importances


# In[46]:


# Create a list of feature names
feature_names = X.columns
feature_names


# In[72]:


feature_importances = pd.DataFrame(ensemble_feature_importances, index= feature_names)
feature_importances.head()


# In[48]:


features = list(feature_importances [feature_importances[0]>0].index)
features


# In[ ]:





# In[76]:


# Plot the feature importances
plt.figure( figsize=(8, 6))
plt.bar(np.arange(len(feature_names)), feature_importances[0], alpha=0.9, width=0.35, align='edge', label='Ensemble model')
plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


# In[65]:


# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.bar(range(X.shape[1]), ensemble_feature_importances, align='center')
plt.xticks(range(X.shape[1]), sorted_features)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


# In[84]:


def model_predict(x_test):
    return ensemble_model.predict(x_test)


# In[100]:



import shap
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

shap.summary_plot(shap_values, x_test)


# In[98]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.summary_plot(shap_values.data, x_test.values, feature_names = x_test.columns)


# In[94]:


x_test.shape


# In[99]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.summary_plot(shap_values.data, x_test.values, feature_names = x_test.columns, plot_type = "violin")


# In[101]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.summary_plot(shap_values, x_test)



# In[112]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.plots.beeswarm(shap_values, plot_type= "violin")


# In[114]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.summary_plot(shap_values, plot_type = "violin")


# In[ ]:


#compute shap values
explainer = shap.Explainer(model_predict, x_test)
shap_values = explainer(x_test)

# Check if shap_values is a vector, and if so, convert it to a matrix
if len(shap_values.shape) == 1:
    shap_values = shap_values.reshape((-1, x_test.shape[1]))
    
shap.summary_plot(shap_values.data, x_test.values, feature_names = x_test.columns, plot_type = "violin")


# In[155]:


import os
print(os.getcwd())


# In[154]:


os.chdir('C:/Users/kirubel/Documents/MuleProject/Code/app')


# In[156]:


#serializing the model (a serialized model file refers to a file that stores the trained model object in a serialized format, allowing it to be saved and loaded for later use. It typically contains all the necessary information to recreate the trained model, including the model architecture, weights, and any additional parameters. )
import pickle


# In[157]:


model_path = 'C:/Users/kirubel/Documents/MuleProject/Code/app/trained_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(ensemble_model, file)


# In[158]:


# Save the model to a file
data = {"model": ensemble_model, "Important features": features}

with open('trained_ensemble_model.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[ ]:





# In[141]:


from platform import python_version

print(python_version())


# In[159]:




# Load the saved model from the file
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[ ]:




