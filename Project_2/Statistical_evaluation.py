# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:35:24 2020

statistical evaluation of models

@author: cm
"""

from toolbox_02450 import jeffrey_interval
import numpy as np
from dataProcessing import *
import scipy.stats as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#%%Importing data
    
    
raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file

ytrue = y
yhat = y
yhatA = y
yhatB = np.random.randint(3, size = y.shape)


#%% Classification

# Compute accuracy
errors = np.sum(yhat != ytrue)
# accuracy: 1accuracy = total correct predictions / total predictions made * 100
accuracy = (len(X)-errors) / len(X) * 100
print('\n')
print('The accuracy of is: {0}%'.format(accuracy))
    
# with sklearn
print('\n')
accuracy_sklearn = accuracy_score(ytrue, yhat)
print('Accuary Score:')
print(accuracy_sklearn)
classification_rep = classification_report(ytrue, yhat)
print('Classification Report:')
print(classification_rep)
confusion_mat = confusion_matrix(ytrue, yhat)
print('Confusion Matrix:')
print(confusion_mat)
# acc from confmat = (tp+tn)/N with tp = true and correct predicted, tn = true and wrong predicted, N = len(X)
# this responds to Classification report: recall.


# Compute the Jeffreys interval
alpha = 0.05

print('\n')
print('Jeffrey Interval for alpha={}:'.format(alpha))
[thetahatA, CIA] = jeffrey_interval(ytrue, yhat, alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

# Compare 2 models


#%% Regression

# Compute accuracy

# Compare 2 models

# compute z with squared error.
zA = np.abs(y - yhatA ) ** 2
zB = np.abs(y - yhatB ) ** 2
print('zA: {0} zB: {1}'.format(np.mean(zA),np.mean(zB)))
# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
print('CIA: ', CIA)
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval
print('CIB: ', CIB)

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print('z:', np.mean(z), 'CI: ',(CI),'p-value: ', (p))