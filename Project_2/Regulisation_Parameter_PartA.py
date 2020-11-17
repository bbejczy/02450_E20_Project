# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:15:22 2020

Regulisation Parameter

@author: cm
"""


from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

import sklearn.linear_model as lm

from dataProcessing import *
import dataProcessing as dP


#%% Import data 


raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
attributeNames = [names for names in attributeNames if names != 'ID'  ]

regression_attribute = 10
y = X[:,regression_attribute]
X_cols = list(range(0,regression_attribute)) + list(range(regression_attribute+1,len(attributeNames)))



print('Regression on Attribute:',attributeNames[regression_attribute])
X_without = X[:,X_cols]
X = standardizeData(X_without)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
# # attributeNames = [u'Offset']+attributeNames
# M = M+1
M = len(X[0])


#%% Regression Parameter

cvf = 10
lambdas = np.power(10.,range(-8,8))
coef = np.empty((cvf,len(X[1])))

CV = model_selection.KFold(cvf, shuffle=True)
# M = X.shape[1]
w = np.empty((M,cvf,len(lambdas)))
train_error = np.empty((cvf,len(lambdas)))
test_error = np.empty((cvf,len(lambdas)))

f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
        
    # Standardize the training and set set based on training set moments
    # mu = np.mean(X_train[:, 1:], 0)
    # sigma = np.std(X_train[:, 1:], 0)
    
    # X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    # X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
               
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))

Error_train_opt = np.min(train_err_vs_lambda)
Error_test_opt = np.min(test_err_vs_lambda)

print('\n')
print('Regulistation Parameter:')
print('Error train:', Error_train_opt)
print('Error test:', Error_test_opt)

# return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda

# Plot

figure(1,figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
legend([attributeNames[i] for i in X_cols], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

#%% Normal linear Regression without Regulisation parameter
cvf = 10
CV = model_selection.KFold(cvf, shuffle=True)
f = 0
error_train_noReg = np.empty(cvf)
error_test_noReg = np.empty(cvf)

for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    m = lm.LinearRegression().fit(X_train, y_train)
    y_train_est = m.predict(X_train)
    y_test_est = m.predict(X_test)
    error_train_noReg[f] = (np.square(y_train-y_train_est)).sum()/len(y_train)
    error_test_noReg[f] = np.sum(np.square(y_test-y_test_est))/len(y_test)

    f=f+1
    
Error_train_noReg = np.mean(error_train_noReg)
Error_test_noReg = np.mean(error_test_noReg)

print('\n')
print('Normal Regression without Regulisation parameter')
print('Error train:', Error_train_noReg)
print('Error test:', Error_test_noReg)