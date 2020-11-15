"""
Created on Sat Nov 14 11:25:00 2020

@author: mateusz
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import pandas as pd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

import sklearn.linear_model as lm

# from dataProcessing import *
import dataProcessing as DP




#%%Function


def Logistic_Regression(X,y,cvf,yhat,ytrue):

    #create random order
   # index = np.arange(0,len(X))
    #np.random.shuffle(index)
    
    #X = X[index,:]
    #y = y[index]
    
    # Standardize the training and set set based on training set mean and std
    # mu = np.mean(X, 0)
    # sigma = np.std(X, 0)
    
    # X = (X - mu) / sigma
    
    
    # Create crossvalidation 
    
    CV = model_selection.KFold(cvf,shuffle=False)
    
    
    
    for train_index, test_index in CV.split(X,y):
                
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        
        # Fit regularized logistic regression model to training data 
        lambda_interval = np.logspace(-8, 8, 20)
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        coefficient_norm = np.zeros(len(lambda_interval))
        
        y_train_est = np.empty((len(lambda_interval),len(y_train))) 
        y_test_est = np.empty((len(lambda_interval),len(y_test)))
        
        
        f = 0
        
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C= 1/lambda_interval[k] )
            
            mdl.fit(X_train, y_train)
        
            y_train_est[k,:] = mdl.predict(X_train).T
            y_test_est[k,:] = mdl.predict(X_test).T
            
            train_error_rate[k] = np.sum(y_train_est[k,:] != y_train) / len(y_train)
            test_error_rate[k] = np.sum(y_test_est[k,:] != y_test) / len(y_test)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
            
        
        min_error = np.min(test_error_rate)
        
        Error_test = test_error_rate.mean()
        Error_train = train_error_rate.mean()
        
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        
        yhat = np.append(yhat,y_test_est[opt_lambda_idx,:])
        # print('yhat',yhat.shape)
        ytrue = np.append(ytrue,y_test)

        
        f+=1
        
        # plt.figure(figsize=(8,8))
        # #plt.plot(np.log10(lambda_interval), train_error_rate*100)
        # #plt.plot(np.log10(lambda_interval), test_error_rate*100)
        # #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
        # plt.semilogx(lambda_interval, train_error_rate*100,'x-')
        # plt.semilogx(lambda_interval, test_error_rate*100,'x-')
        # plt.semilogx(opt_lambda, min_error*100, 'o')
        # plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
        # plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        # plt.ylabel('Error rate (%)')
        # plt.title('Classification error')
        # plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        # plt.ylim([0, 10])
        # plt.grid()
        # plt.show()    
        
        # plt.figure(figsize=(8,8))
        # plt.semilogx(lambda_interval, coefficient_norm,'k')
        # plt.ylabel('L2 Norm')
        # plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        # plt.title('Parameter vector L2 norm')
        # plt.grid()
        # plt.show()
        
    return Error_train, Error_test, opt_lambda, yhat, ytrue


# plt.figure(figsize=(8,8))
# #plt.plot(np.log10(lambda_interval), train_error_rate*100)
# #plt.plot(np.log10(lambda_interval), test_error_rate*100)
# #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
# plt.semilogx(lambda_interval, train_error_rate*100,'x-')
# plt.semilogx(lambda_interval, test_error_rate*100,'x-')
# plt.semilogx(opt_lambda, min_error*100, 'o')
# plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.ylabel('Error rate (%)')
# plt.title('Classification error')
# plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
# plt.ylim([0, 10])
# plt.grid()
# plt.show()    

# plt.figure(figsize=(8,8))
# plt.semilogx(lambda_interval, coefficient_norm,'k')
# plt.ylabel('L2 Norm')
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.title('Parameter vector L2 norm')
# plt.grid()
# plt.show()  

#%%import data
if __name__ == '__main__':
    
    raw_data,X,y,C,N,M, cols,filename,attributeNames,classNames = DP.getData() #importing the raw data from the file
    # attributeNames = [names for names in attributeNames if names != 'ID'  ]
    
    #randomise the order
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    
    X = X[index,:]
    y = y[index]
    
    # Standardise Data
    X = DP.standardizeData(X)
    
    #%%Classification 
    
    cvf = 10
    yhat = []
    ytrue = []
    
    Error_train, Error_test, opt_lambda, yhat, ytrue = Logistic_Regression(X, y, cvf, yhat,ytrue)
    
    print('lambda:', opt_lambda)
    print('Error test:',Error_test)
    print('Error train:',Error_train)


    # plt.figure(figsize=(8,8))
    # #plt.plot(np.log10(lambda_interval), train_error_rate*100)
    # #plt.plot(np.log10(lambda_interval), test_error_rate*100)
    # #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    # plt.semilogx(lambda_interval, train_error_rate*100,'x-')
    # plt.semilogx(lambda_interval, test_error_rate*100,'x-')
    # plt.semilogx(opt_lambda, min_error*100, 'o')
    # plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
    # plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    # plt.ylabel('Error rate (%)')
    # plt.title('Classification error')
    # plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
    # plt.ylim([0, 10])
    # plt.grid()
    # plt.show()    
    
    # plt.figure(figsize=(8,8))
    # plt.semilogx(lambda_interval, coefficient_norm,'k')
    # plt.ylabel('L2 Norm')
    # plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    # plt.title('Parameter vector L2 norm')
    # plt.grid()
    # plt.show()  