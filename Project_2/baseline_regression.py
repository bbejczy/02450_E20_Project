# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:45:53 2020

Baseline model for linear regression

@author: cm
"""
import numpy as np
from sklearn import model_selection
from dataProcessing import *


def model_baseline_regression(y):
    y_est = np.empty(y.shape)
    y_est[:] = np.mean(y)
       
    return y_est    

def baseline_regression(X,y,K_inner):
    
    CV = model_selection.KFold(K_inner, shuffle=True)
    
    #init Variables
    Error_test = np.empty((K_inner,1))
     
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Put here the models:
        y_train_est = model_baseline_regression(y_train)
        
        Error_test[k] = np.sum((y_train - y_train_est) ** 2)/len(y_train)
        
        # end of for-loop
        k+=1
        
    Error_test_loop = np.mean(Error_test)
    
    return Error_test_loop

#%%

if __name__ == '__main__':
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #%% Pre-processing the data
    
    cent_data = centerData(X)
    
    X = standardizeData(cent_data) #normalized data
    
    #%% K-Fold-Validation
    
    K_inner = 10
    
    Error_test = baseline_regression(X,y,K_inner)
    
    print("Error_test^2: ", Error_test)
    
    
    


