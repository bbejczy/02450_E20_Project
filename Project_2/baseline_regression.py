# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:45:53 2020

Baseline model for linear regression

@author: cm
"""
import numpy as np
from sklearn import model_selection
from dataProcessing import *


def baseline_regression_estimator(y):
    y_est = np.empty(y.shape)
    y_est[:] = np.mean(y)
       
    return y_est

def yest_to_yhat(y_est, y_test, yhat, ytrue):
    
    yhat = np.append(yhat,y_est)
    ytrue = np.append(ytrue,y_test)
    
    
    return yhat, ytrue
    
def baseline_regression_model(y_test):
                
        # Put here the models:
            
        y_test_est = baseline_regression_estimator(y_test)
        
        # yhat, ytrue = yest_to_yhat(y_test_est, y_test, yhat, ytrue)
        
        
        Error_test = np.sum((y_test - y_test_est) ** 2)/len(y_test)
        
        return Error_test, y_test_est
    
       

def baseline_regression(X,y,K_inner,yhat,ytrue):
    
    CV = model_selection.KFold(K_inner, shuffle=False)
    
    #init Variables
    Error_test = np.empty((K_inner,1))
     
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        Error_test, y_test_est = baseline_regression_model(y_test) 
       
        yhat, ytrue = yest_to_yhat(y_test_est, y_test, yhat, ytrue)
        
        # end of for-loop
        k+=1
        
    Error_test_loop = np.mean(Error_test)
    
    return Error_test_loop,yhat, ytrue

#%%

if __name__ == '__main__':
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #randomise the order
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    
    X = X[index,:]
    y = y[index]
    
    #%% Pre-processing the data
    
    cent_data = centerData(X)
    
    X = standardizeData(cent_data) #normalized data
    
    #%% K-Fold-Validation
    
    K_inner = 10
    yhat = []
    ytrue = []
    
    Error_test, yhat, ytrue = baseline_regression(X,y,K_inner, yhat, ytrue)
    
    print("Error_test^2: ", Error_test)
    
    
    


