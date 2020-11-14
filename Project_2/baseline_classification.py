# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:45:53 2020

Baseline model for Classification

@author: cm
"""
import numpy as np
from sklearn import model_selection
from dataProcessing import *



def model_baseline_classification(y):
    y_est = np.empty(y.shape)
    # find largest class
    unique, unique_count = np.unique(y, return_counts=True)
    
    index = np.argmax(unique_count)
    
    # estimate the class to be of the largest
    y_est[:] = unique[index]
    
    return y_est

def yest_to_yhat(y_est, y_test, yhat, ytrue):
    
    yhat = np.append(yhat,y_est)
    ytrue = np.append(ytrue,y_test)
    
    
    return yhat, ytrue

def baseline_classification(X,y,K_inner,yhat,ytrue):
    """
    simple baseline model for classification with K-fold cross-validation

    Parameters
    ----------
    X : array
        Data matrix.
    y : array of Nx1
        Class vector
    K_inner : int
        Number of inner K-folds.

    Returns
    -------
    Error_test_loop : int
        Squared test error.

    """
    
    CV = model_selection.KFold(K_inner, shuffle=False)
    
    #init Variables
    Error_test = np.empty((K_inner,1))
    
             
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        # X_train = X[train_index]
        # y_train = y[train_index]
        # X_test = X[test_index]
        y_test = y[test_index]
        
        # Put here the models:
        y_test_est = model_baseline_classification(y_test)
        
        yhat, ytrue = yest_to_yhat(y_test_est, y_test, yhat, ytrue)        
        
        Error_test[k] = np.sum(y_test_est != y_test)/len(y_test)
        
        # end of for-loop
        k+=1
        
    Error_test_loop = np.mean(Error_test)
    
    return Error_test_loop, yhat, ytrue

#%%

if __name__ == '__main__':
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #%% Pre-processing the data
    
    cent_data = centerData(X)
    
    X = standardizeData(cent_data) #normalized data
    
    #%% K-Fold-Validation
    
    K_inner = 10
    yhat = []
    ytrue = []
    
    Error_test, yhat, ytrue = baseline_classification(X,y,K_inner, yhat, ytrue)
    
    print("Error_test^2: {0}%".format(Error_test*100))
    
    
    


