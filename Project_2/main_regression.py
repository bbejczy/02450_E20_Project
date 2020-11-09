# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""

from dataProcessing import *
from dataVisualization import *
from PCA_analysis import * 
from baseline_regression import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate


# =============================================================================
#     MAIN
# =============================================================================
if __name__ == '__main__':
    #%%Importing data
    
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #%% Pre-processing the data
    
    cent_data = centerData(X)
    
    X = standardizeData(cent_data) #normalized data
   
    #%% Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    
    # Initialize variables

    Error_test = np.empty((K,1))
    opt_lambda = np.empty((K,1))
    h =  np.empty((K,1))
    
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10 
        
        # Put here the models:

    
        Error_test = baseline_regression(X_train,y_train,internal_cross_validation)
    
        print("Error_test^2: ", Error_test)
        
        
        # end of for-loop
        k+=1