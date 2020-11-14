# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""
import dataProcessing as DP
# from dataVisualization import *
# from PCA_analysis import * 
import baseline_classification as BLC
import Statistical_evaluation as stats

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, jeffrey_interval, mcnemar

import Decission_Tree as dtree

# from ANN import *

# error_ANN = []

# =============================================================================
#     MAIN
# =============================================================================
if __name__ == '__main__':
    #%%Importing data
    
    
    raw_data,X,y,C,N,M, cols,filename,attributeNames,classNames = DP.getData() #importing the raw data from the file

     #randomise the order
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    
    X = X[index,:]
    y = y[index]
    
    # Standardise Data
    X = DP.standardizeData(X)
    
    #%% Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 5
    CV = model_selection.KFold(K, shuffle=True)
    
    # Initialize variables
    modelNames = ['Baseline', 'Decission Tree']

    Error_test = np.empty((K,len(modelNames)))
    Error_train = np.empty((K,len(modelNames)))
    
    yhat_BLC = []
    ytrue_BLC = []

    yhat_tree = []
    ytrue_tree = []

    ytrue = []  
    yhat = []
    
    opt_lambda = np.empty((K,1))
    h =  np.empty((K,1))
    
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        print("CV fold ", k+1)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10 
        
        # Baseline
        Error_test[k,0], yhat_temp, ytrue_temp = BLC.baseline_classification(X_train,y_train,internal_cross_validation, yhat, ytrue)
        yhat_BLC = np.append(yhat_BLC,yhat_temp)
        ytrue_BLC = np.append(ytrue_BLC,ytrue_temp)
        print('\n')
        print('Baseline')
        print("Error_test^2: {:.2f}%".format(Error_test[k,0]*100))
        
        # Decission Tree
       
        tc = dtree.classifier_complexity(X_train, y_train, attributeNames, classNames)
        Error_train[k,1],Error_test[k,1],yhat_temp,ytrue_temp=dtree.classifier_model(X_train, y_train, K, yhat, ytrue,tc)
        yhat_tree = np.append(yhat_tree,yhat_temp)
        ytrue_tree = np.append(ytrue_tree,ytrue_temp)
        print('\n')
        print('Decission Tree:')
        print("Error_test^2: ", Error_test[k,1], 'With tree depth:', tc)
        
        # end of for-loop
        k+=1


#%% Statistical evaluation


# # Just an example until the all models are finished
# yhatA = yhat
# yhatB = np.random.randint(3, size = ytrue.shape)

# # evaluation of 1 model
# stat.evaluate_1_classifier(ytrue,yhat)

# # Compare 2 models
# stat.compare_2_classifiers(ytrue, yhatA, yhatB)
    
