# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""

import dataProcessing as DP
# from dataVisualization import *
# from PCA_analysis import * 
import baseline_regression as BLR
import Decission_Tree as dtree
# from ANN_regression import *
import Regulisation_Parameter as RG
import  Statistical_evaluation as stats
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)


import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer



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
    
    # Select attribute we want to predict
    regression_attribute = 10
    print('Regression on Attribute:',attributeNames[regression_attribute])

    y = X[:,regression_attribute]
    X_cols = list(range(0,regression_attribute)) + list(range(regression_attribute+1,len(attributeNames)))
    # attributes without the classification one
    attributeNames = [attributeNames[i] for i in X_cols]
    X_without = X[:,X_cols]
    
    
    # Standardise Data
    X = DP.standardizeData(X_without)
    
    
    # Add OneHotEncoding of classes
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
    raw_data = np.array(columnTransformer.fit_transform(raw_data), dtype = np.float)
    z = raw_data[:,[0,1,2]]
    X = np.concatenate((z,X),1)
    attributeNames = [u'Cultivar1',u'Cultivar2',u'Cultivar2']+attributeNames
   
    #%% Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    
    # Initialize variables
    modelNames = ['Baseline', 'Linear Regression with lambda', 'Decission Tree']
    
    Error_test = np.empty((K,len(modelNames)))
    Error_train = np.empty((K,len(modelNames)))
    
    yhat_temp = []
    ytrue_temp = []
    
    yhat_BLR = []
    ytrue_BLR = []

    yhat_tree = []
    ytrue_tree = []

    ytrue = []  
    yhat = []
    
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
        
        # Baseline
        Error_test[k,0], yhat_temp, ytrue_temp = BLR.baseline_regression(X_train,y_train,internal_cross_validation, yhat, ytrue)
        yhat_BLR = np.append(yhat_BLR,yhat_temp)
        ytrue_BLR = np.append(ytrue_BLR,ytrue_temp)
        print('\n')
        print('Baseline Regression')
        print("Error_test^2: ", Error_test[k,0])
        
        # Regression with Regulisation Parameter lambda
        Error_test[k,1], Error_train[k,1], opt_lambda, yhat_temp,ytrue_temp =  RG.Linear_Regression(X_train,y_train,internal_cross_validation, yhat, ytrue)
        yhat_LRR = np.append(yhat_BLR,yhat_temp)
        ytrue_LRR = np.append(ytrue_BLR,ytrue_temp)
        print('\n')
        print('Linear Regression with Regulisation Parameter')
        print("Error_test^2: ", Error_test[k,1], 'With lambda opt:', opt_lambda)
        
        # Decission Tree 
        tc = dtree.regressor_complexity(X_train,y_train, attributeNames, classNames)
        Error_train[k,2],Error_test[k,2],yhat_temp,ytrue_temp = dtree.regressor_model(X_train,y_train,K,yhat,ytrue,tc)
        yhat_tree = np.append(yhat_tree,yhat_temp)
        ytrue_tree = np.append(ytrue_tree,ytrue_temp)
        print('\n')
        print('Decission Tree:')
        print("Error_test^2: ", Error_test[k,2], 'With tree depth:', tc)
       
        
        # end of for-loop
        k+=1
        
#%% Statistical analysis

# Just an example until the all models are finished
# yhatA = yhat
# yhatB = np.random.randint(3, size = ytrue.shape)

    
# # Compute accuracy
# stats.evaluate_1_regression(ytrue,yhat)


# # Compare 2 models
# stats.compare_2_regressions(ytrue,yhatA,yhatB)

