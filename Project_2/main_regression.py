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
import Regulisation_Parameter as RG
import ANN_regression as ANN
import  Statistical_evaluation as stats
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)


import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer

#supress warnings from tensorflow
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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
    modelNames = ['Baseline', 'Linear Regression with lambda', 'Decission Tree','ANN']
    
    Error_test = np.zeros((K,len(modelNames)))
    Error_train = np.zeros((K,len(modelNames)))
    
    yhat_temp = []
    ytrue_temp = []
    
    yhat_BLR = []
    ytrue_BLR = []
    
    yhat_LRR = []
    ytrue_LRR = []
    

    yhat_tree = []
    ytrue_tree = []

    ytrue = []  
    yhat = []
    
    yhat_ANN = []
    
    opt_lambda = np.empty((K,1))
    tc = np.empty((K,1))
    h =  np.empty((K,1))
    
    k=0
    for train_index, test_index in CV.split(X,y):
        print('\n')
        print("CV fold ", k+1)
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
        # print('\n')
        # print('Baseline Regression')
        # print("Error_test^2: ", Error_test[k,0])
        
        # Regression with Regulisation Parameter lambda
        Error_test[k,1], Error_train[k,1], opt_lambda[k], yhat_temp,ytrue_temp =  RG.Linear_Regression(X_train,y_train,internal_cross_validation, yhat, ytrue)
        yhat_LRR = np.append(yhat_LRR,yhat_temp)
        ytrue_LRR = np.append(ytrue_LRR,ytrue_temp)
        # print('\n')
        # print('Linear Regression with Regulisation Parameter')
        # print("Error_test^2: ", Error_test[k,1], 'With lambda opt:', opt_lambda[k])
        
        # Decission Tree 
        tc[k] = dtree.regressor_complexity(X_train,y_train, attributeNames, classNames)
        Error_train[k,2],Error_test[k,2],yhat_temp,ytrue_temp = dtree.regressor_model(X_train,y_train,K,yhat,ytrue,tc[k])
        yhat_tree = np.append(yhat_tree,yhat_temp)
        ytrue_tree = np.append(ytrue_tree,ytrue_temp)
        # print('\n')
        # print('Decission Tree:')
        # print("Error_test^2: ", Error_test[k,2], 'With tree depth:', tc[k])
       
        # ANN Regression
        Error_test[k, 3], h[k], yhat_temp = ANN.ANN_reg(X_train, y_train, M, attributeNames, classNames, K) 
        yhat_ANN = np.append(yhat_ANN, yhat_temp)
        
        # end of for-loop
        k+=1
 
    # Calculate the mean errors of all and the mean values of the depths
    Total_Error_train = np.mean(Error_train, axis=0)    
    Total_Error_test = np.mean(Error_test, axis=0)
    Total_opt_lambda = np.mean(opt_lambda)
    Total_tc = np.mean(tc)
    
    
        
#%% Statistical analysis

    # since all ytrue are the same we take the baseline as the "real" ytrue
ytrue = ytrue_BLR

# # Compute accuracy
print(modelNames[0])
stats.evaluate_1_regression(ytrue,yhat_BLR)
print('\n')
print(modelNames[1])
stats.evaluate_1_regression(ytrue,yhat_LRR)
print('\n')
print(modelNames[2])
stats.evaluate_1_regression(ytrue,yhat_tree)
print('\n')
print(modelNames[3])
stats.evaluate_1_regression(ytrue,yhat_ANN)


# # Compare 2 models
print('\n')
print(modelNames[0], 'vs.', modelNames[1])
stats.compare_2_regressions(ytrue,yhat_BLR,yhat_LRR)
print('\n')
print(modelNames[0], 'vs.', modelNames[2])
stats.compare_2_regressions(ytrue,yhat_BLR,yhat_tree)
print('\n')
print(modelNames[1], 'vs.', modelNames[2])
stats.compare_2_regressions(ytrue,yhat_LRR,yhat_tree)

print('\n')
print(modelNames[0], 'vs.', modelNames[3])
stats.compare_2_regressions(ytrue,yhat_BLR,yhat_ANN)

print('\n')
print(modelNames[1], 'vs.', modelNames[3])
stats.compare_2_regressions(ytrue,yhat_LRR,yhat_ANN)
    
    