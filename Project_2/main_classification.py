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
import Classification_Logistic_Regression as LogReg

import ANN_multiClass as ANN


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
    
    # Standardise Data
    X = DP.standardizeData(X)
    
    #%% Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    
    # Initialize variables
    modelNames = ['Baseline', 'Decission Tree', 'Logistic Regression', 'ANN Classification']

    Error_test = np.empty((K,len(modelNames)))
    Error_train = np.empty((K,len(modelNames)))
    
    yhat_BLC = []
    ytrue_BLC = []

    yhat_tree = []
    ytrue_tree = []
    
    yhat_LR = []
    ytrue_LR = []
    
    ytrue = []
    yhat = []

    outer_h = []
    yhat_ANNc = []
    
    opt_lambda = np.empty((K,1))
    h =  np.empty((K,1))
    tc = np.empty((K,1))
    
    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        # print('\n')
        # print("CV fold ", k+1)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10 
        
        # Baseline
        Error_test[k,0], yhat_temp, ytrue_temp = BLC.baseline_classification(X_train,y_train,internal_cross_validation, yhat, ytrue)
        yhat_BLC = np.append(yhat_BLC,yhat_temp)
        ytrue_BLC = np.append(ytrue_BLC,ytrue_temp)
        # print('Baseline')
        # print("Error_test^2: {:.2f}%".format(Error_test[k,0]*100))
        # print('bl',yhat_BLC.shape)
        
        # Decission Tree
       
        tc[k] = dtree.classifier_complexity(X_train, y_train, attributeNames, classNames)
        Error_train[k,1],Error_test[k,1],yhat_temp,ytrue_temp=dtree.classifier_model(X_train, y_train, internal_cross_validation, yhat, ytrue,tc[k])
        yhat_tree = np.append(yhat_tree,yhat_temp)
        ytrue_tree = np.append(ytrue_tree,ytrue_temp)
        # print('Decission Tree:')
        # print("Error_test^2: ", Error_test[k,1], 'With tree depth:', tc)
        # print('tree',yhat_tree.shape)
        
        # Logistic Regression
        Error_train[k,2],Error_test[k,2],opt_lambda[k],yhat_temp,ytrue_temp = LogReg.Logistic_Regression(X_train, y_train, internal_cross_validation, yhat, ytrue)
        yhat_LR = np.append(yhat_LR,yhat_temp)
        ytrue_LR = np.append(ytrue_LR,ytrue_temp)
        # print('Logistic Regression:')
        # print("Error_test^2: ", Error_test[k,2], 'With lambda:', opt_lambda[k])
        # print('LR',yhat_LR.shape)
        # end of for-loop
        
        # ANN Classification
        # Error_test[k,3], h_temp, yhat_temp = ANN.ANN_multiClass(X_train,y_train,internal_cross_validation,C)
        # yhat_ANNc = np.append(yhat_ANNc, yhat_temp)
        # outer_h = np.append(outer_h, h_temp)
        
        k+=1
        
    
    # Calculate the mean errors of all and the mean values of the depths
    Total_Error_train = np.mean(Error_train, axis=0)    
    Total_Error_test = np.mean(Error_test, axis=0)
    Total_opt_lambda = np.mean(opt_lambda)
    Total_tc = np.mean(tc)
    
#%% Statistical evaluation


 # since all ytrue are the same we take the baseline as the "real" ytrue
    ytrue = ytrue_BLC
    
    # # Compute accuracy
    print(modelNames[0])
    stats.evaluate_1_classifier(ytrue,yhat_BLC)
    print('\n')
    print(modelNames[1])
    stats.evaluate_1_classifier(ytrue,yhat_LR)
    print('\n')
    print(modelNames[2])
    stats.evaluate_1_classifier(ytrue,yhat_tree)
    
    # # Compare 2 models
    
    print('\n')
    print(modelNames[0], 'vs.', modelNames[1])
    stats.compare_2_classifiers(ytrue,yhat_BLC,yhat_LR)
    print('\n')
    print(modelNames[0], 'vs.', modelNames[2])
    stats.compare_2_classifiers(ytrue,yhat_BLC,yhat_tree)
    print('\n')
    print(modelNames[1], 'vs.', modelNames[2])
    stats.compare_2_classifiers(ytrue,yhat_LR,yhat_tree)
    
