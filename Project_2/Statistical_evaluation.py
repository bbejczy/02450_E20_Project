# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:35:24 2020

statistical evaluation of models

@author: cm
"""

from toolbox_02450 import jeffrey_interval, mcnemar
import numpy as np
from dataProcessing import *
import scipy.stats as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score

def accuracy_classificaion(ytrue, yhat):    
    # Compute accuracy
    errors = np.sum(yhat != ytrue)
    # accuracy: 1accuracy = total correct predictions / total predictions made * 100
    accuracy = (len(ytrue)-errors) / len(ytrue) * 100
    # print('\n')
    print('The accuracy of is: {0}%'.format(accuracy))
        
    # with sklearn
    # print('\n')
    accuracy_sklearn = accuracy_score(ytrue, yhat)
    print('Accuary Score:')
    print(accuracy_sklearn)
    classification_rep = classification_report(ytrue, yhat)
    print('Classification Report:')
    print(classification_rep)
    confusion_mat = confusion_matrix(ytrue, yhat)
    print('Confusion Matrix:')
    print(confusion_mat)
    # acc from confmat = (tp+tn)/N with tp = true and correct predicted, tn = true and wrong predicted, N = len(X)
    # this responds to Classification report: recall.
    
def evaluate_1_classifier(ytrue,yhat):
    # print('\n')    
    accuracy_classificaion(ytrue,yhat)
    
    # Compute the Jeffreys interval
    alpha = 0.05
    
    # print('\n')
    print('Jeffrey Interval for alpha={}:'.format(alpha))
    [thetahatA, CIA] = jeffrey_interval(ytrue, yhat, alpha=alpha)
    
    print("Theta point estimate", thetahatA, " CI: ", CIA)

def compare_2_classifiers(ytrue, yhatA, yhatB):
    # Compute the McNemar test
    alpha = 0.05
    [thetahat, CI, p] = mcnemar(ytrue, yhatA, yhatB, alpha=alpha)
    # print('\n')
    print("theta = theta_A-theta_C point estimate", thetahat, " CI: ", CI, "p-value", p)
    
    # thetaA denote the (true) chance classier MA is correct and thetaB the (true) chance classier MB is correct
    # theta = thetaA - thetaB
    # with the interpretation that if theta > 0 model A is preferable over B.
    
    # The interpretation is that the lower p is, the more evidence there is A is better than B,
    # but only interpret the p-value together with the estimate thetahat and ideally the confidence
    # interval computed above.
    
    # return thetahat, CI, p
    
def evaluate_1_regression(ytrue,yhat):
    # print('\n')
    z = np.abs(ytrue - yhat ) ** 2
    error_measure = (np.sum(z))/ len(ytrue)
    print('Estimated error z:',error_measure)
    
    alpha = 0.05
    CI = st.t.interval(1-alpha, df=len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    print('Confidence Interval CI:', CI)
    
    accuracy_measure = r2_score(ytrue,yhat)
    print("R-squared score:", accuracy_measure)

def compare_2_regressions(ytrue,yhatA,yhatB):
    # print('\n')
    # compute z with squared error.
    zA = np.abs(ytrue - yhatA ) ** 2
    zB = np.abs(ytrue - yhatB ) ** 2
    print('zA: {0} zB: {1}'.format(np.mean(zA),np.mean(zB)))
    # compute confidence interval of model A
    alpha = 0.05
    CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
    print('CIA: ', CIA)
    CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval
    print('CIB: ', CIB)
    
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    z = zA - zB
    CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    
    print('z:', np.mean(z), 'CI: ',(CI),'p-value: ', (p))

#%%Importing data
if __name__ == '__main__':
        
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    ytrue = y
    yhat = y
    yhatA = y
    yhatB = np.random.randint(3, size = y.shape)
    
    
    
    #%% Classification

    # evaluation of 1 model
    evaluate_1_classifier(ytrue,yhat)
    
    # Compare 2 models
    compare_2_classifiers(ytrue, yhatA, yhatB)
    
    #%% Regression
    
    # Compute accuracy
    evaluate_1_regression(ytrue,yhat)

    
    # Compare 2 models
    compare_2_regressions(ytrue,yhatA,yhatB)