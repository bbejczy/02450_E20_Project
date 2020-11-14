# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:19:45 2020

Decision Tree Classifier

@author: cm
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import numpy as np
import dataProcessing as dP
import matplotlib.pyplot as plt
import graphviz

#%% Funtions

def classifier_complexity(X,y, attributeNames, classNames):
    
    # split data
    K = 10
    CV = KFold(K, shuffle=False)
    # X_train, X_test, y_train, y_test = KFold(X,y, test_size=0.33)
    
    # Tree complexity parameter - constraint on maximum depth
    tc = np.arange(1, 15, 1)
    
    # Initialize variable
    error_rate_test = np.ones((len(tc),K))
    error_rate_train = np.ones((len(tc),K))
    
    k=0
    for train_index, test_index in CV.split(X):
        # print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
    
        for i, t in enumerate(tc):
            # train model
            classifier = DecisionTreeClassifier(max_depth= t)
            classifier = classifier.fit(X_train,y_train)
            
            # predict
            y_train_est = classifier.predict(X_train)
            y_test_est = classifier.predict(X_test)
            
            # # plot
    
            # plt.figure(dpi = 500)
            # plot_tree(classifier, filled=True, feature_names=attributeNames, class_names=classNames)
            # plt.show()
    
            # Evaluate misclassification rate over train/test data (in this CV fold)
            error_rate_train[i,k] = np.sum(y_train_est != y_train) / float(len(y_train_est))
            error_rate_test[i,k] = np.sum(y_test_est != y_test) / float(len(y_test_est))
        
        
        k+=1
        
    Error_test = error_rate_test.mean(1)
    Error_train = error_rate_train.mean(1)
    
    
    
    # plot errors
    plt.figure()
    plt.plot(tc, Error_train)
    plt.plot(tc, Error_test)
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
    plt.legend(['Error_train','Error_test'])
    plt.show()
    
    # decistion tree for best split
    
    index_opt= np.argmin(Error_test)
    tc_opt = tc[index_opt]
    # print('Optimal Tree Depth:',tc_opt)
    # print('Error test opt',Error_test[index_opt])
    # print('Error train opt:',Error_train[index_opt])
    
    # plot tree, not exactly the same but shape is right
    classifier = DecisionTreeClassifier(max_depth= tc_opt)
    classifier = classifier.fit(X_train,y_train)
    plt.figure(dpi = 500)
    plot_tree(classifier, filled=True, feature_names=attributeNames, class_names=classNames)
    plt.show()
    
    # Errors
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_est))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_est))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_est)))
    return tc_opt

def classifier_model(X,y,K,yhat,ytrue,tc):
    # split data
    K = 10
    CV = KFold(K, shuffle=False)
    # X_train, X_test, y_train, y_test = KFold(X,y, test_size=0.33)
    
    # Initialize variable
    error_rate_test = np.ones(K)
    error_rate_train = np.ones(K)
    
    k=0
    for train_index, test_index in CV.split(X):
        # print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
    
       
        # train model
        classifier = DecisionTreeClassifier(max_depth=tc)
        classifier = classifier.fit(X_train,y_train)
        
        # predict
        y_train_est = classifier.predict(X_train)
        y_test_est = classifier.predict(X_test)
        
        
        yhat = np.append(yhat,y_test_est)
        ytrue = np.append(ytrue,y_test)
    
        
        # # plot

        # plt.figure(dpi = 500)
        # plot_tree(classifier, filled=True, feature_names=attributeNames, class_names=classNames)
        # plt.show()

        # Evaluate misclassification rate over train/test data (in this CV fold)
        error_rate_train[k] = np.sum(y_train_est != y_train) / float(len(y_train_est))
        error_rate_test[k] = np.sum(y_test_est != y_test) / float(len(y_test_est))
        
        
        k+=1
        
    Error_test = error_rate_test.mean()
    Error_train = error_rate_train.mean()
    
    # decistion tree for best split
    # print('Error test opt',Error_test)
    # print('Error train opt:',Error_train)
    
    # plot tree, not exactly the same but shape is right
    # classifier = DecisionTreeClassifier()
    # classifier = classifier.fit(X_train,y_train)
    # plt.figure(dpi = 500)
    # plot_tree(classifier, filled=True, feature_names=attributeNames, class_names=classNames)
    # plt.show()
    
    # Errors
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_est))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_est))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_est)))
    return Error_train,Error_train,yhat,ytrue

def regressor_complexity(X,y,attributeNames,classNames):
    
    K = 10
    #Split data 
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
    
    CV = KFold(K, shuffle=False)
    
    # Tree complexity parameter - constraint on maximum depth
    tc = np.arange(1, 21, 1)
    
    # Initialize variable
    error_rate_test = np.ones((len(tc),K))
    error_rate_train = np.ones((len(tc),K))
    
    k=0
    for train_index, test_index in CV.split(X):
        # print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
    
        for i, t in enumerate(tc):
            # train model
            regressor = DecisionTreeRegressor(max_depth=t)
            regressor.fit(X_train,y_train)
            # predict
            y_train_est = regressor.predict(X_train)
            y_test_est = regressor.predict(X_test)
            
            # Evaluate error rate over train/test data (in this CV fold)
            error_rate_train[i,k] =(np.square(y_train-y_train_est)).sum()/len(y_train)
            error_rate_test[i,k] = np.sum(np.square(y_test-y_test_est))/len(y_test)
        
        k+=1
    
    Error_test = error_rate_test.mean(1)
    Error_train = error_rate_train.mean(1)
    
    
    
    # plot errors
    plt.figure()
    plt.plot(tc, Error_train)
    plt.plot(tc, Error_test)
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('MSE for K={0} CV)'.format(K))
    plt.legend(['Error_train','Error_test'])
    plt.show()
    
    # decistion tree for best split
    
    index_opt= np.argmin(Error_test)
    tc_opt = tc[index_opt]
    # print('Optimal Tree Depth:',tc_opt)
    # print('Error test opt',Error_test[index_opt])
    # print('Error train opt:',Error_train[index_opt])
    
    # plot tree, not exactly the same but shape is right
    regressor = DecisionTreeRegressor(max_depth= tc_opt)
    regressor = regressor.fit(X_train,y_train)
    plt.figure(dpi = 500)
    plot_tree(regressor, filled=True, feature_names=attributeNames)#, class_names=classNames)
    plt.show()
    
    # Graphviz
    # Creates dot file named tree.dot
    # dot_data = export_graphviz(
    #             regressor,
    #             out_file =  None,
    #             feature_names = attributeNames,
    #             class_names = classNames,
    #             filled = True,
    #             rounded = True)
        
    # graph = graphviz.Source(dot_data)
    # graph
    # graph.render('dtree_render_'+clf_name,view=True)
    
    
    
    # # Errors
    # print('\n')
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_est))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_est))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_est)))
     
    return tc_opt

def regressor_model(X,y,K,yhat,ytrue,tc):
    # split data 
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
    
    CV = KFold(K, shuffle=False)
        
    # Initialize variable
    error_rate_test = np.ones(K)
    error_rate_train = np.ones(K)
    
    k=0
    for train_index, test_index in CV.split(X):
        # print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
    
        # train model
        regressor = DecisionTreeRegressor(max_depth=tc)
        regressor.fit(X_train,y_train)
        # predict
        y_train_est = regressor.predict(X_train)
        y_test_est = regressor.predict(X_test)
        
        yhat = np.append(yhat,y_test_est)
        ytrue = np.append(ytrue,y_test)
        
        # Evaluate error rate over train/test data (in this CV fold)
        error_rate_train[k] =(np.square(y_train-y_train_est)).sum()/len(y_train)
        error_rate_test[k] = np.sum(np.square(y_test-y_test_est))/len(y_test)
        
        k+=1
    
    Error_test = error_rate_test.mean()
    Error_train = error_rate_train.mean()

    
    
    # # decistion tree for best split
    # print('Error test opt',Error_test)
    # print('Error train opt:',Error_train)
    
    # plot tree, not exactly the same but shape is right
    # regressor = DecisionTreeRegressor(max_depth= tc_opt)
    # regressor = regressor.fit(X_train,y_train)
    # plt.figure(dpi = 500)
    # plot_tree(regressor, filled=True, feature_names=attributeNames, class_names=classNames)
    # plt.show()
    
    # Graphviz
    # Creates dot file named tree.dot
    # dot_data = export_graphviz(
    #             regressor,
    #             out_file =  None,
    #             feature_names = attributeNames,
    #             class_names = classNames,
    #             filled = True,
    #             rounded = True)
        
    # graph = graphviz.Source(dot_data)
    # graph
    # graph.render('dtree_render_'+clf_name,view=True)
    
    
    
    # # Errors
    # print('\n')
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_est))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_est))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_est)))
    return Error_train,Error_test,yhat,ytrue

if __name__ == '__main__':
    
    #%% import data
    raw_data,X,y,C,N,M, cols,filename,attributeNames,classNames = dP.getData()
    X = dP.standardizeData(X)
    
    #%% Classification
    
    tc = classifier_complexity(X, y, attributeNames, classNames)
    
    K = 10
    yhat = []
    ytrue = []
    Error_train,Error_test,yhat,ytrue=classifier_model(X, y, K, yhat, ytrue,tc)
    
    #%% Regression
    
    regression_attribute = 1
    y = X[:,regression_attribute]
    X_cols = list(range(0,regression_attribute)) + list(range(regression_attribute+1,len(attributeNames)))
    
    
    
    print('Regression on Attribute:',attributeNames[regression_attribute])
    X_without = X[:,X_cols]
    X = dP.standardizeData(X_without)
    K = 10
    yhat = []
    ytrue = []
    
    tc = regressor_complexity(X,y,attributeNames, classNames)
    
    regressor_model(X,y,K,yhat,ytrue,tc)
    
    
    
    
    
    #%% Easy regression
    
    
    
    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
    
    regressor = DecisionTreeRegressor()
    regressor = regressor.fit(X_train,y_train)
    
    # predict
    y_train_est = regressor.predict(X_train)
    y_test_est = regressor.predict(X_test)
    
    # Evaluate misclassification rate over train/test data (in this CV fold)
    Error_train = (np.square(y_train-y_train_est)).sum()/len(y_train)
    Error_test = np.sum(np.square(y_test-y_test_est))/len(y_test)
    
    print('Error test:',Error_test)
    print('Error train:',Error_train)
    
    plt.figure(dpi = 500)
    plot_tree(regressor, filled=True, feature_names=attributeNames, class_names=classNames)
    plt.show()


