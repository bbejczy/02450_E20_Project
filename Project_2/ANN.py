# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:15:58 2020

@author: bbejc
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from scipy import stats

from dataProcessing import *

raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
K = 10
# =============================================================================
# Function definitions
# =============================================================================

def ANN(K, X, y, M):
    
    y = y.reshape(len(y),1)

    # Normalize data
    X = stats.zscore(X)
                    
    ## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
    do_pca_preprocessing = False
    if do_pca_preprocessing:
        Y = stats.zscore(X,0)
        U,S,V = np.linalg.svd(Y,full_matrices=False)
        V = V.T
        #Components to be included as features
        k_pca = 3
        X = X @ V[:,:k_pca]
        N, M = X.shape
    
    
    # Parameters for neural network classifier
    n_hidden_units = 8     # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 3000
    
    # K-fold crossvalidation
    K = 10                  # only three folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)
    
    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
        
    
    return mse


error = ANN(K, X, y, M)

print("error rate:",error)