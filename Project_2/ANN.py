# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:15:58 2020

@author: bbejc
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
import torch
from torch import nn
import torch.nn.functional as F
from scipy import stats
from dataProcessing import *

# =============================================================================
# Function definitions
# =============================================================================

def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):

    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        net = model()
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        #optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve

def ANN(K, X, y, M):
    
    y = y.reshape(len(y),1)

    # Normalize data
    X = stats.zscore(X)
                          
    # Parameters for neural network classifier
    n_hidden_units = 8     # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    

    CV = model_selection.KFold(K, shuffle=True)

    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )

    
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    # print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # # Train the net on training data

        net, final_loss, learning_curve = train_neural_net(model,
                                                              loss_fn,
                                                              X=X_train,
                                                              y=y_train,
                                                              n_replicates=n_replicates,
                                                              max_iter=max_iter)
                
        # Determine estimated class labels for test set
        y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
        y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_test = y_test.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_test_est != y_test)
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
        errors.append(error_rate) # store error rate for current CV fold 
        
        yhat = y_test_est
        ytrue = y_test
        
        
    
    return errors, yhat, ytrue


#%%
if __name__ == '__main__':
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #%% K-Fold-Validation
    
    K_inner = 10
    yhat = []
    ytrue = []

    errors, yhat, ytrue = ANN(K_inner, X, y, M)
    
    

print(*errors)