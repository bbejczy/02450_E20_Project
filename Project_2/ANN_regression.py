# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:53:47 2020

@author: bbejc
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from dataProcessing import *
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
#     Function definition
# =============================================================================

def oneOutOfK(attributeNames, classNames, X): 
    #One-out-of-K encoder
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(y)  
    
    attributeNames.insert(0,classNames[2])
    attributeNames.insert(0,classNames[1])
    attributeNames.insert(0,classNames[0])
    
    appendedClasses = np.concatenate((onehot_encoded, X), axis=1)
    
    return appendedClasses

def ANN_reg(X, y, M, attributenNames, classNames):
    
    X = oneOutOfK(attributeNames, classNames, X)
    
    # Parameters for neural network classifier
    n_hidden_units = 4      # number of hidden units
    n_replicates = 10        # number of networks trained in each k-fold
    max_iter = 1000
    
    # K-fold crossvalidation
                      # only three folds to speed up this example
    K=10
    CV = model_selection.KFold(K, shuffle=True)
    
    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M+3, n_hidden_units), #M features to n_hidden_units
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
        
        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')
    
    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')
        
    print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0,2]]
    biases = [net[i].bias.data.numpy() for i in [0,2]]
    tf =  [str(net[i]) for i in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
    
    # Print the average classification error rate
    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
    
    
    
    print('Ran ANN_regression')
    
    return errors

#%% Dummy main

if __name__ == '__main__':

    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #Pre-processing the data
        
    y = y.reshape(len(y),1)
    cent_data = centerData(X)
    X = standardizeData(cent_data) #normalized data
    
    genError = ANN_reg(X, y, M, attributeNames, classNames)
    
    print(*genError)