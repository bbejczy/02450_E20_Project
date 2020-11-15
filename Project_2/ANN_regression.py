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
import os, sys

# =============================================================================
#     Function definition
# =============================================================================

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def oneOutOfK(attributeNames, classNames, X, y): 
    #One-out-of-K encoder
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(y)  
    
    attributeNames.insert(0,classNames[2])
    attributeNames.insert(0,classNames[1])
    attributeNames.insert(0,classNames[0])
    
    appendedClasses = np.concatenate((onehot_encoded, X), axis=1)
    
    return appendedClasses

def ANN_reg(X, y, M, attributenNames, classNames, K):
    y = y.reshape(len(y),1)
    X = oneOutOfK(attributeNames, classNames, X, y)
    M = X.shape[1]
    # Parameters for neural network classifier
    #n_hidden_units = 4      # number of hidden units
    n_replicates = 10        # number of networks trained in each k-fold
    max_iter = 1000
    
    yhat =[]
    
    h_best = np.empty(K)
    h_best_error = np.empty(K)
    
    # K-fold crossvalidation
                      # only three folds to speed up this example
    
    CV = model_selection.KFold(K, shuffle=True)
    
    # Setup figure for display of learning curves and error rates in fold
    #summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    
    
    # print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    h = range(2,9,2)
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print("Run fold {}".format(k+1))
        # print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        counter = 0
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        y_est_Mat = np.zeros((len(y_test), len(h)))
        
        h_error_rate = np.empty(len(h))
        
        for n_hidden_units in h:
            print("Run fold {0} with h {1}".format(k+1, n_hidden_units))
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                            )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            with HiddenPrints():
            # Train the net on training data
                net, final_loss, learning_curve = train_neural_net(model,
                                                                   loss_fn,
                                                                   X=X_train,
                                                                   y=y_train,
                                                                   n_replicates=n_replicates,
                                                                   max_iter=max_iter)
         
            # print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            
            y_test_est_numpy = y_test_est.data.numpy()
            
            y_test_est_numpy = y_test_est_numpy.reshape(len(y_test_est_numpy))
            
            y_est_Mat[:,counter]= y_test_est_numpy
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors.append(mse) # store error rate for current CV fold 
            h_error_rate[counter] = mse
            counter +=1
            # Display the learning curve for the best net in the current fold
            # h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
            # h.set_label('CV fold {0}'.format(k+1))
            # summaries_axes[0].set_xlabel('Iterations')
            # summaries_axes[0].set_xlim((0, max_iter))
            # summaries_axes[0].set_ylabel('Loss')
            # summaries_axes[0].set_title('Learning curves')
    
        best_h_index = np.argmin(h_error_rate)
        h_best[k] = h[best_h_index]
        h_best_error[k] = h_error_rate[best_h_index] 
        
        yhat = np.append(yhat,y_est_Mat[:,best_h_index])
    
    # Display the MSE across folds
    # summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
    # summaries_axes[1].set_xlabel('Fold')
    # summaries_axes[1].set_xticks(np.arange(1, K+1))
    # summaries_axes[1].set_ylabel('MSE')
    # summaries_axes[1].set_title('Test mean-squared-error')
        
    # print('Diagram of best neural net in last fold:')
    # weights = [net[i].weight.data.numpy().T for i in [0,2]]
    # biases = [net[i].bias.data.numpy() for i in [0,2]]
    # tf =  [str(net[i]) for i in [1,2]]
    # draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
    
    # Print the average classification error rate
    # print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
    
    unique, ucount = np.unique(h_best, return_counts=True)
    
    index = np.argmax(ucount)
    
    best_h_total = unique[index]
    
    best_error_total = np.min(h_best_error)
    
    
    
    # print('Ran ANN_regression')
    
    return best_error_total, best_h_total, yhat

#%% Dummy main

if __name__ == '__main__':

    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #Pre-processing the data
    K = 2
        
    y = y.reshape(len(y),1)
    cent_data = centerData(X)
    X = standardizeData(cent_data) #normalized data
    
    error, h, yhat = ANN_reg(X, y, M, attributeNames, classNames, K)
    
    # print(error, h, yhat)