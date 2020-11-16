# exercise 8.3.1 Fit neural network classifiers using softmax output weighting
import matplotlib.pyplot as plt
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import torch
from scipy import stats
from sklearn import model_selection
from dataProcessing import *
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def ANN_multiClass(X,y,K,C):
    
    M = len(X[0])
    
    #y = y.reshape(len(y),1)
    y = y.squeeze()
    
    yhat = []
    
    h_best = np.zeros(K)
    h_best_error = np.zeros(K)
    
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    
    CV = model_selection.KFold(K, shuffle=False)
    #%% Model fitting and prediction
    
    h = range(1,10,2)
    
    mat_error = np.zeros((K,len(h)))
    
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        # print('\n Crossvalidation fold: {0}/{1} %%%%%%%%%%% \n'.format(k+1,K))
        total_error_fold = 0
        counter = 0
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        
        y_test = y_test.numpy()
        
        h_error_rate = np.empty(len(h))
        
        y_est_Mat = np.zeros((len(y_test), len(h)))
        
        
        for n_hidden_units in h:
            
            model = lambda: torch.nn.Sequential(
                                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                                        torch.nn.ReLU(), # 1st transfer function
                                        # Output layer:
                                        # H hidden units to C classes
                                        # the nodes and their activation before the transfer 
                                        # function is often referred to as logits/logit output
                                        torch.nn.Linear(n_hidden_units, C+3), # C logits
                                        # To obtain normalised "probabilities" of each class
                                        # we use the softmax-funtion along the "class" dimension
                                        # (i.e. not the dimension describing observations)
                                        torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                                        )
            # Since we're training a multiclass problem, we cannot use binary cross entropy,
            # but instead use the general cross entropy loss:
            loss_fn = torch.nn.CrossEntropyLoss()
            
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()
                
            # Train the network:
            with HiddenPrints():
                net, _, _ = train_neural_net(model, loss_fn,
                                             X=torch.tensor(X_train, dtype=torch.float),
                                             y=torch.tensor(y_train, dtype=torch.long),
                                             n_replicates=1,
                                             max_iter=1000)
            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
            
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            
            # print(y_test_est)
            
            y_est_Mat[:,counter]= y_test_est
            
            
            # print(y_est_Mat)
            # print(len(yhat))
            
            # Determine errors
            e = (y_test_est != y_test)
            #print(e)
            test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)
            h_error_rate[counter] = test_error_rate
            #plt.bar(h+w[counter], np.squeeze(np.asarray(h_error_rate)), width = 0.1)
            # mat_error[k,counter] = test_error_rate
            counter += 1
            # print("h unit: {0} for fold {1}".format(n_hidden_units, k+1))
            # print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))
            # print('Number of miss-classifications for ANN:\n\t {0} out of {1}\n'.format(sum(e),len(e)))
     
        best_h_index = np.argmin(h_error_rate)
        h_best[k] = h[best_h_index]
        h_best_error[k] = h_error_rate[best_h_index] 
        
        yhat = np.append(yhat,y_est_Mat[:,best_h_index])
        # print(yhat)
        # yhat_length = len(yhat)
        # print("lenght of yhat: {}".format(yhat_length))
            
        
            
        # print("Best h value for fold{0} is {1}, with the error rate of {2}".format(k+1, best_h_index+1, h_error_rate[best_h_index]))
        # print("Total error: {0} in fold{1}".format(total_error_fold,k+1))
    
    
    #%%
    
    # plt.figure()
    # ax = plt.axes()
    # ax.set_xlabel('h value')
    # ax.set_xticks(h)
    # ax.set_ylabel('e_test')
    # ax.set_title('Classification error for K fold')
    # m,n = mat_error.shape
    # for i in range(m):
    #     for j in range(n):
    #         plt.bar(h+w[i], mat_error[i,j], width = 0.1, color=color_list[i])
    
    # plt.show()    
    unique, ucount = np.unique(h_best, return_counts=True)
    
    index = np.argmax(ucount)
    
    best_h_total = unique[index]
    
    best_error_total = np.min(h_best_error)
    

    
    
    # print('Error of best h:',best_h_total, 'is:', best_error_total)
    
    return best_error_total, best_h_total, yhat

#%% Dummy main

if __name__ == '__main__':

    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    index = np.arange(0,len(X))
    np.random.shuffle(index)
    
    X = X[index,:]
    y = y[index]
    
    
    #Pre-processing the data
        
    y = y.reshape(len(y),1)
    cent_data = centerData(X)
    X = standardizeData(cent_data) #normalized data
    
    K = 2
    
    classError, h, yhat = ANN_multiClass(X,y,K,C)
    
    print(classError, h, yhat)