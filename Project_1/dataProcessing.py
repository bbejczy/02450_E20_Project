# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:02:44 2020

@author: bbejc
"""

import numpy as np
import pandas as pd
# =============================================================================
# Function definitions
# =============================================================================


def importData(filename):
      
    df = pd.read_csv(filename, header=None)    
    raw_data = df.to_numpy()
    
    cols = range(1, len(attributeNames)) #range object
    
    X = raw_data[:, cols]
    y = raw_data[:,0]
    C = len(np.unique(y))
   
    N, M = X.shape
    
    return raw_data,X,y,C,N,M, cols

def centerData(data):
    mean = np.mean(data, axis=0)
    center_data = data - mean
    return center_data

def standardizeData(data):
    mean = np.mean(data, axis=0)
    normalize_data = (data - mean)/np.std(data, axis=0)
    return normalize_data

def stat(X):

    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    median_X = np.median(X,axis=0)
    min_X = X.min(axis=0)
    max_X = X.max(axis=0)
    less_than_25_X = np.quantile(X, .25, axis = 0)
    less_than_75_X = np.quantile(X, .75, axis = 0)

    return mean_X,std_X,median_X,min_X,max_X,less_than_25_X,less_than_75_X



filename = '../Project_1/data/wine.data' #add the first folder in this line to the python workspace
    
    
attributeNames = ['ID','Alc', 'Mal-Ac', 'Ash', 'Al-Ash', 'Mg',\
          'T-Phe', 'Flav', 'Nflav-Phe', 'PACs',\
          'Col', 'Hue', 'OD280/315', 'Proline']
    
classNames = ['Clutivar1', 'Cultivar2','Clutivar3']

# =============================================================================
#     MAIN - change values after here
# =============================================================================
if __name__ == '__main__':
    raw_data,X,y,C,N,M, cols = importData(filename)
    
    # stats
    
    mean_X,std_X,median_X,min_X,max_X,less_than_25_X,less_than_75_X = stat(X)
    