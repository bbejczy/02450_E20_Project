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

# =============================================================================
#     MAIN - change values after here
# =============================================================================

filename = '../02450_E20_Project/data/wine.data' #add the first folder in this line to the python workspace


attributeNames = ['ID','Alc', 'Mal-Ac', 'Ash', 'Al-Ash', 'Mg',\
          'T-Phe', 'Flav', 'Nflav-Phe', 'PACs',\
          'Col', 'Hue', 'OD280/315', 'Proline']
    
classNames = ['Clutivar1', 'Cultivar2','Clutivar3']