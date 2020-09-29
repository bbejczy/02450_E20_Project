# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.linalg import svd

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# =============================================================================
# Function definitions
# =============================================================================


def importData(filename):
      
    df = pd.read_csv(filename, header=None)    
    raw_data = df.to_numpy()
    
    cols = range(1, 14)
    X = raw_data[:, cols]
    y = raw_data[:,0]
    C = len(np.unique(y))
   
    N, M = X.shape
    
    return raw_data,X,y,C,N,M
    
def centerData(data):
    mean = np.mean(data, axis=0)
    center_data = data - mean
    return center_data

def normalizeData(data):
    mean = np.mean(data, axis=0)
    normalize_data = (data - mean)/np.std(data, axis=0)
    return normalize_data

def boxplot(data, names):

    plt.boxplot(data)
    plt.xticks(range(1, 14), names, rotation='vertical')
    plt.title('Wine Data Boxplot')
    plt.show()
    
def correlationMatrix(data):
    correlationMatrix = np.corrcoef(data) #Return Pearson product-moment correlation coefficients.
    data = pd.DataFrame(data).corr() #Compute pairwise correlation of columns, excluding NA/null values.
 
    sn.heatmap(data, xticklabels=range(1,14), yticklabels=range(1,14), annot=True, annot_kws={"size":7})
 
    plt.title('Wine Data Correlation Matrix', fontsize = 14)
    
    return correlationMatrix

    
# =============================================================================
#     MAIN
# =============================================================================
    
filename = '../02450_E20_Project/data/wine.data' #add the first folder in this line to the python workspace 

attributeNames = ['ID','Alc', 'Mal-Ac', 'Ash', 'Al-Ash', 'Mg',\
          'T-Phe', 'Flav', 'Nflav-Phe', 'PACs',\
          'Col', 'Hue', 'OD280/315', 'Proline']
    
classNames = ['Clutivar1', 'Cultivar2','Clutivar3']


raw_data,X,y,C,N,M = importData(filename) #importing the raw data from the file


#%% Pre-processing the data

cent_data = centerData(X)

data = normalizeData(cent_data) #normalized data

#%% Boxplot 

boxplot(data, attributeNames[1:M+1]) #could use len(attributeNames) instead of M+1

#%% Correlation Matrix

correlationMatrix(data)

#%% 





