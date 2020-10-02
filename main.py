# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.linalg import svd

#from PCA_analysis import example

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

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

def normalizeData(data):
    mean = np.mean(data, axis=0)
    normalize_data = (data - mean)/np.std(data, axis=0)
    return normalize_data

def boxPlot(data, names):

    plt.boxplot(data)
    plt.xticks(cols, names, rotation='vertical')
    plt.title('Wine Data Boxplot')
    plt.show()
    
def correlationMatrix(data):
    correlationMatrix = np.corrcoef(data)   #Return Pearson product-moment correlation coefficients.
    data = pd.DataFrame(data).corr()        #Compute pairwise correlation of columns, excluding NA/null values.
 
    sn.heatmap(data, xticklabels=cols, yticklabels=cols, annot=True, annot_kws={"size":7})
 
    plt.title('Wine Data Correlation Matrix', fontsize = 14)
    
    return correlationMatrix

def histogram(data,grid):
    
    plt.figure(figsize=(14,10))
    plt.suptitle('Histogram', fontsize=20)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5) #plot sizing adjustments
    
    #Subplot grid dimensions
    dim = np.round(np.sqrt(grid))
    row = dim
    col = dim
    cnt = 1

    for i in range(M):   
        plt.subplot(row, col, cnt)
        plt.hist(data[:,i], bins = 36)
        plt.title(attributeNames[i], fontsize=12)
        cnt = cnt + 1
    
    plt.show()
    
# =============================================================================
#     MAIN
# =============================================================================
    
filename = '../02450_E20_Project/data/wine.data' #add the first folder in this line to the python workspace 

attributeNames = ['ID','Alc', 'Mal-Ac', 'Ash', 'Al-Ash', 'Mg',\
          'T-Phe', 'Flav', 'Nflav-Phe', 'PACs',\
          'Col', 'Hue', 'OD280/315', 'Proline']
    
classNames = ['Clutivar1', 'Cultivar2','Clutivar3']


raw_data,X,y,C,N,M,cols = importData(filename)      #importing the raw data from the file


#%% Pre-processing the data

cent_data = centerData(X)

data = normalizeData(cent_data) #normalized data

#%% Boxplot 

boxPlot(data, attributeNames[1:M+1]) #could use len(attributeNames) instead of M+1

#%% Correlation Matrix

correlationMatrix(data)

#%% Histogram

histogram(data,M)





