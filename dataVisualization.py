# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:52:32 2020

@author: bbejc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.linalg import svd

from dataProcessing import *

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show


# =============================================================================
# Function definitions
# =============================================================================


def boxPlot(data, names, coloumns):

    plt.boxplot(data)
    plt.xticks(coloumns, names, rotation='vertical')
    plt.title('Wine Data Boxplot')
    plt.show()
    
def correlationMatrix(data, coloumns):
    correlationMatrix = np.corrcoef(data) #Return Pearson product-moment correlation coefficients.
    data = pd.DataFrame(data).corr() #Compute pairwise correlation of columns, excluding NA/null values.
 
    sn.heatmap(data, xticklabels=coloumns, yticklabels=coloumns, annot=True, annot_kws={"size":7})
 
    plt.title('Wine Data Correlation Matrix', fontsize = 14)
    
    return correlationMatrix

def histogram(data,grid, attributeNames):
    
    plt.figure(figsize=(14,10))
    plt.suptitle('Histogram', fontsize=20)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5) #plot sizing adjustments
    
    #Subplot grid dimensions
    dim = np.round(np.sqrt(grid))
    row = dim
    col = dim
    cnt = 1

    for i in range(grid):   
        plt.subplot(row, col, cnt)
        plt.hist(data[:,i], bins = 50)
        plt.title(attributeNames[i+1], fontsize=12)
        cnt = cnt + 1
    
    plt.show()
    

    
    

    