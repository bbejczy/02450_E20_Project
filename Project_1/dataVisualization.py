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
    
    boxPlotFig = plt.figure()
    plt.boxplot(data)
    plt.xticks(coloumns, names, rotation='vertical')
    plt.title('Wine Data Boxplot')
    plt.show()
    boxPlotFig.savefig('../Plots/BoxPlot.pdf')
    
def correlationMatrix(data, coloumns, attributeNames):
    
    correlationMatrix = np.corrcoef(data) #Return Pearson product-moment correlation coefficients.
    data = pd.DataFrame(data).corr() #Compute pairwise correlation of columns, excluding NA/null values.
    
    corrMatFig = plt.figure();
    plt.title('Wine Data Correlation Matrix', fontsize = 14)
    sn.set(font_scale=0.7)
    plt.gcf().subplots_adjust(bottom=0.15)
    sn.heatmap(data, xticklabels=attributeNames[1:], yticklabels=attributeNames[1:], annot=True, annot_kws={"size":7}, cmap="viridis")
    corrMatFig.savefig('Plots/CorrelationMatrix.pdf')
    
    return correlationMatrix

def histogram(data,grid, attributeNames):
    
    histFig = plt.figure(figsize=(14,10))
    plt.suptitle('Wine Data Histogram', fontsize=20)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4) #plot sizing adjustments
    
    #Subplot grid dimensions
    dim = np.round(np.sqrt(grid))
    row = dim
    col = dim
    cnt = 1

    for i in range(grid):   
        plt.subplot(row, col, cnt)
        plt.hist(data[:,i], bins = 39)
        plt.xlim(xmin=-4, xmax = 4)
        plt.title(attributeNames[i+1], fontsize=12)
        cnt = cnt + 1
    
    plt.show()
    histFig.savefig('Plots/Histogram.pdf')
    

    
    

    