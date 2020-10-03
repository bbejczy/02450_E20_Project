# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""


from dataVisualization import *
from dataProcessing import *




# =============================================================================
#     MAIN
# =============================================================================

#%%Importing data
    
raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file

#%% Pre-processing the data

cent_data = centerData(X)

data = standardizeData(cent_data) #normalized data

#%% Boxplot 

boxPlot(data, attributeNames[1:M+1], cols) #could use len(attributeNames) instead of M+1

#%% Correlation Matrix

correlationMatrix(data, cols)

#%% Histogram

histogram(data,M, attributeNames)





