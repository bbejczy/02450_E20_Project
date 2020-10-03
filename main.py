# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc, cm, matty
"""

from dataVisualization import *
from dataProcessing import *
from PCA_analysis import * 

# =============================================================================
#     MAIN
# =============================================================================
if __name__ == '__main__':
    #%%Importing data
    
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    #%% Pre-processing the data
    
    cent_data = centerData(X)
    
    data = standardizeData(cent_data) #normalized data
   
    #%% stats
    
    mean_X,std_X,median_X,min_X,max_X,less_than_25_X,less_than_75_X = stat(X)
    
    
    #%% Boxplot 
    
    boxPlot(data, attributeNames[1:M+1], cols) #could use len(attributeNames) instead of M+1
    
    #%% Correlation Matrix
    
    correlationMatrix(data, cols)
    
    #%% Histogram
    
    histogram(data,M, attributeNames) 

    #%% with Eigenvectors
    X_stand = standardizeData(X)
    values, vectors, explained_variances = EigenvaluePCA(X_stand)
    
    # plot "variance explained"
    plotVariance(explained_variances)
    
    # Project data on PCAs
    projected_data = X_stand @ vectors
    
    # PCA Component coefficients
    pcs = range(5)
    
    PCACoefficients(pcs,vectors,M)
    
    # Visualization
    PCx = 0
    PCy = 1
    PCz = 2
    
    # 2D
    plot2DPCA(projected_data,PCx,PCy,C,y)
    
    # 3D
    plot3DPCA(projected_data,PCx,PCy,PCz,C,y)
    
    
    # ScatterPlot of all PCAs
    
        
    PCAScatterPlot(projected_data,pcs,C,y)