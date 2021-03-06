# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:01:39 2020

@author: cm
"""

import numpy as np
import matplotlib.pyplot as plt
from dataProcessing import *

# import sklearn.preprocessing

from scipy.linalg import svd
# from dataProcessing import *
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# =============================================================================
# Function definitions
# =============================================================================


# def PCAimportData(filename):
#     df = pd.read_csv(filename, header=None)    
#     raw_data = df.to_numpy()
#     cols = range(1, len(attributeNames)) #range object
#     X = raw_data[:, cols]
#     y = raw_data[:,0]
#     C = len(np.unique(y))
#     N, M = X.shape
#     return raw_data,X,y,C,N,M,cols
    
# def PCAcenterData(data):
#     mean = np.mean(data, axis=0)
#     center_data = data - mean
#     return center_data

# def PCAstandardizeData(data):
#     mean = np.mean(data, axis=0)
#     normalize_data = (data - mean)/np.std(data, axis=0)
#     return normalize_data
   
def EigenvaluePCA(X_stand):
    # # Standardizing the features
    # std_scaler = sklearn.preprocessing.StandardScaler()
    # X_scaled = std_scaler.fit_transform(X)
    # Standardizing the features
    # X_stand = standardizeData(X)
    # Covariance matrix
    features = X_stand.T
    cov_matrix = np.cov(features)
    # Eigenvaluedecomposit
    values, vectors = np.linalg.eig(cov_matrix)
    # explained variances = rho
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values)) # look up formula in 
    return values, vectors, explained_variances

def SVDPCA(X_stand):
    # PCA by computing SVD of Y
    U,S,Vh = svd(X_stand,full_matrices=False)
    V=Vh.T
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()
    
    return rho,V

def plotVariance(rho):
    threshold = 0.9
    # Plot variance explained
    fig = plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()
    
    fig.savefig('../Plots/VarianceExplained.pdf')


def plot2DPCA(projected_data,PCx,PCy,C,y):
    fig=plt.figure()
    for c in range(1,C+1):
        class_mask = (y==c)
        plt.plot(projected_data[class_mask,PCx], projected_data[class_mask,PCy],'o')
        
    plt.legend(classNames, loc='lower right')
    plt.xlabel('PC{}'.format(PCx+1))
    plt.ylabel('PC{}'.format(PCy+1))
    plt.title('PCs in 2 Dimenstions')
    plt.grid()
    plt.show()
    
    fig.savefig('../Plots/2DPCA.pdf')
    
def plot3DPCA(projected_data,PCx,PCy,PCz,C,y):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    for c in range(1,C+1):
        class_mask = (y==c)
        plt.plot(projected_data[class_mask,PCx], projected_data[class_mask, PCy], projected_data[class_mask,PCz],'o')
        
    plt.legend(classNames)
    ax.set_xlabel('PC{}'.format(PCx+1))
    ax.set_ylabel('PC{}'.format(PCy+1))
    ax.set_zlabel('PC{}'.format(PCz+1))
    plt.title('PCs in 3 Dimenstions', fontsize=20)
    plt.grid()
    plt.show()
    
    fig.savefig('../Plots/3DPCA.pdf')

def PCACoefficients(pcs,vectors,M):
    fig = plt.figure()
    legendStrs = ['PC'+str(e+1) for e in pcs]
    bw = 1/(len(pcs)+2)
    r = np.arange(1,M+1)
    for i in pcs:    
        plt.bar(r+i*bw, vectors[:, i], width=bw)
    plt.xticks(r, attributeNames[1:],fontsize=5)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('PCA Component Coefficients')
    plt.show()
    
    fig.savefig('../Plots/PCACoefficients.pdf')
    
def PCAScatterPlot(projected_data,pcs,C,y):
    rows = len(pcs)
    cols = len(pcs)
    counter_endcount = rows*cols
    counter = 1
    
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5) #plot sizing adjustments

    for y_pointer in pcs:
        for x_pointer in pcs:
            plt.subplot(rows,cols,counter)
            for c in range(1,C+1):
                class_mask = (y==c)
                plt.plot(projected_data[class_mask,x_pointer], projected_data[class_mask,y_pointer],'o')
                plt.xlim(-5,5)
                plt.ylim(-5,5)
                plt.grid()
            
                # for axis labels
                if  counter > counter_endcount-cols: 
                    plt.xlabel('PC{}'.format(x_pointer+1), fontsize=15)
                for r in range(cols):    
                    if counter==rows*r+1:
                        plt.ylabel('PC{}'.format(y_pointer+1), fontsize=15)
                
            counter=counter+1
    plt.suptitle('PCAs', fontsize=40)
    fig.legend(classNames, loc='upper right', fontsize=15)
    plt.show()
    
    fig.savefig('../Plots/PCAScatter.pdf')

#%% =============================================================================
#     MAIN
# =============================================================================



if __name__ == '__main__':
    
    # Import data
    
    
    raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file
    
    
    # Standarize data
    X_stand = standardizeData(X)
    
    
    
    #%% with SVD
    rho,V = SVDPCA(X_stand)
    
    # plot "variance explained"
    plotVariance(rho)
    
    # Project the centered data onto principal component space
    Z = X_stand @ V
    
    #PCA Component coefficients
    pcs = range(5)
    
    PCACoefficients(pcs,V,M)
    
    # Visualization over specific vectors
    PCx = 0
    PCy = 1
    PCz = 2
    
    # 2D
    plot2DPCA(Z,PCx,PCy,C,y)
    
    # 3D
    plot3DPCA(Z,PCx,PCy,PCz,C,y)
    
    
    
    #%% with Eigenvectors - alternative approach, currently unused
     
    
    # values, vectors, explained_variances = EigenvaluePCA(X_stand)
    
    # # plot "variance explained"
    # plotVariance(explained_variances)
    
    # # Project data on PCAs
    # projected_data = X_stand @ vectors
    
    # # PCA Component coefficients
    # pcs = range(5)
    
    # PCACoefficients(pcs,vectors)
    
    # # Visualization
    # PCx = 0
    # PCy = 1
    # PCz = 2
    
    # # 2D
    # plot2DPCA(projected_data,PCx,PCy)
    
    # # 3D
    # plot3DPCA(projected_data,PCx,PCy,PCz)
    
    
    #%% ScatterPlot of all PCAs 
    
    pcs = range(5)
         
        
    PCAScatterPlot(Z,pcs,C,y)
    
    
    print('ran PCA analysis')