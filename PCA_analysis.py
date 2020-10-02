# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:01:39 2020

@author: cm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing

from scipy.linalg import svd
import scipy.linalg as linalg

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
    
    return raw_data,X,y,C,N,M,cols
    
def centerData(data):
    mean = np.mean(data, axis=0)
    center_data = data - mean
    return center_data

def standardizeData(data):
    mean = np.mean(data, axis=0)
    normalize_data = (data - mean)/np.std(data, axis=0)
    return normalize_data


def plotVariance(rho):
    threshold = 0.9
    
    # Plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()
    
def getRho(data):
    U,S,V = svd(data,full_matrices=False)

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()
    return rho

# =============================================================================
#     MAIN
# =============================================================================
    
filename = '../02450_E20_Project/data/wine.data' #add the first folder in this line to the python workspace 

attributeNames = ['ID','Alc', 'Mal-Ac', 'Ash', 'Al-Ash', 'Mg',\
          'T-Phe', 'Flav', 'Nflav-Phe', 'PACs',\
          'Col', 'Hue', 'OD280/315', 'Proline']
    
classNames = ['Clutivar1', 'Cultivar2','Clutivar3']


raw_data,X,y,C,N,M,cols = importData(filename) #importing the raw data from the file


#%% SVD

# standarize data
X_stand = standardizeData(X)

# PCA by computing SVD of Y
U,S,V = svd(X_stand,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

plotVariance(rho)



#%% with Eigenvectors

# Standardizing the features
std_scaler = sklearn.preprocessing.StandardScaler()
X_scaled = std_scaler.fit_transform(X)

# Covariance matrix
features = X_scaled.T
cov_matrix = np.cov(features)

# Eigenvaluedecomposit
values, vectors = np.linalg.eig(cov_matrix)


# explained variances = rho
explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values)) # look up formula in 
 
print(np.sum(explained_variances), '\n', explained_variances)

plotVariance(explained_variances)

# Project data on PCAs

projected_1 = X_scaled @ vectors.T[0] # @ is same as .dot()
projected_2 = X_scaled.dot(vectors.T[1])
projected_3 = X_scaled.dot(vectors.T[2])

# make a table
res = pd.DataFrame(projected_1, columns=['PC1'])
res['PC2'] = projected_2
res['Y'] = y
res.head()

# Visualization

# 2D

plt.figure()
for c in cols:
    class_mask = (y==c)
    plot(projected_1[class_mask], projected_2[class_mask],'o')
    
plt.legend(classNames, loc='lower right')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCs in 2 Dimenstions')
plt.grid()
plt.show()

# 3D
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
for c in cols:
    class_mask = (y==c)
    plot(projected_1[class_mask], projected_2[class_mask], projected_3[class_mask],'o')
    
plt.legend(classNames)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('PCs in 3 Dimenstions', fontsize=20)
plt.grid()
plt.show()

#%% PCA Component coefficients


pcs = [0,1,2,3,4,5]   # outside of function and pass
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = 1/(len(pcs)+2)
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r, attributeNames[1:],fontsize=5)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()


