# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:00:07 2020

@author: bbejc
"""

import numpy as np
import pandas as pd

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

filename = '../data/wine.data'
df = pd.read_csv(filename, header=None)

raw_data = df.to_numpy()

 

cols = range(1, 14)
X = raw_data[:, cols]
Xclass = raw_data[:,0]

 

N, M = X.shape