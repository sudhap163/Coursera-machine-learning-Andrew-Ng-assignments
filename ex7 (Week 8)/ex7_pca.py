# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:52:47 2018

@author: Sudha
"""

import scipy.io
import numpy as np


#loading files

data = scipy.io.loadmat('ex7data1.mat')
X = data["X"]

# normalising data

normalisedX = X - np.mean(X, axis = 1).reshape(50, 1)

covariance = np.dot(normalisedX.T, normalisedX)/len(X)
eig_val, eig_vector = np.linalg.eig(covariance)


idx = eig_val.argsort()[::-1]   
eig_val = eig_val[idx]
eig_vector = eig_vector[:,idx]

k_eig = eig_vector[:,0:1]

x_new = np.dot(X,k_eig)
x_rec = np.dot(x_new,k_eig.T)