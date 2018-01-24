# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:16:34 2018

@author: Sudha
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat("ex5data1.mat")

# separating training set, cross validation set and test set

X = data["X"][:8]
Y = data["y"][:8]

Xval = data["X"][8:10]
Yval = data["y"][8:10]

Xtest = data["X"][10:12]
Ytest = data["y"][10:12]

# plotting the data

plt.plot(X, Y, 'ko')
plt.xlabel('Amount of water flowing')
plt.ylabel('Change in water level')
plt.show()

# Regularized linear regression cost function

# adding x0
    
X = np.hstack((np.ones((8,1)), X.reshape(8,1)))

# calculation

theta = np.array([1, 1])
error = np.dot(X, theta.T).reshape(8,1) - Y

cost = np.sum(np.dot(error.T, error))/(2*8) + (np.sum(np.dot(theta.T, theta)) - theta[0]*theta[0])*0.5/(2*8)

# Regularized linear regression gradient

grad = np.dot(X.T, np.dot(X, theta.T).reshape(8,1) - Y)/8 + (np.sum(theta) - theta[0])/8

