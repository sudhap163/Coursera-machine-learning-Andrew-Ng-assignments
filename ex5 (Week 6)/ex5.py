# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:16:34 2018

@author: Sudha
"""

import scipy.io
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

