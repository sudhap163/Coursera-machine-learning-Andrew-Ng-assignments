# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:57:08 2018

@author: Sudha
"""

import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# loading first dataset

data = scipy.io.loadmat("ex8data1.mat")

X = np.array(data["X"])
Xval = np.array(data["Xval"])
Yval = np.array(data["yval"])

# plot the data

plt.plot(X[:, 0], X[:, 1], 'b+')
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')
plt.show()

# calculating Gaussian parameters

def estimateGaussian(X):
    mu = (np.sum(X, axis = 0)/(len(X))).reshape(1,2)
    sigma_square = (np.sum(np.power((X - mu), 2), axis = 0)/len(X)).reshape(1, 2)
    return mu, sigma_square

# calculating probablility
mu, sigma_square = mu, sigma_square = estimateGaussian(X)
def computeProbability(X):
    mu, sigma_square = estimateGaussian(X)
    covariance = np.dot(X.T, X).reshape(2,2)
    prob = np.array(np.exp( -0.5 * np.dot(np.dot((X[0].reshape(1,2) - mu), covariance), (X[0].reshape(1,2) - mu)))/(2*np.pi*sigma_square[0][0]*sigma_square[0][1]))
    for i in range(1, 307):
        np.append(prob, np.exp( -0.5 * np.dot(np.dot((X[i].reshape(1,2) - mu), covariance), (X[i].reshape(1,2) - mu)))/(2*np.pi*sigma_square[0][1]*sigma_square[0][1]))
    return prob

# selecting threshold

pval = computeProbability(Xval)