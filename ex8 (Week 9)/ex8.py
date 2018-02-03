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
    mu = (np.sum(X, axis = 1)/(len(X))).reshape(307,1)
    sigma_square = (np.sum(np.power((X - mu), 2), axis = 1)/len(X)).reshape(307, 1)
    return mu, sigma_square

# calculating probablility

def computeProbability(X):
    mu, sigma_square = mu, sigma_square = estimateGaussian(X)
    return (1/np.power((2*math.pi*sigma_square), 0.5)) * np.exp(-np.power((X - mu), 2)/(2 * sigma_square))

# selecting threshold

probability = computeProbability(Xval)