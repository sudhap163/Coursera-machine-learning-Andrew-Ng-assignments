# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:49:14 2018

@author: Sudha
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm

data = scipy.io.loadmat("ex6data1.mat")

X = np.array(data["X"])
Y = np.array(data["y"])

# plotting the data

X_0 = []
X_1 = []
for i in range(0, len(Y)):
    if ( Y[i] == 0 ):
        X_0.append(X[i])
    else:
        X_1.append(X[i])
        
X_0 = np.array(X_0)
X_1 = np.array(X_1)

plt.plot(X_0[:,0], X_0[:,1], 'yo')
plt.plot(X_1[:,0], X_1[:,1], 'k+')
plt.xlabel('X[:, 0]')
plt.ylabel('X[:, 1]')
plt.show()

# Gaussian kernel

def gaussianKernel( X1, X2, sigma = 0.3 ):
    #return np.exp(-np.sum( np.power(( X1 - X2), 2), axis = 1)/(2*sigma**2))
    mat = np.zeros((len(X1), len(X2)))
    for i in range(0,len(X1)):
        for j in range(0,len(X2)):
            mat[i][j] = np.exp(-np.sum( np.power(( X1[i] - X2[j]), 2))/(2*sigma**2))
    return mat

# k = gaussianKernel(np.array([1, 2, 1]), np.array([0, 4, -1]), 2)

# TO FIND VALUE OF C AND SIGMA

# loading data

data = scipy.io.loadmat("ex6data3.mat")

X = np.array(data["X"])
Y = np.array(data["y"])
Xval = np.array(data["Xval"])
Yval = np.array(data["yval"])

# plotting the data

X_0 = []
X_1 = []
for i in range(0, len(Y)):
    if ( Y[i] == 0 ):
        X_0.append(X[i])
    else:
        X_1.append(X[i])
        
X_0 = np.array(X_0)
X_1 = np.array(X_1)

plt.plot(X_0[:,0], X_0[:,1], 'yo')
plt.plot(X_1[:,0], X_1[:,1], 'k+')
plt.xlabel('X[:, 0]')
plt.ylabel('X[:, 1]')
plt.show()

'''
# SVM classifier

sigma = 0.3

clf = svm.SVC(C = 1, kernel = 'precomputed')
model = clf.fit( gaussianKernel(X, X, sigma), Y )
p = model.predict( gaussianKernel(Xval, X) )
'''

# predict 

C = sigma = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
error = np.zeros((8, 8, 200))

for i in range(len(C)):
    for j in range(len(sigma)):
        clf = svm.SVC(C[i], kernel = 'precomputed')
        model = clf.fit( gaussianKernel(X, X, sigma[j]), Y )
        p = model.predict( gaussianKernel(Xval, X) )
        error[i][j] = p.reshape(200,1) - Yval

''' min = np.argmin(np.argmin(error, axis = 0), axis = 1) '''

min = error[0][0]

for i in range(len(C)):
    for j in range(len(sigma)):
        if (error[i][j] != 0):
            print()