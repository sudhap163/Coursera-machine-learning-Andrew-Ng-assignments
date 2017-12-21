# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:57:29 2017

@author: Sudha
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

data = scipy.io.loadmat("ex3data1.mat")

Y = data["y"]
X = data["X"]

# displayData

q = X.reshape(5000,20,20)
for i in range(0,5000):
    q[i] = q[i].T
m=np.random.choice(range(5000),100,replace=False)
n = m.tolist()
M = []
for i in range(0,10):
    w = np.concatenate((q[n[i+0]],q[n[i+1]],q[n[i+2]],q[n[i+3]],q[n[i+4]],q[n[i+5]],q[n[i+6]],q[n[i+7]],q[n[i+8]],q[n[i+9]]), axis =1)
    M.append(w)    
N = np.concatenate((M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9]),axis=0)
imgplot = plt.imshow(N, cmap='gray')

# costFunction

theta = 0

def sigmoid(z):                     # function to calculate sigmoid value
    return (1/(1 + np.exp(-z)))

def costFunction( X, Y, theta):

    z = np.dot(X, theta.T)
    g = sigmoid(z)
    a = np.multiply(Y, np.log(g))
    b = np.multiply((1-Y), np.log(1-g))
    cost = -a-b
    cost = np.sum(cost)/len(Y)
    
    return cost

print(costFunction( X, Y, theta))