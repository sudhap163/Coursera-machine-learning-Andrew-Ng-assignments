# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:57:29 2017

@author: Sudha
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
    w = np.concatenate((q[n[(i*10)+0]],q[n[(i*10)+1]],q[n[(i*10)+2]],q[n[(i*10)+3]],q[n[(i*10)+4]],q[n[(i*10)+5]],q[n[(i*10)+6]],q[n[(i*10)+7]],q[n[(i*10)+8]],q[n[(i*10)+9]]), axis =1)
    M.append(w)    
N = np.concatenate((M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9]),axis=0)
imgplot = plt.imshow(N, cmap='gray')

# logistic regression

X = X.reshape(5000, 400)
k = np.zeros((10, 5000))

for i in range(0,10):
    for j in range (0,5000):
        if (Y[j] == (i+1)):
            k[i][j] = 1
        else:
            k[i][j] = 0

lr = []

for i in range(0,10):
    lr.append(LogisticRegression())
    lr[i].fit(X, k[i])
    if ( i==0 ):
        probability = lr[i].predict_proba(X)[:,1].reshape(5000,1)
    else:
        probability = np.hstack((probability, lr[i].predict_proba(X)[:,1].reshape(5000,1)))

newY = (np.argmax(probability, axis = 1) + 1).reshape(5000,1)

error = Y - newY

accuracy = ((5000 - np.nonzero(error)[0].shape[0])/5000)*100
            
print ('Accuracy: ', accuracy)