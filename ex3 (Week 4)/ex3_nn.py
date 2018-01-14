# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:50:18 2018

@author: Sudha
"""

import numpy as np
import scipy.io

data = scipy.io.loadmat("ex3data1.mat")
weights = scipy.io.loadmat("ex3weights.mat")

Y = data["y"]
X = data["X"]

X = np.hstack((np.ones((5000,1)), X.reshape(5000,400)))

theta1 = weights["Theta1"].reshape(25, 401)
theta2 = weights["Theta2"].reshape(10, 26)

# sigmoid function

def sigmoid(z):
    return (1/(1 + np.exp(-z)))
    
# feedforward propagation

n1 = np.dot(X, theta1.T)                    # 5000x25
a1 = sigmoid(n1)                            # 5000x25

a1 = np.hstack((np.ones((5000,1)), a1))     # 5000x26

n2 = np.dot(a1, theta2.T)                   # 5000x10
a2 = sigmoid(n2)

# finding accuracy

newY = (np.argmax(a2, axis = 1) + 1).reshape(5000,1)

error = Y - newY

accuracy = ((5000 - np.nonzero(error)[0].shape[0])/5000)*100
            
print ('Accuracy: ', accuracy)



