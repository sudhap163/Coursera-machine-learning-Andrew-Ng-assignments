# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:21:10 2017

@author: Sudha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("ex2data2.csv")

Y = data["Accepted"]
del data["Accepted"]

X = np.array(data)
Y = np.array(Y)

# plotData

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
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()


# adding bias column to features

bias = pd.Series(1,index=range(len(Y))) 
data["Bias"] = bias
'''
Next four lines is to bring bias column to the first since bias column is X0 and previously it is added to the end of the dataframe
'''
Header_X_Bias = list(data.columns.values)
Header_X_Bias = Header_X_Bias[:-1]
Header_X_Bias.insert(0,"Bias")
data = data[Header_X_Bias]

X = np.array(data)
theta = np.array([0,0,0])

# sigmoid

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

print(sigmoid(0))

# costFunction

def costFunction( X, Y, reg_factor, theta):
    z = np.dot(X, theta.T)
    g = sigmoid(z)
    a = np.multiply(Y, np.log(g))
    b = np.multiply((1-Y), np.log(1-g))
    cost = -a-b
    cost = np.sum(cost)/len(Y) + (reg_factor/(2*len(Y))) * (((np.sum(theta))**2) - theta[0]**2)
    
    return cost

print(costFunction( X, Y, 1, theta))
