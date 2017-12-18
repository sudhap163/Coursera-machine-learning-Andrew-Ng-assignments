# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:00:12 2017

@author: Sudha
"""

import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt

data = pd.read_csv("ex2data1.csv")

Y = data["Admitted"]
del data["Admitted"]

X = np.array(data)
Y = np.array(Y)

# plotData

X_0 = []
X_1 = []
for i in range(0,len(Y)):
    if(Y[i]==0):
        X_0.append(X[i])
    else:
        X_1.append(X[i])

X_s0 = np.array(X_0)
X_s1 = np.array(X_1)

plt.plot(X_s0[:,0],X_s0[:,1] , 'yo')
plt.plot(X_s1[:,0],X_s1[:,1] , 'k+')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
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

'''
Now X is (100,3)
Y is (100,1)
theta is (1,3)

Now, X*theta.T = (100,1) 
'''
# sigmoid

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

print(sigmoid(0))

# costFunction

def costFunctionMatrix( X, Y, theta):

    z = np.dot(X, theta.T)
    g = sigmoid(z)
    a = np.multiply(Y, np.log(g))
    b = np.multiply((1-Y), np.log(1-g))
    cost = -a-b
    cost = np.sum(cost)/len(Y)
    
    return cost

print(costFunctionMatrix( X, Y, theta))

def costFunctionElement(X, Y, theta):
    
    cost = 0    
    for i in range(len(Y)):
        
        z = np.dot(X[i], theta.T)
        g = sigmoid(z)
        a = Y[i]*math.log(g)
        b = (1-Y[i])*math.log(1-g)
        cost += -a-b
    
    cost = cost/len(Y)
    
    return cost
    
print(costFunctionElement( X, Y, theta))

alpha = 0.01
Iterations = 5000

Cost_History = []
Theta_History = []

def gradient(X,Y,Theta,Iterations,alpha):
    for i in range(Iterations):
        Loss = sigmoid(np.dot(X, Theta.T)) - Y
        Cost = costFunctionMatrix(X,Y,Theta)
        dJ = (np.dot(X.T,Loss))/len(Y) #Calculating Partial differentiation of Cost function
        Cost_History.append(Cost)
        Theta_History.append(Theta)
        
        Theta = Theta - (alpha*dJ) #New Theta
    return Theta

theta = gradient(X,Y,theta,Iterations,alpha)

print(theta)

print(sigmoid(np.dot(np.array([1,45,85]).T,theta)))