# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:57:37 2018

@author: Sudha
"""

import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat("ex4data1.mat")
weights = scipy.io.loadmat("ex4weights.mat")

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

# setting up

X = np.hstack((np.ones((5000,1)), X.reshape(5000,400)))

theta1 = weights["Theta1"].reshape(25, 401)
theta2 = weights["Theta2"].reshape(10, 26)

Yk = np.empty([5000, 10])

for i in range(0,5000):
    for j in range (0,10):
        if (Y[i] == (j+1)):
            Yk[i][j] = 1
        else:
            Yk[i][j] = 0

# sigmoid function

def sigmoid(z):
    return (1/(1 + np.exp(-z)))
    
# feedforward propagation

n1 = np.dot(X, theta1.T)                    # 5000x25
a1 = sigmoid(n1)                            # 5000x25

a1 = np.hstack((np.ones((5000,1)), a1))     # 5000x26

n2 = np.dot(a1, theta2.T)                   # 5000x10
a2 = sigmoid(n2)

# cost function 

cost_vector = -(1/5000)*np.sum(np.sum(np.multiply(Yk, np.log(a2)) + np.multiply((1-Yk), np.log(1-a2))))

cost = 0

for i in range(0, 5000):
    for j in range(0, 10):
        cost = cost + Yk[i][j] *  math.log(a2[i][j]) + (1 - Yk[i][j]) * math.log(1 - a2[i][j])

cost = -cost/5000

# regularised cost function

theta1_sum = 0
theta2_sum = 0

for i in range(1, 25):
    for j in range(1, 400):
        theta1_sum += theta1[i][j]

for i in range(1, 10):
    for j in range(1, 25):
        theta2_sum += theta2[i][j]

cost_reg = cost + (1/(2*5000)) * (theta1_sum + theta2_sum)

# sigmoid gradient function

def sigmoidGradient(z):
    return (sigmoid(z)*(1 - sigmoid(z)))
    
# random initialization

theta1 = np.random.uniform(low=-0.12, high=0.12, size=(26, 401))
theta2 = np.random.uniform(low=-0.12, high=0.12, size=(10, 26))

# backpropagation

delta1 = np.zeros((26,401))
delta2 = np.zeros((10, 26))

ITR=500
alpha=0.1
for j in range(0,ITR):
    for i in range(0, 5000):
        a1 = X[i].reshape(401, 1)
        z2 = np.dot(theta1, a1) #26,1
        a2 = sigmoid(z2) #26,1
        
        #a2 = np.vstack((np.ones((1,1)), a2.reshape(25,1)))
        z3 = np.dot(theta2, a2) #10,1
        a3 = sigmoid(z3) #10,1
        
        error3 = a3 - Yk[i].reshape(10,1) #10,1
        error2 = np.multiply(np.dot(theta2.T, error3), sigmoidGradient(z2)) #26,1
        
        delta1 += np.dot(error2, a1.T)
        delta2 += np.dot(error3, a2.T)
    delta1 /=5000.0
    delta2 /=5000.0
    
    theta1 -= delta1*alpha
    theta2 -= delta2*alpha
    print(j)
    
# costfunction after finding thetas

n1 = np.dot(X, theta1.T)                    # 5000x26
a1 = sigmoid(n1)                            # 5000x26

n2 = np.dot(a1, theta2.T)                   # 5000x10
a2 = sigmoid(n2)

newY = (np.argmax(a2, axis = 1) + 1).reshape(5000,1)

error = Y - newY

accuracy = ((5000 - np.nonzero(error)[0].shape[0])/5000)*100
            
print ('Accuracy: ', accuracy)
    

