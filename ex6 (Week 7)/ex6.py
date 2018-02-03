# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:49:14 2018

@author: Sudha
"""

import scipy.io
import numpy as np
import pandas as pd
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

def gaussianKernel( U, V, sigma = 0.3 ):
    return np.exp((-1/2*sigma*sigma)*(np.sum(np.power((U-V),2))))

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

# finding gram matrix

def gram(U,V,sigma=0.1):
    G = np.zeros((U.shape[0], V.shape[0]))
    for i in range(0,U.shape[0]):
        for j in range(0,V.shape[0]):
            G[i][j] = gaussianKernel(U[i],V[j],sigma)
    return G

C = s = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

ACC = []
max_acc_value = 0
max_acc_index = np.array([0, 0])

for i in range(0,8):
    a = []
    for j in range(0,8):
        clf = svm.SVC(C = C[i], kernel = "precomputed")
        clf.fit(gram(X,X,s[j]),Y.reshape(len(Y),))
        yTest = clf.predict(np.dot(Xval,X.T))
        e = yTest - Yval.reshape(len(Yval))
        acc = e.shape[0] - np.count_nonzero(e)
        acc = (acc*100)/e.shape[0]
        if (max_acc_value < acc):
            max_acc_value = acc
            max_acc_index = np.array([i,j]).reshape(1,1)
        a.append(acc)
        print (j, max_acc_value)
    ACC.append(a)
    
print('Highest value of C : ', C[max_acc_index[0][0]], ' and sigma : ', s[max_acc_index[0][1]])

# plt.plot(s,ACC[0],s,ACC[1],s,ACC[2],s,ACC[3],s,ACC[4],s,ACC[5],s,ACC[6],s,ACC[7])

# Spam classifier

# loading files

vocab = np.array(pd.read_csv('vocab.csv'))
words = vocab[:, 1]

spam_data = scipy.io.loadmat('spamTrain.mat')
X = spam_data["X"]
Y = spam_data["y"]

test_data = scipy.io.loadmat('spamTest.mat')
Xtest = test_data["Xtest"]
Ytest = test_data["ytest"]

clf = svm.SVC()
clf.fit(X, Y.reshape(len(Y),))

acc_train = (X.shape[0] - np.count_nonzero( clf.predict(X).reshape(len(X), 1) - Y ))*100/len(X)

acc_test = (Xtest.shape[0] - np.count_nonzero( clf.predict(Xtest).reshape(len(Xtest), 1) - Ytest ))*100/len(Xtest)

print("Accuracy of training set : ", acc_train)
print("Accuracy of test set : ", acc_test)