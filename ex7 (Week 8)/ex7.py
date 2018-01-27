# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 08:20:18 2018

@author: Sudha
"""

import scipy.io
import numpy as np

data = scipy.io.loadmat("ex7data2.mat")

X = np.array(data['X'])
'''
for i in range(2, 10):
    centroids = np.random.uniform(low=0.0, high=9.0, size=(i,2))
    print(centroids)
#    distance = math.sqrt()
'''

dist = np.zeros((3, 300))

def distance(cent):
    return np.power( np.sum(np.power((X-cent), 2), axis = 1), 0.5).reshape(300,1)

def findClosestCentroid():
    centroids = np.random.uniform(low=0.0, high=9.0, size=(3,2))
    for i in range(3):
        centroids_i = np.repeat( centroids[i], 300, axis = 0).reshape(2, 300).T
        distance(centroids_i)
        dist[i] = distance(centroids_i).T
  
findClosestCentroid()
print((np.argmin(dist.T, axis = 1).reshape(300,1))