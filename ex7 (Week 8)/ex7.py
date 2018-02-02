# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 08:20:18 2018

@author: Sudha
"""

import scipy.io
import numpy as np

data = scipy.io.loadmat("ex7data2.mat")

X = np.array(data['X'])

dist = np.zeros((3, 300))

def distance(cent):
    return np.power(np.sum(np.power((X - cent), 2), axis = 1), 0.5).reshape(300, 1)

def findClosestCentroid(centroids):
    for i in range(len(centroids)):
        centroids_i = np.repeat( centroids[i], 300, axis = 0).reshape(2, 300).T
        dist[i] = distance(centroids_i).T
    return np.argmin(dist.T, axis = 1)
  
#closest_centroid = findClosestCentroid()

def computeCentroids(closest_centroid):
    count = 0
    new_centroids = np.zeros((len(closest_centroid), 2))
    for i in range(len(closest_centroid)):
        index_i = np.where(closest_centroid == i)[0]
        if ( len(index_i) != 0):
            new_centroids[i] = np.sum(X[index_i])/len(index_i)
            ++count
        else:
            new_centroids = np.delete(new_centroids, count, 0)
    return new_centroids

#new_centroids = computeCentroids(closest_centroid)

def iterateKmeans(ite):
    centroids = np.random.uniform(low=0.0, high=9.0, size=(3,2))
    print(centroids)
    for i in range(0,ite):
        closest_centroid = findClosestCentroid(centroids)
        new_centroids = computeCentroids(closest_centroid)
        centroids = new_centroids
        #print(centroids)

iterateKmeans(100)