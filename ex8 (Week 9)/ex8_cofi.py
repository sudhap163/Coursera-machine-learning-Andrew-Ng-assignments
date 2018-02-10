# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:37:19 2018

@author: Sudha
"""

import scipy.io
import numpy as np

data = scipy.io.loadmat('ex8_movies.mat')

Y = np.array(data['Y'])
R = np.array(data['R'])

