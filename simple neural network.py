# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:44:37 2018

@author: K
"""
import numpy as np
import time
import scipy.io
import math
#import scipy as sp

#import matplotlib.animation as animation
import matplotlib.pyplot as plt

#plt.style.use('dark_background')
#plt.figure(figsize = (8,5))



# define the sigmoid function
def sigmoid(x, derivative=False):

    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
    
    
alpha = 1
nb_neurones = 100


datas = scipy.io.loadmat('ex4data1.mat');
X = datas["X"]; 
y =  datas["y"];
#time.sleep(10)



# inputs
X = np.array([  
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])
    
y = np.array([[0, 1, 1, 1, 0, 1]]).T

W2 = 2*np.random.random((nb_neurones,X.shape[1] + 1)) - 1
W3 = 2*np.random.random((y.shape[1],nb_neurones + 1)) - 1

nb_iteration = 5000

for i in range(nb_iteration):
    
    X_tilde = np.hstack((np.ones((X.shape[0], 1)), X))
    A2 = sigmoid(np.dot(X_tilde,W2.T))
    A2_tilde = np.hstack((np.ones((A2.shape[0], 1)), A2))
    A3 = sigmoid(np.dot(A2_tilde,W3.T))
    
    Delta3 = (A3 - y)*sigmoid(A3,True)
    
    #Delta2 = Delta3*sigmoid(A2,True) # ok
    Delta2 = np.dot(Delta3,W3[:, 1:])*sigmoid(A2,True) 
    
    print("Delta2.shape => ",Delta2.shape)
    print("Delta3.shape => ",Delta3.shape)
    
    W3_dp_all = Delta3[:, :, np.newaxis] * A2_tilde[:, np.newaxis, :]
    W3_dp = np.average(W3_dp_all, axis = 0)
    
    W2_dp_all = Delta2[:, :, np.newaxis] * X_tilde[:, np.newaxis, :]
    W2_dp = np.average(W2_dp_all, axis = 0)
    
    W2 = W2 - alpha*W2_dp
    W3 = W3 - alpha*W3_dp
    
print("Output After Training: \n{}".format(A3))
