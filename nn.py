# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:17:32 2019

@author: Karim Bouchekoura
"""

import numpy as np

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
    
    
alpha = 2
nb_neurones_1 = 16
nb_neurones_2 = 16
nb_neurones_3 = 2


# inputs
X = np.array([  
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])
    
y = np.array([[0,1], [1,1],[0,0],[1,0],[0,1],[1,1]])
print("This is the datas the neural network algorithm is trained on")

print("X",' => ',"y")
strx = ''
for i in range(0,len(X)):
    r= X[i]
    print(r,' => ',y[i])

# Initialy, weights are choosen arbitrarily
W2 = 2*np.random.random((nb_neurones_1,X.shape[1] + 1)) - 1
W3 = 2*np.random.random((nb_neurones_2,nb_neurones_1 + 1)) - 1
W4 = 2*np.random.random((nb_neurones_3,nb_neurones_2 + 1)) - 1
W5 = 2*np.random.random((y.shape[1],nb_neurones_3 + 1)) - 1

nb_iteration = 1000

L2 = []

# X_tilde is basicaly the same as X but we add a line of ones to consider bias as any other weight
# maybe shoud X_tilde be named A1_tilde
X_tilde = np.hstack((np.ones((X.shape[0], 1)), X))


for i in range(nb_iteration):
    
    # Forward propagation
    # let's apply the weights to the inputs (the whole dataset is in matrix X_tilde)
    # A2 contains the values of the first hidden layer
    A2 = sigmoid(np.dot(X_tilde,W2.T))
    
    # prepare first layer 
    A2_tilde = np.hstack((np.ones((A2.shape[0], 1)), A2))
    # A3 contains the values of the second hidden layer
    A3 = sigmoid(np.dot(A2_tilde,W3.T))

    # prepare second layer 
    A3_tilde = np.hstack((np.ones((A3.shape[0], 1)), A3))
    # A4 contains the values of the third hidden layer
    A4 = sigmoid(np.dot(A3_tilde,W4.T))

    # prepare third layer 
    A4_tilde = np.hstack((np.ones((A4.shape[0], 1)), A4))
    # prepare the output of the neural network 
    # we want A5 to be close to the y values
    A5 = sigmoid(np.dot(A4_tilde,W5.T))
    
    # so let's calculate the error between A5 and y
    # L is ||A5 -y||² where ||.|| is L2 Euclidian Norm
    # could also have been the Entropy Function
    L =np.dot((A5 -y).T, (A5 -y)).sum()
    # store the error in this epoch
    L2.insert(i,L) 
    
    # Back propagation
    # let's calculate the error of each layer
    Delta5 = (A5 - y)*sigmoid(A5,True)
    Delta4 = np.dot(Delta5,W5[:, 1:])*sigmoid(A4,True) 
    Delta3 = np.dot(Delta4,W4[:, 1:])*sigmoid(A3,True) 
    Delta2 = np.dot(Delta3,W3[:, 1:])*sigmoid(A2,True) 

    # let's calculate the gradient of each the weights matrixes
    W5_dp_all = Delta5[:, :, np.newaxis] * A4_tilde[:, np.newaxis, :]
    W5_dp = np.average(W5_dp_all, axis = 0)

    W4_dp_all = Delta4[:, :, np.newaxis] * A3_tilde[:, np.newaxis, :]
    W4_dp = np.average(W4_dp_all, axis = 0)

    W3_dp_all = Delta3[:, :, np.newaxis] * A2_tilde[:, np.newaxis, :]
    W3_dp = np.average(W3_dp_all, axis = 0)

    W2_dp_all = Delta2[:, :, np.newaxis] * X_tilde[:, np.newaxis, :]
    W2_dp = np.average(W2_dp_all, axis = 0)

    # correct the weights of alpha*gradient
    W5 = W5 - alpha*W5_dp
    W4 = W4 - alpha*W4_dp
    W3 = W3 - alpha*W3_dp
    W2 = W2 - alpha*W2_dp
    # let's hope those weights will make A5 close to y
    # actually the loop should stop when the difference between two error values is smaller than 10^(-3) or whatever condition saying this has converged


print("This is what the neural network outputs after training: \n{}".format(A5)+"\n(should be close to y)\n\n")


print("And this is a plot of the quadratic error function for each iteration")
import matplotlib.pyplot as plt
plt.xlabel('Number of iterations')
plt.ylabel('L2 error')
plt.title('Neural network quadratic error')
plt.plot(L2);
