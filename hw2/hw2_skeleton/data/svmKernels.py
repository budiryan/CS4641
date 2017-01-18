"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    result = X1.dot(X2.T)
    result = np.power((result + 1), _polyDegree)
    return result


def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n, d = X2.shape
    result = np.zeros((X1.shape[0], X2.shape[0]))
    # Vectorized Solution
    for i in range(n):
        result[:, i] = np.sum(np.power((X1 - X2[i, :]), 2), axis=1)
    return np.exp(- result / (2 * np.power(_gaussSigma, 2)))
