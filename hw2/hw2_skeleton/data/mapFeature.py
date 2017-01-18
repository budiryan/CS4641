
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.

    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial

    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    import numpy as np
    degree = 6
    result = np.ones(shape=(x1.size, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = np.power(x1, i - j) * np.power(x2, j)
            result = np.c_[result, temp]

    # delete the column of ones
    result = np.delete(result,0,axis=1)
    return result
