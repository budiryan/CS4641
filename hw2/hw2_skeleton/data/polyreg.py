'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda
        self.mean = list()
        self.std = list()


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        # The n here is number of training examples
        Xex = X.copy()
        if degree == 1:
            Xex = Xex.reshape((len(Xex), 1))
        else:
            for i in range(2, self.degree + 1):
                Xex = np.c_[Xex, np.power(X, i)]
        return Xex

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        # TODO
        Xex = self.polyfeatures(X, self.degree)

        # Standarization part
        Xex_t = Xex.T

        # Reset before training another one
        self.mean = list()
        self.std = list()

        # based on the transpose, apply standardization
        for i in range(len(Xex_t)):
            temp_mean = np.mean(Xex_t[i])
            self.mean.append(temp_mean)
            temp_std = np.std(Xex_t[i])
            self.std.append(temp_std)
            if len(Xex_t[i]) == 1:
                break
            for j in range(len(Xex_t[i])):
                Xex_t[i][j] -= temp_mean
                Xex_t[i][j] /= temp_std

        Xex = Xex_t.T

        Xex = np.c_[np.ones([len(Xex), 1]), Xex]

        n, d = Xex.shape
        d = d - 1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(Xex.T.dot(Xex) + regMatrix).dot(Xex.T).dot(y)

    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        # expand the polynomial
        Xex = self.polyfeatures(X, self.degree)
        
        # Standarization part, only do it if sample has length more than 1
        for sample in Xex:
            for index, feature in enumerate(sample):
                sample[index] -= self.mean[index]
                sample[index] /= self.std[index]

        Xex = np.c_[np.ones([len(Xex), 1]), Xex]
        # Predict now
        prediction = Xex.dot(self.theta.T)
        return prediction


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''

    n = len(Xtrain);

    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    model = PolynomialRegression(degree, regLambda)
    for i in range(1, n + 1):
        if i == 1:
            errorTrain[i - 1] = 0
            errorTest[i - 1] = 0
            continue
        model.fit(np.array(Xtrain[0:i]), np.array(Ytrain[0:i]))
        
        # intentionally leave errorTrain[0] and errorTest[0] will be 0
        # initialize variables to be zero
        test_error = 0
        training_error = 0

        # Get the training error
        training_error = np.power(model.predict(Xtrain[0:i]) - Ytrain[0:i], 2)
        training_error = np.sum(training_error) / len(training_error)
        errorTrain[i - 1] = training_error

        # Get the train error
        test_error = np.power(model.predict(Xtest) - Ytest, 2)
        test_error = np.sum(test_error) / len(test_error)
        errorTest[i - 1] = test_error
    return (errorTrain, errorTest)
