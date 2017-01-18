'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import numpy as np


class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLamda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hasConverged(self, old_theta, new_theta):
        result = np.power(old_theta - new_theta, 2)
        result = np.sum(result)
        result = np.sqrt(result)
        return result < self.epsilon

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        # n is number of training examples
        # d is number of features
        n, d = X.shape
        if not isinstance(theta, np.ndarray):
            theta = np.matrix(np.zeros((d, 1)))
        result = (-y.transpose() * np.log(self.sigmoid(X * theta))) - ((1 - y.transpose()) * np.log(1 - self.sigmoid(X * theta)))
        theta_distance = np.power(theta, 2)
        theta_distance = (np.sum(theta_distance) * regLambda) / 2
        result += theta_distance
        return result.flat[0]

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:a
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n, d = X.shape
        i = 0  # counter

        done_converging = False

        while (i < self.maxNumIters) and not done_converging:
            # using matrix operation
            updated_theta = theta.copy()
            updated_theta = theta - (X.transpose() * (self.sigmoid(X * theta) - y)) * (self.alpha)

            # regularization part, have to remember not to regularize the first one
            for j in range(1, d):
                updated_theta[j] -= (self.alpha * theta[j] * regLambda)
            i += 1
            done_converging = self.hasConverged(updated_theta, theta)
            theta = updated_theta.copy()
        return updated_theta

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        X = np.c_[np.ones([len(X), 1]), X]
        n, d = X.shape
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d, 1)))
        self.theta = self.computeGradient(self.theta, X, y, self.regLamda)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        X = np.c_[np.ones([len(X), 1]), X]
        prediction = np.array(self.sigmoid(X * self.theta))
        for i in range(len(prediction)):
            prediction[i] = 1 if prediction[i] >= 0.5 else 0    
        return prediction
