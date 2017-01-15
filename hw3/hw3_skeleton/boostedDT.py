'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.models = []
        self.err = np.zeros(numBoostingIters)
        self.alpha = np.zeros(numBoostingIters)
        for i in range(numBoostingIters):
            self.models.append(tree.DecisionTreeClassifier(max_depth=maxTreeDepth))

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
        # find the length of training set and init all weights
        self.weights = np.ones(len(X))
        self.weights *= (1 / float(len(X)))
        self.y_list = list(set(y))
        self.y_list.sort()
        self.K = len(self.y_list)


        for i in range(self.numBoostingIters):
            self.models[i].fit(X=X, y=y, sample_weight=self.weights)
            err_numerator = 0
            predictions = self.models[i].predict(X)
            for j in range(len(self.weights)):
                if y[j] != predictions[j]:
                    err_numerator += self.weights[j]

            self.err[i] = err_numerator / float(np.sum(self.weights))

            self.alpha[i] = 0.5 * (np.log((1 - self.err[i]) / float(self.err[i])) + np.log(self.K - 1))

            for j in range(len(self.weights)):
                if y[j] != predictions[j]:
                    self.weights[j] *= np.exp(self.alpha[i])

            # normalization part..
            self.weights /= np.sum(self.weights)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # TODO
        # Initialization part, use dict to record all the predictions given by each classifier
        result_dict = []
        predictions = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            temp_dict = {}
            for j in range(self.K):
                temp_dict[self.y_list[j]] = 0
            result_dict.append(temp_dict)
        for i in range(self.numBoostingIters):
            temp_prediction = self.models[i].predict(X)
            for row_index, row in enumerate(temp_prediction):
                # If the label exists in y_list:
                for y in self.y_list:
                    if row == y:
                        result_dict[row_index][y] += self.alpha[i]
        # Get the max label for each row of prediction
        for i in range(len(X)):
            predictions[i] = max(result_dict[i], key=result_dict[i].get)

        return predictions
