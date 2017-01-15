'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.not_found = 'dne'


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        self.y_list = np.array(list(set(y)))
        self.y_list.sort()
        self.y_probabilities = np.zeros(len(self.y_list))

        for elem in y:
            if elem in self.y_list:
                self.y_probabilities[elem] += 1

        self.y_probabilities /= float(len(y))
        self.y_probabilities = self.y_probabilities

        self.conditional_probabilities = np.empty(shape=(len(self.y_list), X.shape[1]), dtype=dict)

        for label in self.y_list:
            indexes = [i for i, x in enumerate(y) if x == label]
            for index_cell, cell in enumerate(self.conditional_probabilities[label]):
                for index in indexes:
                    if self.conditional_probabilities[label][index_cell] is None:
                        self.conditional_probabilities[label][index_cell] = {}
                        self.conditional_probabilities[label][index_cell][X[index][index_cell]] = 1
                    elif not X[index][index_cell] in self.conditional_probabilities[label][index_cell]:
                        self.conditional_probabilities[label][index_cell][X[index][index_cell]] = 1
                    else:
                        self.conditional_probabilities[label][index_cell][X[index][index_cell]] += 1

                if self.useLaplaceSmoothing:
                    self.conditional_probabilities[label][index_cell][self.not_found] = 0
                else:
                    # np.log(0) is not possible
                    self.conditional_probabilities[label][index_cell][self.not_found] = 1.0 / (len(indexes) + len(self.conditional_probabilities[label][index_cell].keys()))

                for key, value in self.conditional_probabilities[label][index_cell].iteritems():
                    if self.useLaplaceSmoothing:
                        self.conditional_probabilities[label][index_cell][key] = (value + 1) / float(len(indexes) + len(self.conditional_probabilities[label][index_cell].keys()))
                        self.conditional_probabilities[label][index_cell][key] = self.conditional_probabilities[label][index_cell][key]
                    else:
                        self.conditional_probabilities[label][index_cell][key] = value / float(len(indexes))

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        predictions = np.zeros(X.shape[0])
        # Get the row
        for x_row_index, x_row in enumerate(X):
            temp_list = np.zeros(len(self.y_list))
            # Analyze every cell
            for class_row_index, class_row in enumerate(self.conditional_probabilities):
                temp_list[class_row_index] += np.log(self.y_probabilities[class_row_index])
                for x_cell_index, x_cell in enumerate(x_row):
                    if x_cell in class_row[x_cell_index]:
                        temp_list[class_row_index] += np.log(class_row[x_cell_index][x_cell])
                    else:
                        temp_list[class_row_index] += np.log(class_row[x_cell_index][self.not_found])

            prediction = max(temp_list)
            prediction = int(np.where(temp_list == prediction)[0])
            predictions[x_row_index] = prediction
        return predictions

    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        predictions = np.zeros(shape=(X.shape[0], len(self.y_list)))
        # Get the row
        for x_row_index, x_row in enumerate(X):
            temp_list = np.zeros(len(self.y_list))
            # Analyze every cell
            for class_row_index, class_row in enumerate(self.conditional_probabilities):
                temp_list[class_row_index] += np.log(self.y_probabilities[class_row_index])
                for x_cell_index, x_cell in enumerate(x_row):
                    if x_cell in class_row[x_cell_index]:
                        temp_list[class_row_index] += np.log(class_row[x_cell_index][x_cell])
                    else:
                        temp_list[class_row_index] += np.log(class_row[x_cell_index][self.not_found])
            predictions[x_row_index] = temp_list
        predictions = np.exp(predictions)
        return predictions
