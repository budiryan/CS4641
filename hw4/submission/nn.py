'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from PIL import Image


class NeuralNet:

    def __init__(self, layers, learningRate, epsilon=0.12, numEpochs=100):
        '''
        Constructor
        Arguments:
                layers - a numpy array of L-2 integers (L is # layers in the network)
                epsilon - one half the interval around zero for setting the initial weights
                learningRate - the learning rate for backpropagation
                numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.weights = []  # containing the matrix theta
        self.reg_lambda = 0

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        self.shape = X.shape
        y_unique = np.unique(y)

        len_y_unique = len(y_unique) if len(y_unique) > 2 else 1

        # first, initialize the weight list
        # initialize first matrix:
        first_layer = (2 * self.epsilon * np.random.sample((self.layers[0], (self.shape[1] + 1)))) - self.epsilon
        self.weights.append(first_layer)

        # create matrix for hidden layer (if any)
        for i in xrange(len(self.layers) - 1):
            self.weights.append((2 * np.random.sample((self.layers[i + 1], (self.layers[i] + 1)))) - self.epsilon)

        # Append the last matrix
        self.weights.append((2 * self.epsilon * np.random.sample((len_y_unique, (self.layers[-1] + 1)))) - self.epsilon)

        # loop according to the number of epochs
        for i in xrange(self.numEpochs):
            D_matrix = []
            gradient_matrix = []
            for j in xrange(len(self.weights)):
                D_matrix.append(np.zeros(shape=(self.weights[j].shape)))
                gradient_matrix.append(np.zeros(shape=(self.weights[j].shape)))

            # For each training instance:
            for j in xrange(self.shape[0]):
                activation_nodes = self._forward_prop(X[j], self.weights)
                y_instance = np.zeros(len_y_unique)

                # Make the current occurence true
                y_instance[int(y[j])] = 1
                errors = []
                last_layer_errors = activation_nodes[-1] - y_instance
                errors.append(last_layer_errors)
                current_error = errors[-1]

                # reversely calculate each layer's error, omit the input layer
                for k in xrange(len(self.layers), 0, -1):
                    prod = np.dot(self.weights[k].T, current_error)
                    current_error = prod * self._g_prime(activation_nodes[k])
                    # insert the calculated error at the front of the list
                    errors.insert(0, current_error)
                    # delete bias unit from the error list for next multiplication
                    current_error = current_error[1:]

                # Compute the gradient
                for k in xrange(len(self.weights)):
                    # Need to delete first error, otherwise size will not match with gradient_matrix
                    errors[k] = errors[k][1:] if k != (len(self.weights) - 1) else errors[k][0:]
                    # Here we have already got the gradients
                    gradient_matrix[k] += np.outer(activation_nodes[k], errors[k]).T

            # Compute the D
            for l in xrange(len(D_matrix)):
                D_matrix[l][:, 0] = (1. / self.shape[0]) * (gradient_matrix[l][:, 0])
                D_matrix[l] = (1. / self.shape[0]) * (gradient_matrix[l]) + (self.reg_lambda * self.weights[l])

            # Update the weight
            for l in xrange(len(self.weights)):
                self.weights[l] = self.weights[l] - (self.learningRate * D_matrix[l])

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n, d = X.shape
        results = np.zeros(n)

        # an n-dimensional numpy array of the predictions
        for i in xrange(n):
            activation = self._forward_prop(X[i], self.weights)
            results[i] = np.argmax(activation[-1])

        return results

    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        # Note: I assume there is only 1 hidden layer

        weights = np.copy(self.weights[0][:, 1:])
        n, d = weights.shape
        size_of_each_image = int(np.sqrt(d))
        size_of_each_image_after_resize = 2 * size_of_each_image
        image_array = []
        for index, row in enumerate(weights):
            new_min, new_max = 0, 255
            old_min, old_max = np.amin(row), np.amax(row)
            for elem_index in range(len(row)):
                weights[index][elem_index] = (((weights[index][elem_index] - old_min) * new_max) / (old_max - old_min)) + new_min
            img = Image.fromarray(np.reshape(weights[index], newshape=(size_of_each_image, size_of_each_image)))
            img = img.resize((int(size_of_each_image_after_resize), int(size_of_each_image_after_resize)))
            image_array.append(img)

        # already have a list of small images, now have to concatenate all of them
        final_img_height = image_array[0].size[0] * len(image_array)
        final_img_width = image_array[1].size[1] * len(image_array)
        img_height = image_array[0].size[0]
        img_width = image_array[0].size[1]

        final_image = Image.new('L', (final_img_height, final_img_width))

        col_count = 0
        row_count = 0
        for image in image_array:
            final_image.paste(image, (col_count * img_height + 400, row_count * img_width + 400))
            row_count += 1
            if row_count % int(np.sqrt(n)) == 0:
                col_count += 1
                row_count = 0

        final_image.show()
        final_image.save(filename)

    def _sigmoid(self, X):
        '''
        Copied from HW2, very useful
        '''
        return 1 / (1 + np.exp(-X))

    def _g_prime(self, X):
        '''
        Computes g prime used in calculating small delta for each layer
        '''
        return X * (1 - X)

    def _calculate_cost(self, h, y):
        '''
        Compute the cost of neural nets
        '''
        return -1 * np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h)))

    def _forward_prop(self, X, theta):
        '''
        Outputs the forward propagation given input X and its associated weights
        '''
        x = np.copy(X)
        # must add a bias unit
        x = np.insert(x, 0, 1)

        result = []
        result.append(x)

        for i in xrange(len(theta)):
            x = self._sigmoid(np.dot(theta[i], x))

            # no need to add 1 in the result
            if i != len(theta) - 1:
                x = np.insert(x, 0, 1)

            result.append(x)

        return result
