'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    results = []
    # Try for 100 times
    for i in range(100):
        # split the data
        Xtrain = X[1:101,:]  # train on first 100 instances
        Xtest = X[101:,:]
        ytrain = y[1:101,:]  # test on remaining instances
        ytest = y[101:,:]

        num_of_folds = 10
        length_of_each_fold = len(y) / num_of_folds
        for i in range(num_of_folds):
            
            # Generate the list with 10 folds cross-validations
            Xtest = X[i * length_of_each_fold:][:length_of_each_fold]
            ytest = y[i * length_of_each_fold:][:length_of_each_fold]
            Xtrain = []
            Xtrain.extend(X[:length_of_each_fold * i])
            Xtrain.extend(X[length_of_each_fold *(i + 1):])
            ytrain = []
            ytrain.extend(y[:length_of_each_fold * i])
            ytrain.extend(y[length_of_each_fold *(i + 1):])

            
            # train the decision tree
            clf = tree.DecisionTreeClassifier(max_depth=1)
            clf = clf.fit(Xtrain,ytrain)

            # output predictions on the remaining data
            y_pred = clf.predict(Xtest)

            # compute the training accuracy of the model
            meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
            results.append(meanDecisionTreeAccuracy)
    
    print 'mean:', np.mean(results)
    print 'Stdev: ', np.std(results, axis=0)
    meanDecisionTreeAccuracy = 0.732653846154
    # TODO: update these statistics based on the results of your experiment
    stddevDecisionTreeAccuracy = 0.0709883626599
    meanDecisionStumpAccuracy = 0.792307692308
    stddevDecisionStumpAccuracy = 0.099108451744
    meanDT3Accuracy = 0.753461538462
    stddevDT3Accuracy = 0.0627041046003

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
