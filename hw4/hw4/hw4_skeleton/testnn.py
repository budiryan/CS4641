import numpy as np
import pickle
from nn import NeuralNet
from sklearn.metrics import accuracy_score

learningRate = 2
numEpochs = 1300

# load the data set
filenameX = 'data/digitsX.dat'
filenameY = 'data/digitsY.dat'
X = np.loadtxt(filenameX, delimiter=',')
y = np.loadtxt(filenameY, delimiter=',')
n, d = X.shape

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)

X = X[idx]
y = y[idx]


def train(X, y):
    # train the neuralNet
    neuralNet = NeuralNet(layers=np.array([25]), learningRate=learningRate, numEpochs=numEpochs)
    print 'Training using: '
    print 'learningRate: ', learningRate
    print 'numEpochs: ', numEpochs
    neuralNet.fit(X, y)
    return neuralNet


def predict(model, X):
    return model.predict(X)


clf = None
try:
    with open('clf.pickle1300', 'rb') as f:
        clf = pickle.load(f)[0]
except:
    pass

if not clf:
    clf = train(X, y)
    with open('clf.pickle1300', 'wb') as f:
        pickle.dump([clf], f)

# output predictions on the remaining data
ypred_NN = predict(clf, X)

# test
clf.visualizeHiddenNodes('hiddenVisualization.png')


accuracyNN = accuracy_score(y, ypred_NN)
print "Neural Net Accuracy = " + str(accuracyNN)
