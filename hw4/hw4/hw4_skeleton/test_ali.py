from nn import NeuralNet
import numpy as numpy
from numpy import loadtxt

filenameX = 'data/digitsX.dat'
filenameY = 'data/digitsY.dat'
dataX = loadtxt(filenameX, delimiter=',')
X = dataX
dataY = loadtxt(filenameY, delimiter=',')
y = dataY

nn = NeuralNet([2, 3], learningRate=0.1)
nn.fit(X,y)

