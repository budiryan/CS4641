import pandas as pd
import numpy as np
from boostedDT import BoostedDT
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


predictors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
label = [10]

df_train = pd.read_csv('data/challengeTrainLabeled.dat', header=None)
df_test = pd.read_csv('data/challengeTestUnlabeled.dat', header=None)
train_df_X = df_train[predictors]
train_df_y = df_train[label]
n, d = df_train.shape

kf = cross_validation.KFold(n, n_folds=10, random_state=1)

# For SVM
C = 6.0
_gaussSigma = 2.5
equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)

# For SVM
algorithms = {
    # 'NearestNeighbor': KNeighborsClassifier(n_neighbors=1),
    'SVMGaussian': svm.SVC(C=C, kernel='rbf', gamma=equivalentGamma),
}

accuracy = 0
predictions = {
    # 'NearestNeighbor':list(),
    'SVMGaussian':list(),
}

# Training the algorithm using the predictors and target.
for algo in algorithms:
    algorithms[algo].fit(np.array(train_df_X), np.array(train_df_y)[:, 0])
    test_prediction = algorithms[algo].predict(df_test)
    test_prediction = map(int, test_prediction)
    np.savetxt('predictions-BestClassifier.dat', test_prediction, delimiter=',')
