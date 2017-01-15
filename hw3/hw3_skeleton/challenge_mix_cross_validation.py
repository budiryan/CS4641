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

df = pd.read_csv('data/challengeTrainLabeled.dat', header=None)
n, d = df.shape

kf = cross_validation.KFold(n, n_folds=10, random_state=1)


# For SVM
C = 6.0
_gaussSigma = 2.5
equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)

algorithms = {
    # 'BoostedDT': AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=1000),
    'NearestNeighbor': KNeighborsClassifier(n_neighbors=1),
    'SVMGaussian': svm.SVC(C=C, kernel='rbf', gamma=equivalentGamma),
}

accuracy = 0
predictions = {
    # 'BoostedDT': list(),
    'NearestNeighbor': list(),
    'SVMGaussian': list(),
}

count = 1
for train, test in kf:
    # Get the training data
    train_df_X = df[predictors].iloc[train, :]
    train_df_y = df[label].iloc[train]
    # Get the test data
    test_df_X = df[predictors].iloc[test, :]
    test_df_y = df[label].iloc[test, :]

    # Training the algorithm using the predictors and target.
    for algo in algorithms:
        algorithms[algo].fit(np.array(train_df_X), np.array(train_df_y)[:, 0])
        test_prediction = algorithms[algo].predict(test_df_X)
        accuracy_now = accuracy_score(np.array(test_df_y), test_prediction)
        predictions[algo].append(accuracy_now)
        print 'Finished predicting for: ', algo
    print 'finished iteratiion: ', count
    print '\n'
    count += 1

for acc in predictions:
    print acc, 'mean accuracy for ', np.mean(predictions[acc])
    print acc, 'std accuracy for ', np.std(predictions[acc])
    print '\n'
