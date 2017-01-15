from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
import time
import numpy as np
import logging


def print_table(accuracy, precision, recall, training_time):
    # Now, print the goddamn table
    headers = ["Metric", "Train", "Test"]
    digits = 3
    fmt = ''.join(['% 15s' for _ in headers])
    fmt += '\n'
    report = fmt % tuple(headers)
    report += '\n'

    values = ['Accuracy']
    for data in accuracy:
        values += ["{0:0.{1}f}".format(data, digits)]
    report += fmt % tuple(values)

    values = ['Precision']
    for data in precision:
        values += ["{0:0.{1}f}".format(data, digits)]
    report += fmt % tuple(values)

    values = ['Recall']
    for data in recall:
        values += ["{0:0.{1}f}".format(data, digits)]
    report += fmt % tuple(values)

    values = ['Training time']
    values += ["{0:0.{1}f}s".format(training_time, digits)]
    values += ["{0:0.{1}f}s".format(training_time, digits)]
    report += fmt % tuple(values)

    print(report)


if __name__ == '__main__':
    logging.basicConfig()
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

    # Naive Bayes classifier
    nb_clf = Pipeline([('vect', CountVectorizer(stop_words='english', lowercase=True)),
                       ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
                       ('clf', MultinomialNB())
                       ])

    nb_accuracy = []
    nb_precision = []
    nb_recall = []
    start_time = time.time()
    nb_clf = nb_clf.fit(twenty_train.data, twenty_train.target)
    nb_training_time = time.time() - start_time

    # predict for train data
    predicted_train_nb = nb_clf.predict(twenty_train.data)
    p, r, f1, s = precision_recall_fscore_support(twenty_train.target, predicted_train_nb)
    nb_accuracy.append(np.mean(twenty_train.target == predicted_train_nb))
    nb_precision.append(np.mean(p))
    nb_recall.append(np.mean(r))

    # predict for test data
    predicted_test_nb = nb_clf.predict(twenty_test.data)
    p, r, f1, s = precision_recall_fscore_support(twenty_test.target, predicted_test_nb)
    nb_accuracy.append(np.mean(twenty_test.target == predicted_test_nb))
    nb_precision.append(np.mean(p))
    nb_recall.append(np.mean(r))


    # Cosine similarity SVM
    sgd_clf = Pipeline([('vect', CountVectorizer(stop_words='english', lowercase=True)),
                        ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
                        ('clf', svm.SVC(C=1.0, kernel=cosine_similarity)),
                        ])
    start_time = time.time()
    sgd_clf = sgd_clf.fit(twenty_train.data, twenty_train.target)
    sgd_training_time = time.time() - start_time
    sgd_accuracy = []
    sgd_precision = []
    sgd_recall = []

    # predict for train data
    predicted_train_sgd = sgd_clf.predict(twenty_train.data)
    p, r, f1, s = precision_recall_fscore_support(twenty_train.target, predicted_train_sgd)
    sgd_accuracy.append(np.mean(twenty_train.target == predicted_train_sgd))
    sgd_precision.append(np.mean(p))
    sgd_recall.append(np.mean(r))

    # predict for test data
    predicted_test_sgd = sgd_clf.predict(twenty_test.data)
    p, r, f1, s = precision_recall_fscore_support(twenty_test.target, predicted_test_sgd)
    sgd_accuracy.append(np.mean(twenty_test.target == predicted_test_sgd))
    sgd_precision.append(np.mean(p))
    sgd_recall.append(np.mean(r))

    print('CLASSIFICATION REPORT FOR NAIVE BAYES CLASSIFIER: \n')
    print_table(nb_accuracy, nb_precision, nb_recall, nb_training_time)
    print('\n')
    print('CLASSIFICATION REPORT FOR SVM CLASSIFIER WITH COSINE SIMILARITY KERNEL: \n')
    print_table(sgd_accuracy, sgd_precision, sgd_recall, sgd_training_time)
