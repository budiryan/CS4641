from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import numpy as np
import time
import logging


"""
Requirements:
- Lowercase all terms? Yes
- Remove stop words? Yes
- Only fit the train data, not test data? Yes
- Use log-scaled term counts for term frequency? Yes
- Normalize the TF-IDF feature vectors? Yes
"""


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
    accuracy = []
    precision = []
    recall = []

    # pre-process training data before proceeding to algorithm
    count_vect = CountVectorizer(stop_words='english', lowercase=True)
    tf_transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)

    # Only transform the train data, make predictions based on training data
    start_time = time.time()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    train_time = time.time() - start_time
    predictions_train = clf.predict(X_train_tfidf)
    p, r, f1, s = precision_recall_fscore_support(twenty_train.target, predictions_train)
    accuracy.append(accuracy_score(twenty_train.target, predictions_train))
    precision.append(np.mean(p))
    recall.append(np.mean(r))

    # DO NOT fit the test data, only transform, then make predictions based on test data
    X_test_counts = count_vect.transform(twenty_test.data)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    predictions_test = clf.predict(X_test_tfidf)
    p, r, f1, s = precision_recall_fscore_support(twenty_test.target, predictions_test)
    accuracy.append(accuracy_score(twenty_test.target, predictions_test))
    precision.append(np.mean(p))
    recall.append(np.average(r))

    # Output another table of statistics
    print_table(accuracy, precision, recall, train_time)
