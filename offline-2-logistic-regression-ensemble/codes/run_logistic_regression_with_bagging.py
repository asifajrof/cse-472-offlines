"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score
import sys

alpha = 0.02
max_iter = 1000
n_estimator = 9

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run_logistic_regression_with_bagging.py <csv_path>')
        sys.exit(1)
    csv_path = sys.argv[1]

    # data load
    X, y = load_dataset(csv_path=csv_path)

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    base_estimator = LogisticRegression(
        alpha=alpha, max_iter=max_iter)
    classifier = BaggingClassifier(
        base_estimator=base_estimator, n_estimator=n_estimator)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # print("y_pred: ", y_pred)
    # print("y_test: ", y_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
