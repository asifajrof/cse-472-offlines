"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score
import sys

alpha = 0.8
max_iter = 10000

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run_logistic_regression.py <csv_path>')
        sys.exit(1)
    csv_path = sys.argv[1]

    # data load
    X, y = load_dataset(csv_path=csv_path)

    # split train and test
    X_train, y_train, X_test1, y_test1, X_test2, y_test2 = split_dataset(
        X, y)  # 2 testing

    # training
    classifier = LogisticRegression(
        alpha=alpha, max_iter=max_iter)
    classifier.fit(X_train, y_train)

    # testing
    y_pred1 = classifier.predict(X_test1)
    y_pred2 = classifier.predict(X_test2)   # 2 testing

    # print("y_pred: ", y_pred)
    # print("y_test: ", y_test)

    # performance on test set
    print("Test 1")
    print('Accuracy ', accuracy(y_true=y_test1, y_pred=y_pred1))
    print('Recall score ', recall_score(y_true=y_test1, y_pred=y_pred1))
    print('Precision score ', precision_score(y_true=y_test1, y_pred=y_pred1))
    print('F1 score ', f1_score(y_true=y_test1, y_pred=y_pred1))

    # 2 testing
    print("Test 2")
    print('Accuracy ', accuracy(y_true=y_test2, y_pred=y_pred2))
    print('Recall score ', recall_score(y_true=y_test2, y_pred=y_pred2))
    print('Precision score ', precision_score(y_true=y_test2, y_pred=y_pred2))
    print('F1 score ', f1_score(y_true=y_test2, y_pred=y_pred2))
