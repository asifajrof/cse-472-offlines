from data_handler import bagging_sampler
import numpy as np
import copy


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator: base estimator -> logistic regression here
        :param n_estimator: number of estimators
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        for i in range(self.n_estimator):
            print(f"fitting estimator {i+1}/{self.n_estimator}...")
            X_sample, y_sample = bagging_sampler(X, y)
            self.base_estimator.fit(X_sample, y_sample)
            self.estimators.append(copy.deepcopy(self.base_estimator))

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        # majority voting
        y_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        return y_pred
