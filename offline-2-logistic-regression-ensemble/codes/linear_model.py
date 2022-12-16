import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, alpha, max_iter):
        """
        figure out necessary params to take as input
        :param float alpha: learning rate
        :param int max_iter: maximum number of iterations
        """
        # todo: implement
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = None
        # self.intial_theta_value = 0.5

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        def sigmoid(theta, X):
            return 1/(1 + np.exp(-X.dot(theta)))

        def J_theta(theta, X, y):
            m = X.shape[0]
            return np.sum(-y * np.log(sigmoid(theta, X)) - (1 - y) * np.log(1 - sigmoid(theta, X)))/m

        def gradient_descent(theta, X, y):
            min_cost = 1e10

            for iter in tqdm(range(self.max_iter)):
                m = X.shape[0]
                sigmx = sigmoid(theta, X)
                diff = sigmx - y
                theta = theta - self.alpha * X.T.dot(diff)/m
                cost = J_theta(theta, X, y)
                if cost <= min_cost:
                    min_cost = cost
                    self.theta = theta
                else:
                    pass

        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.theta = np.random.rand(X.shape[1])
        # self.theta = np.full(X.shape[1], self.intial_theta_value)
        gradient_descent(self.theta, X, y)
        print("Logistic Regression fit done")
        print("theta: ", self.theta)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        def sigmoid(theta, X):
            return 1/(1 + np.exp(-X.dot(theta)))

        sigmx = sigmoid(self.theta, X)
        y_pred = np.round(sigmx).astype(int)
        return y_pred
