import numpy as np


class LogisticRegression:
    def __init__(self, alpha, max_iter, verbose=False):
        """
        figure out necessary params to take as input
        :param float alpha: learning rate
        :param int max_iter: maximum number of iterations
        """
        # todo: implement
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.theta = None
        if self.verbose:
            print("Logistic Regression initialized")
            print("alpha: ", self.alpha)
            print("max_iter: ", self.max_iter)
            print("theta: ", self.theta)

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        if self.verbose:
            print("Logistic Regression fit started")

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def theta_T_X(theta, X):
            return np.dot(theta, X)

        def h_theta(theta, X):
            h = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                h[i] = sigmoid(theta_T_X(theta, X[i]))
            return h

        def J_theta(theta, X, y):
            return np.mean(-y * np.log(h_theta(theta, X)) - (1 - y) * np.log(1 - h_theta(theta, X)))

        def gradient_descent(theta, X, y):
            min_cost = 1e10

            for iter in range(self.max_iter):
                theta = theta - self.alpha * np.dot(
                    X.T, (h_theta(theta, X) - y)) / X.shape[0]
                cost = J_theta(theta, X, y)
                if cost <= min_cost:
                    min_cost = cost
                    self.theta = theta
                else:
                    pass
                if self.verbose:
                    print("iter: ", iter, "cost: ", cost)

        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        # self.theta = np.random.rand(X.shape[1])
        self.theta = np.full(X.shape[1], 0.5)
        gradient_descent(self.theta, X, y)
        if self.verbose:
            print("Logistic Regression fit finished")
            print("theta: ", self.theta)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        def sigmoid(x):
            sigm = 1 / (1 + np.exp(-x))
            # return 0 or 1
            return 1 if sigm >= 0.5 else 0

        def theta_T_X(theta, X):
            return np.dot(theta, X)

        def h_theta(theta, X):
            h = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                h[i] = sigmoid(theta_T_X(theta, X[i]))
            return h.astype(int)
        return h_theta(self.theta, X)
