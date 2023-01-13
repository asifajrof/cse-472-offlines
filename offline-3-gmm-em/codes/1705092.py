# gmm with em algorithm using numpy

import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

k_init = 1
k_final = 10
n_max_iter = 100


def load_dataset(file_path):
    dataset = np.loadtxt(file_path, delimiter=' ')
    return dataset


class GMM_EM:
    def __init__(self, k=1, max_iter=100, eps=1e-6):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.mu = None
        self.sigma = None
        self.w = None
        self.phi = None
        self.N = None
        self.D = None

    def model_initialize(self, dataset):
        self.N, self.D = dataset.shape
        self.mu = dataset[np.random.randint(0, self.N, size=self.k)]
        # dataset's sigma as all cluster's sigma + identity matrix
        cov = np.cov(dataset.T) + np.eye(self.D)
        self.sigma = np.full(shape=(self.k, self.D, self.D),
                             fill_value=cov)
        self.phi = np.full(shape=self.k, fill_value=1 / self.k)
        self.w = np.full(shape=(self.N, self.k), fill_value=1 / self.k)

    def E_step(self, dataset):
        N_pdf = np.zeros(shape=(self.N, self.k))
        for j in range(self.k):
            N_distr = multivariate_normal(
                mean=self.mu[j], cov=self.sigma[j], allow_singular=True)
            N_pdf[:, j] = N_distr.pdf(dataset)
        numerator = N_pdf * self.phi
        denominator = np.sum(numerator, axis=1)
        self.w = numerator / denominator.reshape(-1, 1)

    def M_step(self, dataset):
        w_sum = np.sum(self.w, axis=0)
        self.phi = w_sum / self.N
        self.mu = np.dot(self.w.T, dataset) / w_sum.reshape(-1, 1)
        for j in range(self.k):
            diff = dataset - self.mu[j]
            self.sigma[j] = np.matmul(
                self.w[:, j] * diff.T, diff) / np.sum(self.w[:, j])

    def log_likelihood(self, dataset):
        N_pdf = np.zeros(shape=(self.N, self.k))
        for j in range(self.k):
            N_distr = multivariate_normal(
                mean=self.mu[j], cov=self.sigma[j], allow_singular=True)
            N_pdf[:, j] = N_distr.pdf(dataset)
        weighted_pdf = N_pdf * self.phi
        sum_weighted_pdf = np.sum(weighted_pdf, axis=1)
        log_likelihood = np.sum(np.log(sum_weighted_pdf))
        return log_likelihood

    def predict(self, dataset):
        N_pdf = np.zeros(shape=(self.N, self.k))
        for j in range(self.k):
            N_distr = multivariate_normal(
                mean=self.mu[j], cov=self.sigma[j], allow_singular=True)
            N_pdf[:, j] = N_distr.pdf(dataset)
        numerator = N_pdf * self.phi
        denominator = np.sum(numerator, axis=1)
        weight = numerator / denominator.reshape(-1, 1)
        prediction = np.argmax(weight, axis=1)
        return prediction

    def run_EM(self, dataset, visualize=False, dataset_2d=None, x=None, y=None, pos=None, cmap_list=None):
        self.model_initialize(dataset=dataset)
        log_likelihood_list = []
        if visualize:
            plt.ion()
        print(f'running EM for max {self.max_iter} iterations...')
        for i in tqdm(range(self.max_iter)):
            self.E_step(dataset=dataset)
            self.M_step(dataset=dataset)

            if visualize:
                plt.clf()
                plt.title(f'Iter: {i}')
                plt.scatter(dataset_2d[:, 0], dataset_2d[:, 1], s=0.5)
                # finding mu and sigma for each cluster
                prediction = self.predict(dataset=dataset)

                for j in range(self.k):
                    points_idx = np.where(prediction == j)[0]
                    # print(points_idx)
                    if len(points_idx) == 0:
                        mu = [0, 0]
                        sigma = np.eye(2)
                    elif len(points_idx) == 1:
                        mu = np.mean(dataset_2d[points_idx], axis=0)
                        sigma = np.eye(2)
                    else:
                        points = dataset_2d[points_idx]
                        mu = np.mean(points, axis=0)
                        sigma = np.cov(points.T)
                    N_distr = multivariate_normal(
                        mean=mu, cov=sigma, allow_singular=True)
                    z = N_distr.pdf(pos)
                    # plt.scatter(mu[j, 0], mu[j, 1], s=5, c='r')
                    plt.contour(x, y, z, cmap=cmap_list[j])
                plt.pause(0.01)

            log_likelihood = self.log_likelihood(dataset=dataset)
            log_likelihood_list.append(log_likelihood)
            if i > 0 and np.abs(log_likelihood - log_likelihood_list[i-1]) < self.eps:
                print(f'leaving EM at iteration {i}...')
                break
        plt.ioff()
        plt.show()

    # def AIC(self, log_likelihood):
    #     # AIC = -2 * log_likelihood + 2*k
    #     AIC = -2 * log_likelihood + 2 * self.k
    #     return AIC

    def find_k_star(self, dataset, k_init, k_final):
        log_likelihood_list = np.zeros(shape=(k_final-k_init))
        # aic_list = np.zeros(shape=(k_final-k_init))
        for k in range(k_init, k_final):
            self.k = k
            print(f'for k={k}')
            self.run_EM(dataset=dataset)
            log_likelihood = self.log_likelihood(dataset=dataset)
            log_likelihood_list[k-k_init] = log_likelihood
            # aic = self.AIC(log_likelihood=log_likelihood)
            # aic_list[k-k_init] = aic

        k_star = np.argmax(log_likelihood_list) + k_init
        # plt.plot(range(k_init, k_final), aic_list)
        # plt.show()
        # best_aic = np.argmin(aic_list)
        # k_star = best_aic + k_init
        return self.mu, self.sigma, self.phi, k_star, log_likelihood_list

    def visualize(self, dataset, dataset_2d):
        # interactive visualize for every iteration
        x, y = np.mgrid[dataset_2d[:, 0].min():dataset_2d[:, 0].max(
        ):0.05, dataset_2d[:, 1].min():dataset_2d[:, 1].max():0.05]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        cmap_list_original = ['viridis', 'plasma',
                              'inferno', 'magma', 'cividis']
        # take k from cmap_list_original
        cmap_list = [cmap_list_original[i]
                     for i in np.random.randint(0, len(cmap_list_original), self.k)]

        self.run_EM(dataset=dataset, visualize=True, dataset_2d=dataset_2d, x=x,
                    y=y, pos=pos, cmap_list=cmap_list)


if __name__ == '__main__':
    dataset = load_dataset(sys.argv[1])
    gmm = GMM_EM(k=k_init, max_iter=n_max_iter)
    mu, sigma, phi, k_star, log_likelihood_list = gmm.find_k_star(
        dataset=dataset, k_init=k_init, k_final=k_final+1)

    # plot log likelihood against k
    plt.plot(range(k_init, k_final+1), log_likelihood_list)
    plt.show()

    # manual k*
    k_star = int(input('Enter k*: '))

    # visualize
    if dataset.shape[1] <= 2:
        gmm = GMM_EM(k=k_star, max_iter=n_max_iter)
        gmm.visualize(dataset=dataset, dataset_2d=dataset)
    elif dataset.shape[1] > 2:
        # pca
        pca = PCA(2)
        pca.fit(dataset)
        dataset_2d = pca.transform(dataset)
        gmm = GMM_EM(k=k_star, max_iter=n_max_iter)
        gmm.visualize(dataset=dataset, dataset_2d=dataset_2d)

    print('done')
