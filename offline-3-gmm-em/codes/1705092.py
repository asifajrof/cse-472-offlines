# gmm with em algorithm using numpy

import sys
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

logging = False
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
        # # dataset's sigma as all cluster's sigma
        # self.sigma = np.full(shape=(self.k, self.D, self.D), fill_value=np.cov(dataset.T))
        # k identity matrix of shape DxD
        self.sigma = np.full(shape=(self.k, self.D, self.D),
                             fill_value=np.eye(self.D))
        self.phi = np.full(shape=self.k, fill_value=1 / self.k)
        self.w = np.full(shape=(self.N, self.k), fill_value=1 / self.k)

        if logging:
            print('model_initialize:')
            print(f'N: {self.N}; D: {self.D}')
            print(f'mu: {self.mu}')
            print(f'mu shape: {self.mu.shape}')
            print(f'sigma: {self.sigma}')
            print(f'sigma shape: {self.sigma.shape}')
            print(f'phi: {self.phi}')
            print(f'phi shape: {self.phi.shape}')
            print(f'w: {self.w}')
            print(f'w shape: {self.w.shape}')

    def E_step(self, dataset):
        N_pdf = np.zeros(shape=(self.N, self.k))
        for j in range(self.k):
            N_distr = multivariate_normal(
                mean=self.mu[j], cov=self.sigma[j], allow_singular=True)
            N_pdf[:, j] = N_distr.pdf(dataset)
        numerator = N_pdf * self.phi
        denominator = np.sum(numerator, axis=1)
        self.w = numerator / denominator.reshape(-1, 1)

        if logging:
            print('E_step:')
            print(f'N_pdf: {N_pdf}')
            print(f'N_pdf shape: {N_pdf.shape}')
            print(f'numerator: {numerator}')
            print(f'numerator shape: {numerator.shape}')
            print(f'denominator: {denominator}')
            print(f'denominator shape: {denominator.shape}')
            print(f'w: {self.w}')
            print(f'w shape: {self.w.shape}')

    def M_step(self, dataset):
        w_sum = np.sum(self.w, axis=0)
        self.phi = w_sum / self.N
        self.mu = np.dot(self.w.T, dataset) / w_sum.reshape(-1, 1)
        for j in range(self.k):
            diff = dataset - self.mu[j]
            self.sigma[j] = np.matmul(
                self.w[:, j] * diff.T, diff) / np.sum(self.w[:, j])

        if logging:
            print('M_step:')
            print(f'phi: {self.phi}')
            print(f'phi shape: {self.phi.shape}')
            print(f'mu: {self.mu}')
            print(f'mu shape: {self.mu.shape}')
            print(f'sigma: {self.sigma}')
            print(f'sigma shape: {self.sigma.shape}')

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

    def run_EM(self, dataset, visualize=False, x=None, y=None, pos=None, cmap_list=None):
        self.model_initialize(dataset=dataset)
        log_likelihood_list = []
        if visualize:
            print(cmap_list)
            plt.ion()
        print(f'running EM for max {self.max_iter} iterations...')
        for i in tqdm(range(self.max_iter)):
            self.E_step(dataset=dataset)
            self.M_step(dataset=dataset)

            if visualize:
                plt.clf()
                plt.title(f'Iter: {i}')
                plt.scatter(dataset[:, 0], dataset[:, 1], s=0.5)
                for j in range(self.k):
                    N_distr = multivariate_normal(
                        mean=self.mu[j], cov=self.sigma[j], allow_singular=True)
                    z = N_distr.pdf(pos)
                    plt.scatter(mu[j, 0], mu[j, 1], s=5, c='r')
                    plt.contour(x, y, z, cmap=cmap_list[j])
                plt.pause(0.01)

            log_likelihood = self.log_likelihood(dataset=dataset)
            # print(f'iter {i}: log_likelihood: {log_likelihood}')
            log_likelihood_list.append(log_likelihood)
            if i > 0 and np.abs(log_likelihood - log_likelihood_list[i-1]) < self.eps:
                # print(f'iter {i}: log_likelihood: {log_likelihood}')
                print(f'leaving')
                break
        plt.ioff()
        plt.show()

        # return self.mu, self.sigma, self.phi, log_likelihood_list

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

    def visualize(self, dataset):
        # interactive visualize for every iteration
        self.model_initialize(dataset=dataset)
        log_likelihood_list = []
        print(f'running EM for max {self.max_iter} iterations...')
        x, y = np.mgrid[dataset[:, 0].min():dataset[:, 0].max():complex(
            0, dataset.shape[0]), dataset[:, 1].min():dataset[:, 1].max():complex(0, dataset.shape[0])]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        cmap_list_original = ['viridis', 'plasma',
                              'inferno', 'magma', 'cividis']
        # take k from cmap_list_original
        cmap_list = [cmap_list_original[i]
                     for i in np.random.randint(0, self.k, self.k)]

        self.run_EM(dataset=dataset, visualize=True, x=x,
                    y=y, pos=pos, cmap_list=cmap_list)


if __name__ == '__main__':
    dataset = load_dataset(sys.argv[1])
    if logging:
        print(dataset)
    gmm = GMM_EM(k=k_init, max_iter=n_max_iter)
    mu, sigma, phi, k_star, log_likelihood_list = gmm.find_k_star(
        dataset=dataset, k_init=k_init, k_final=k_final+1)
    # print(f'k*: {k_star}')
    # print(f'mu: {mu}')
    # print(f'sigma: {sigma}')
    # print(f'phi: {phi}')
    # plot likelihoods
    # from sklearn.mixture import GaussianMixture
    # gmm_skl = GaussianMixture(n_components=k_final,
    #                           max_iter=n_max_iter).fit(dataset)
    # gmm_scores = gmm_skl.score_samples(dataset)

    # print('Means by sklearn:\n', gmm_skl.means_)
    # print('Means by our implementation:\n', mu)
    # print('Scores by sklearn:\n', gmm_scores[0:20])
    # print('Scores by our implementation:\n', sample_likelihoods.reshape(-1)[0:20])

    plt.plot(range(k_init, k_final+1), log_likelihood_list)
    plt.show()

    # manual k*
    k_star = int(input('Enter k*: '))
    print(dataset.shape[1])

    if dataset.shape[1] <= 2:
        # visualize
        gmm = GMM_EM(k=k_star, max_iter=n_max_iter)
        gmm.visualize(dataset)
        print(f'done')
