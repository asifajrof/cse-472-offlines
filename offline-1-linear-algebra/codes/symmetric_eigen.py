import numpy as np

# define global variables

MAX_INT = 20

# define functions


def rand_inv_symm_matrix(n):
    mat = None
    det = 0
    while True:
        mat = np.random.randint(-MAX_INT, MAX_INT, size=(n, n))
        mat = mat + mat.T
        if np.allclose(mat, mat.T) and np.linalg.det(mat) != 0:
            break
    return mat


# main
if __name__ == '__main__':
    n = int(input('Enter the dimension of the matrix: '))
    mat_A = rand_inv_symm_matrix(n)
    print(f'Matrix A: {mat_A}')

    # eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eig(mat_A)
    print(f'Eigen values: {eigen_values}')
    # eigen vectors are columns of the matrix
    print(f'Eigen vectors: {eigen_vectors}')

    # reconstruct the matrix
    # A = V * diag(lambda) * V^T
    mat_B = np.matmul(eigen_vectors, np.matmul(
        np.diag(eigen_values), eigen_vectors.T))
    print(f'Matrix B: {mat_B}')

    # check if A == B
    print(f'Are A and B equal? {np.allclose(mat_A, mat_B)}')
