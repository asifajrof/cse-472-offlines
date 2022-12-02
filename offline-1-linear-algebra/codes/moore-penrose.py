import numpy as np

# define global variables

MAX_INT = 20

# define functions


def rand_matrix(n, m):
    mat = np.random.randint(-MAX_INT, MAX_INT, size=(n, m))
    return mat


# main
if __name__ == '__main__':
    n = int(input('Enter the dimension n of the matrix: '))
    m = int(input('Enter the dimension m of the matrix: '))

    mat_A = rand_matrix(n, m)
    print(f'Matrix A: {mat_A}')

    # singular value decomposition
    left_singular_vectors, singular_values, right_singular_vectors_T = np.linalg.svd(
        mat_A)
    print(f'Singular values: {singular_values}')

    print(f'Left-singular vectors: {left_singular_vectors}')
    print(f'right-singular vectors : {right_singular_vectors_T.T}')

    # # reconstruct the matrix
    # singular_values_matrix = np.zeros((n, m))
    # np.fill_diagonal(singular_values_matrix, singular_values)
    # mat_B = np.matmul(left_singular_vectors, np.matmul(
    #     singular_values_matrix, right_singular_vectors_T))
    # print(f'Matrix B: {mat_B}')

    # # check if A == B
    # print(f'Are A and B equal? {np.allclose(mat_A, mat_B)}')

    # moore-penrose pseudo inverse using np.linalg.pinv
    mat_A_pinv = np.linalg.pinv(mat_A)
    print(f'Matrix A pseudo inverse (using numpy): {mat_A_pinv}')

    # moore-penrose pseudo inverse using svd
    singular_values_matrix_plus = np.zeros((m, n))
    # m x n cause pseudo-inverse
    np.fill_diagonal(singular_values_matrix_plus, [
                     1/i if not np.allclose(i, 0) else 0 for i in singular_values])
    mat_A_pinv_svd = np.matmul(right_singular_vectors_T.T, np.matmul(
        singular_values_matrix_plus, left_singular_vectors.T))
    print(f'Matrix A pseudo inverse (using svd): {mat_A_pinv_svd}')

    # check if mat_A_pinv == mat_A_pinv_svd
    print(
        f'Are these two pseudo inverse equal? {np.allclose(mat_A_pinv, mat_A_pinv_svd)}')
