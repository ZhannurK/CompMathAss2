import numpy as np

def cramer_method(coeff_matrix, const_terms):
    det = np.linalg.det(coeff_matrix)
    if abs(det) < 1e-10:
        raise ValueError("Determinant is zero, system has no unique solution.")

    solutions = []
    for i in range(coeff_matrix.shape[1]):
        modified_matrix = coeff_matrix.copy()
        modified_matrix[:, i] = const_terms
        solutions.append(np.linalg.det(modified_matrix) / det)

    return solutions


def gauss_elimination(coeff_matrix, const_terms):
    augmented_matrix = np.hstack((coeff_matrix, const_terms.reshape(-1, 1)))
    n = len(const_terms)

    for i in range(n):
        max_row = np.argmax(abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])) / augmented_matrix[i, i]

    return x


def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i, j]) for j in range(n) if j != i)
        if abs(matrix[i, i]) <= row_sum:
            return False
    return True


def make_diagonally_dominant(coeff_matrix, const_terms):
    n = coeff_matrix.shape[0]
    for i in range(n):
        for j in range(i, n):
            if abs(coeff_matrix[j, i]) > sum(abs(coeff_matrix[j, k]) for k in range(n) if k != i):
                coeff_matrix[[i, j]] = coeff_matrix[[j, i]]
                const_terms[[i, j]] = const_terms[[j, i]]
    return coeff_matrix, const_terms


def jacobi_method(coeff_matrix, const_terms, tol=1e-7, max_iter=100, relaxation=1.0):
    coeff_matrix, const_terms = make_diagonally_dominant(coeff_matrix, const_terms)

    n = len(const_terms)
    x = gauss_elimination(coeff_matrix, const_terms)  # Use Gaussian elimination as initial guess
    x_new = np.zeros(n)

    for _ in range(max_iter):
        for i in range(n):
            sum_j = np.dot(coeff_matrix[i, :], x) - coeff_matrix[i, i] * x[i]
            x_new[i] = (1 - relaxation) * x[i] + relaxation * (const_terms[i] - sum_j) / coeff_matrix[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new.copy()

    raise ValueError("Jacobi method did not converge.")


def gauss_seidel_method(coeff_matrix, const_terms, tol=1e-7, max_iter=100):
    n = len(const_terms)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum_j = np.dot(coeff_matrix[i, :], x) - coeff_matrix[i, i] * x[i]
            x[i] = (const_terms[i] - sum_j) / coeff_matrix[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x

    raise ValueError("Gauss-Seidel method did not converge.")


coeff_matrix = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
], dtype=float)

const_terms = np.array([18, 26, 34, 82], dtype=float)

cramer_solution = cramer_method(coeff_matrix, const_terms)
print("Cramer's Rule Solution:", cramer_solution)

gauss_solution = gauss_elimination(coeff_matrix, const_terms)
print("Gaussian Elimination Solution:", gauss_solution)

jacobi_solution = jacobi_method(coeff_matrix, const_terms)
print("Jacobi Method Solution:", jacobi_solution)

gauss_seidel_solution = gauss_seidel_method(coeff_matrix, const_terms)
print("Gauss-Seidel Method Solution:", gauss_seidel_solution)
