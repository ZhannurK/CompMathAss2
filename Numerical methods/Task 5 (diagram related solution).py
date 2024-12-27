import numpy as np

def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
    return det

def matrix_copy(matrix):
    return [row[:] for row in matrix]

def cramer_method(A, b):
    det_A = determinant(A)
    n = len(b)
    x = [0] * n
    for i in range(n):
        A_i = matrix_copy(A)
        for j in range(n):
            A_i[j][i] = b[j]
        det_A_i = determinant(A_i)
        x[i] = det_A_i / det_A
    return x

def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def jacobi(a, b, n_iter=150, tol=1e-6):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    for m in range(n_iter):
        for i in range(n):
            sigma = sum(a[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - sigma) / a[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Jacobi method converged in {m + 1} iterations.")
            return x_new
        x = x_new.copy()
    raise ValueError("Jacobi method did not converge.")

def gauss_seidel(a, b, n_iterations=150, tolerance=1e-4):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for itr in range(1, n_iterations + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(a[i, j] * x[j] for j in range(n) if i != j)
            x[i] = (b[i] - sigma) / a[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            print(f"Gauss-Seidel method converged in {itr} iterations.")
            return x
    raise ValueError("Gauss-Seidel method did not converge.")


def reorder_rows_for_diagonal_dominance(A, B):
    n = len(A[0])
    for i in range(n):
        # Find the row with the largest absolute diagonal element in column i
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))

        # Swap rows in both A and B
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            B[i], B[max_row] = B[max_row], B[i]
    return A, B

A = [
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
]

b = [18, 26, 34, 82]

x_cramer = cramer_method(A, b)
print("Solution using Cramer's method:", x_cramer)

x_gauss = gauss_elimination(A, b)
print("Solution using Gaussian elimination:", x_gauss)

reorder_rows_for_diagonal_dominance(A, b)

try:
    x_jacobi = jacobi(A, b)
    print(f"Solution using Jacobi method: {x_jacobi}")
except ValueError as e:
    print(f"Jacobi method error: {e}")

try:
    x_gauss_seidel = gauss_seidel(A, b)
    print(f"Solution using Gauss-Seidel method: {x_gauss_seidel}")
except ValueError as e:
    print(f"Gauss-Seidel method error: {e}")
