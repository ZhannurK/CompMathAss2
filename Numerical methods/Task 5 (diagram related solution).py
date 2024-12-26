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
    if det_A == 0:
        raise ValueError("The determinant of A is zero. Cramer's method cannot be applied.")
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
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]
    return x

def jacobi_method(A, b, tol=1e-5, max_iterations=100):
    n = len(b)
    x = [0] * n
    for _ in range(max_iterations):
        x_new = [0] * n
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new
        x = x_new
    raise ValueError("Jacobi method did not converge within the maximum number of iterations.")

def gauss_seidel_method(A, b, tol=1e-5, max_iterations=100):
    n = len(b)
    x = [0] * n
    for _ in range(max_iterations):
        x_new = x[:]
        for i in range(n):
            sum_ax = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new
        x = x_new
    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

A = [
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
]

b = [18, 26, 34, 82]

try:
    x_cramer = cramer_method(A, b)
    print("Solution using Cramer's method:", x_cramer)
except ValueError as e:
    print(e)

x_gauss = gauss_elimination(matrix_copy(A), b[:])
print("Solution using Gaussian elimination:", x_gauss)

try:
    x_jacobi = jacobi_method(A, b)
    print("Solution using Jacobi method:", x_jacobi)
except ValueError as e:
    print(e)

try:
    x_gauss_seidel = gauss_seidel_method(A, b)
    print("Solution using Gauss-Seidel method:", x_gauss_seidel)
except ValueError as e:
    print(e)
