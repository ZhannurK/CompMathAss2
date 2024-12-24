import numpy as np

def cramer(A, B):
    n = len(A)
    det_A = np.linalg.det(A)
    if det_A == 0:
        print("The system has no unique solution.")
        return None

    solutions = []
    for i in range(n):
        A_i = np.copy(A)
        A_i[:, i] = B
        solutions.append(np.linalg.det(A_i) / det_A)

    return solutions

def gauss_elimination(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    n = len(B)

    for i in range(n):
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        B[[i, max_row]] = B[[max_row, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            B[j] -= factor * B[i]

    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - np.dot(A[i, i + 1:], X[i + 1:])) / A[i, i]

    return X

def jacobi(A, B, tolerance=1e-10, max_iterations=100):
    n = len(B)
    X = np.zeros(n)
    X_new = np.zeros(n)

    for _ in range(max_iterations):
        for i in range(n):
            s = sum(A[i, j] * X[j] for j in range(n) if j != i)
            X_new[i] = (B[i] - s) / A[i, i]

        if np.linalg.norm(X_new - X, ord=np.inf) < tolerance:
            return X_new

        X[:] = X_new

    print("Jacobi method did not converge.")
    return X

def gauss_seidel(A, B, tolerance=1e-10, max_iterations=100):
    n = len(B)
    X = np.zeros(n)

    for _ in range(max_iterations):
        X_old = np.copy(X)

        for i in range(n):
            s1 = sum(A[i, j] * X[j] for j in range(i))
            s2 = sum(A[i, j] * X_old[j] for j in range(i + 1, n))
            X[i] = (B[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(X - X_old, ord=np.inf) < tolerance:
            return X

    print("Gauss-Seidel method did not converge.")
    return X

if __name__ == "__main__":
    A = np.array([
        [3, -5, 47, 20],
        [11, 16, 17, 10],
        [56, 22, 11, -18],
        [17, 66, -12, 7]
    ])
    B = np.array([18, 26, 34, 82])

    print("Solution using Cramer's Rule:")
    print(cramer(A, B))

    print("\nSolution using Gauss Elimination:")
    print(gauss_elimination(A, B))

    print("\nSolution using Jacobi Method:")
    print(jacobi(A, B))

    print("\nSolution using Gauss-Seidel Method:")
    print(gauss_seidel(A, B))
