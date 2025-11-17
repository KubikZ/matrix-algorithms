import numpy as np
from inversion import recursive_inverse

def LU_factorization(A, mat_mul):

    n = A.shape[0]

    if n == 1:
        return np.eye(1), A.copy()

    k = n // 2

    A11 = A[:k, :k]
    A12 = A[:k, k:]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    L11, U11 = LU_factorization(A11, mat_mul)
    U11_inv = recursive_inverse(U11)
    L21 = mat_mul(A21, U11_inv)
    L11_inv = recursive_inverse(L11)
    U12 = mat_mul(L11_inv, A12)
    L22 = S = A22 - mat_mul(mat_mul(mat_mul(A21, U11_inv), L11_inv), A12)
    Ls, Us = LU_factorization(S, mat_mul)
    U22 = Us
    L22 = Ls

    top_right_zero = np.zeros((k, n - k), dtype=A.dtype)
    bottom_left_zero = np.zeros((n - k, k), dtype=A.dtype)

    L = np.block([
        [L11, top_right_zero],
        [L21, L22]
    ])

    U = np.block([
        [U11, U12],
        [bottom_left_zero, U22]
    ])

    return L, U

def gaussian_elimination(A, b, mat_mul):
    n = A.shape[0]
    
    if n == 1:
        return b / A[0, 0]

    k = n // 2

    A11 = A[:k, :k]
    A12 = A[:k, k:]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    b1 = b[:k]
    b2 = b[k:]

    L11, U11 = LU_factorization(A11, mat_mul)
    L11_inv = recursive_inverse(L11)
    U11_inv = recursive_inverse(U11)
    S = A22 - mat_mul(mat_mul(mat_mul(A21, U11_inv), L11_inv), A12)
    Ls, Us = LU_factorization(S, mat_mul)
    Ls_inv = recursive_inverse(Ls)

    RHS1 = mat_mul(L11_inv, b1)
    RHS2 = mat_mul(Ls_inv, b2) - mat_mul(mat_mul(mat_mul(mat_mul(Ls_inv, A21), U11_inv), L11_inv), b1)
    RHS = np.vstack(RHS1, RHS2)

    U12 = mat_mul(L11_inv, A12)
    U22 = Us

    bottom_left_zero = np.zeros((n - k, k), dtype=A.dtype)

    U = np.block([
        [U11, U12],
        [bottom_left_zero, U22]
    ])

    U_inv = recursive_inverse(U)

    x = mat_mul(U_inv, RHS)
    
    return x