import numpy as np
import common

def recursive_inverse(A, mat_mul):    
    n = A.shape[0]
    
    if n == 1:
        common.counter_mul += 1
        inv = np.zeros((1, 1))
        inv[0][0] = 1.0 / A[0][0]
        return inv
    
    k = n // 2
    
    A11 = A[:k, :k]
    A12 = A[:k, k:]
    A21 = A[k:, :k]
    A22 = A[k:, k:]

    A11_inv = recursive_inverse(A11, mat_mul)
    S22 = A22 - mat_mul(mat_mul(A21, A11_inv), A12)
    S22_inv = recursive_inverse(S22, mat_mul)

    B11 = A11_inv + mat_mul(mat_mul(mat_mul(mat_mul(A11_inv, A12), S22_inv), A21), A11_inv)
    B12 = -mat_mul(mat_mul(A11_inv, A12), S22_inv)
    B21 = -mat_mul(mat_mul(S22_inv, A21), A11_inv)
    B22 = S22_inv
    
    result = np.block([
        [B11, B12],
        [B21, B22]
    ])

    return result

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
    U11_inv = recursive_inverse(U11, mat_mul)
    L21 = mat_mul(A21, U11_inv)
    L11_inv = recursive_inverse(L11, mat_mul)
    U12 = mat_mul(L11_inv, A12)
    L22 = S = A22 - mat_mul(mat_mul(mat_mul(A21, U11_inv), L11_inv), A12)
    common.counter_sub += 1
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
    L11_inv = recursive_inverse(L11, mat_mul)
    U11_inv = recursive_inverse(U11, mat_mul)
    S = A22 - mat_mul(mat_mul(mat_mul(A21, U11_inv), L11_inv), A12)
    Ls, Us = LU_factorization(S, mat_mul)
    Ls_inv = recursive_inverse(Ls, mat_mul)

    RHS1 = mat_mul(L11_inv, b1)
    RHS2 = mat_mul(Ls_inv, b2) - mat_mul(mat_mul(mat_mul(mat_mul(Ls_inv, A21), U11_inv), L11_inv), b1)
    common.counter_sub += 1
    RHS = np.vstack([RHS1, RHS2])

    U12 = mat_mul(L11_inv, A12)
    U22 = Us

    bottom_left_zero = np.zeros((n - k, k), dtype=A.dtype)

    U = np.block([
        [U11, U12],
        [bottom_left_zero, U22]
    ])

    U_inv = recursive_inverse(U, mat_mul)

    x = mat_mul(U_inv, RHS)
    
    return x

def determinant(A, mat_mul):
    L, U = LU_factorization(A, mat_mul)
    det_U = 1
    for i in range(len(U)):
        det_U *= U[i][i]  * L[i][i]

    return det_U