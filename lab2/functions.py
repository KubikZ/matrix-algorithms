import numpy as np
import common

def recursive_inverse(A, mat_mul):    
    n = A.shape[0]
    
    if n == 1:
        common.counter_mul += 1
        inv = np.zeros((1, 1))
        inv[0][0] = 1.0 / A[0][0]
        return inv
    
    if n == 2:
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        common.counter_mul += 3
        common.counter_sub += 1
        
        inv = np.zeros((2, 2))
        inv[0][0] = A[1][1] / det
        inv[0][1] = -A[0][1] / det
        inv[1][0] = -A[1][0] / det
        inv[1][1] = A[0][0] / det
        common.counter_mul += 4
        common.counter_sub += 2
        return inv
    
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    A22_inv = recursive_inverse(A22, mat_mul)
    
    A22_inv_A21 = mat_mul(A22_inv, A21)
    A12_A22_inv_A21 = mat_mul(A12, A22_inv_A21)
    S = common.mat_sub(A11, A12_A22_inv_A21)
    
    S_inv = recursive_inverse(S, mat_mul)
    
    B11 = S_inv
    
    A12_A22_inv = mat_mul(A12, A22_inv)
    B12 = mat_mul(S_inv, A12_A22_inv)

    for row in range(B12.shape[0]):
        for col in range(B12.shape[1]):
            B12[row][col] = -B12[row][col]
            common.counter_sub += 1

    A22_inv_A21_S_inv = mat_mul(A22_inv_A21, S_inv)
    B21 = A22_inv_A21_S_inv.copy()

    for row in range(B21.shape[0]):
        for col in range(B21.shape[1]):
            B21[row][col] = -B21[row][col]
            common.counter_sub += 1
    
    A22_inv_A21_S_inv_A12 = mat_mul(A22_inv_A21_S_inv, A12)
    A22_inv_A21_S_inv_A12_A22_inv = mat_mul(A22_inv_A21_S_inv_A12, A22_inv)
    B22 = common.mat_add(A22_inv, A22_inv_A21_S_inv_A12_A22_inv)
    
    result = np.zeros((n, n))
    result[:mid, :mid] = B11
    result[:mid, mid:] = B12
    result[mid:, :mid] = B21
    result[mid:, mid:] = B22
    
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