"""
Recursive matrix inversion using block matrix inversion.
"""

import numpy as np
import common

def recursive_inverse(A, mat_mul_function):
    """
    Recursive matrix inversion using block matrix inversion.
    Divides the matrix into 2x2 blocks and recursively computes the inverse.
    """
    
    n = A.shape[0]
    
    # Base case: 1x1 matrix
    if n == 1:
        common.counter_mul += 1
        inv = np.zeros((1, 1))
        inv[0][0] = 1.0 / A[0][0]
        return inv
    
    # Base case: 2x2 matrix (can use direct formula for efficiency)
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
    
    # Recursive case: divide into blocks
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    # Recursively compute A22^(-1)
    A22_inv = recursive_inverse(A22, mat_mul_function)
    
    # Compute Schur complement: S = A11 - A12 * A22^(-1) * A21
    A22_inv_A21 = mat_mul_function(A22_inv, A21)
    A12_A22_inv_A21 = mat_mul_function(A12, A22_inv_A21)
    S = common.mat_sub(A11, A12_A22_inv_A21)
    
    # Recursively compute S^(-1)
    S_inv = recursive_inverse(S, mat_mul_function)
    
    # Compute block components
    # B11 = S^(-1)
    B11 = S_inv
    
    # B12 = -S^(-1) * A12 * A22^(-1)
    A12_A22_inv = mat_mul_function(A12, A22_inv)
    B12 = mat_mul_function(S_inv, A12_A22_inv)
    # Negate B12 (B12 is already a new matrix from mat_mul_function, so no copy needed)
    for row in range(B12.shape[0]):
        for col in range(B12.shape[1]):
            B12[row][col] = -B12[row][col]
            common.counter_sub += 1
    
    # B21 = -A22^(-1) * A21 * S^(-1)
    A22_inv_A21_S_inv = mat_mul_function(A22_inv_A21, S_inv)
    B21 = A22_inv_A21_S_inv.copy()  # Make a copy before negating
    # Negate B21
    for row in range(B21.shape[0]):
        for col in range(B21.shape[1]):
            B21[row][col] = -B21[row][col]
            common.counter_sub += 1
    
    # B22 = A22^(-1) + A22^(-1) * A21 * S^(-1) * A12 * A22^(-1)
    #     = A22^(-1) + A22_inv_A21_S_inv * A12 * A22^(-1)
    A22_inv_A21_S_inv_A12 = mat_mul_function(A22_inv_A21_S_inv, A12)
    A22_inv_A21_S_inv_A12_A22_inv = mat_mul_function(A22_inv_A21_S_inv_A12, A22_inv)
    B22 = common.mat_add(A22_inv, A22_inv_A21_S_inv_A12_A22_inv)
    
    # Combine blocks into result
    result = np.zeros((n, n))
    result[:mid, :mid] = B11
    result[:mid, mid:] = B12
    result[mid:, :mid] = B21
    result[mid:, mid:] = B22
    
    return result

