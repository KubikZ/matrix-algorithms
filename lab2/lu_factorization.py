"""
Recursive LU factorization using block LU decomposition.
"""

import numpy as np
import common
from inversion import recursive_inverse


def forward_substitution(L, b):
    """
    Solve L * x = b using forward substitution.
    L is a lower triangular matrix.
    """
    n = L.shape[0]
    x = np.zeros((n, b.shape[1]))
    
    for i in range(n):
        for j in range(b.shape[1]):
            sum_val = b[i, j]
            for k in range(i):
                common.counter_add += 1
                common.counter_mul += 1
                sum_val -= L[i, k] * x[k, j]
            if abs(L[i, i]) < 1e-10:
                raise ValueError(f"Singular matrix: L[{i},{i}] = 0")
            common.counter_mul += 1
            x[i, j] = sum_val / L[i, i]
    
    return x


def backward_substitution(U, b):
    """
    Solve U * x = b using backward substitution.
    U is an upper triangular matrix.
    """
    n = U.shape[0]
    x = np.zeros((n, b.shape[1]))
    
    for i in range(n - 1, -1, -1):
        for j in range(b.shape[1]):
            sum_val = b[i, j]
            for k in range(i + 1, n):
                common.counter_add += 1
                common.counter_mul += 1
                sum_val -= U[i, k] * x[k, j]
            if abs(U[i, i]) < 1e-10:
                raise ValueError(f"Singular matrix: U[{i},{i}] = 0")
            common.counter_mul += 1
            x[i, j] = sum_val / U[i, i]
    
    return x


def recursive_lu(A, mat_mul_function):
    """
    Recursive LU factorization using block LU decomposition.
    Returns L (lower triangular) and U (upper triangular) such that A = L * U.
    
    Algorithm:
    For A = [A11  A12]
            [A21  A22]
    
    1. Recursively compute L11, U11 = LU(A11)
    2. Solve L11 * U12 = A12 for U12
    3. Solve L21 * U11 = A21 for L21
    4. Compute Schur complement: S = A22 - L21 * U12
    5. Recursively compute L22, U22 = LU(S)
    
    Result:
    L = [L11   0 ]
        [L21  L22]
    
    U = [U11  U12]
        [ 0   U22]
    """
    n = A.shape[0]
    
    # Base case: 1x1 matrix
    if n == 1:
        if abs(A[0, 0]) < 1e-10:
            raise ValueError("Singular matrix: cannot factorize")
        L = np.array([[1.0]])
        U = np.array([[A[0, 0]]])
        return L, U
    
    # Base case: 2x2 matrix - use direct computation
    if n == 2:
        if abs(A[0, 0]) < 1e-10:
            raise ValueError("Singular matrix: pivot is zero")
        
        L = np.zeros((2, 2))
        U = np.zeros((2, 2))
        
        # L[0,0] = 1, U[0,0] = A[0,0]
        L[0, 0] = 1.0
        U[0, 0] = A[0, 0]
        
        # U[0,1] = A[0,1]
        U[0, 1] = A[0, 1]
        common.counter_mul += 0  # Just assignment
        
        # L[1,0] = A[1,0] / U[0,0]
        common.counter_mul += 1
        L[1, 0] = A[1, 0] / U[0, 0]
        
        # U[1,1] = A[1,1] - L[1,0] * U[0,1]
        common.counter_mul += 1
        common.counter_sub += 1
        U[1, 1] = A[1, 1] - L[1, 0] * U[0, 1]
        
        # L[1,1] = 1
        L[1, 1] = 1.0
        
        return L, U
    
    # Recursive case: divide into blocks
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    # Step 1: Recursively compute L11, U11 = LU(A11)
    L11, U11 = recursive_lu(A11, mat_mul_function)
    
    # Step 2: Solve L11 * U12 = A12 for U12
    # This is equivalent to: U12 = L11^(-1) * A12
    # We use forward substitution: solve L11 * U12 = A12
    U12 = forward_substitution(L11, A12)
    
    # Step 3: Solve L21 * U11 = A21 for L21
    # This is equivalent to: L21 = A21 * U11^(-1)
    # We solve column by column: for each column j of L21, solve U11^T * L21^T[:,j] = A21^T[:,j]
    # Since U11 is upper triangular, U11^T is lower triangular, so we use forward substitution
    U11_T = U11.T
    A21_T = A21.T
    L21_T = forward_substitution(U11_T, A21_T)
    L21 = L21_T.T
    
    # Step 4: Compute Schur complement: S = A22 - L21 * U12
    L21_U12 = mat_mul_function(L21, U12)
    S = common.mat_sub(A22, L21_U12)
    
    # Step 5: Recursively compute L22, U22 = LU(S)
    L22, U22 = recursive_lu(S, mat_mul_function)
    
    # Combine blocks
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    L[:mid, :mid] = L11
    L[mid:, :mid] = L21
    L[mid:, mid:] = L22
    
    U[:mid, :mid] = U11
    U[:mid, mid:] = U12
    U[mid:, mid:] = U22
    
    return L, U

