"""
Recursive Gaussian elimination and determinant calculation.
"""

import numpy as np
import common
from multiplication import strassen
from lu_factorization import forward_substitution, backward_substitution


def recursive_gaussian_elimination(A, b=None):
    """
    Recursive Gaussian elimination (simplified version without pivoting for now).
    Uses LU factorization approach: A = L * U, then solve Ly = b, then Ux = y.
    
    Args:
        A: Matrix to eliminate (n x n)
        b: Optional right-hand side vector (n x m) for solving Ax = b
    
    Returns:
        U: Upper triangular matrix (row echelon form)
        P: Identity matrix (no pivoting in this version)
        x: Solution vector if b is provided, None otherwise
    """
    from lu_factorization import recursive_lu
    
    n = A.shape[0]
    
    # Use LU factorization
    L, U = recursive_lu(A)
    
    P = np.eye(n)  # No pivoting in this version
    
    if b is not None:
        # Solve Ly = b using forward substitution
        y = forward_substitution(L, b)
        # Solve Ux = y using backward substitution
        x = backward_substitution(U, y)
        return U, P, x
    
    return U, P


def recursive_determinant(A):
    """
    Recursive determinant calculation using block matrix determinant formula.
    
    For a block matrix:
    A = [A11  A12]
        [A21  A22]
    
    det(A) = det(A11) * det(A22 - A21 * A11^(-1) * A12)
           = det(A11) * det(Schur complement)
    
    Or using LU factorization: det(A) = det(L) * det(U) = det(U)
    """
    n = A.shape[0]
    
    # Base case: 1x1 matrix
    if n == 1:
        common.counter_mul += 0  # Just return the value
        return A[0, 0]
    
    # Base case: 2x2 matrix
    if n == 2:
        common.counter_mul += 2
        common.counter_sub += 1
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        return det
    
    # Use LU factorization for efficiency
    # det(A) = det(L) * det(U) = 1 * det(U) = product of diagonal of U
    from lu_factorization import recursive_lu
    
    L, U = recursive_lu(A)
    
    # Compute determinant as product of diagonal of U
    det = 1.0
    for i in range(n):
        common.counter_mul += 1
        det *= U[i, i]
    
    return det


def recursive_determinant_block(A):
    """
    Alternative recursive determinant using block matrix formula directly.
    This is less efficient but demonstrates the block approach.
    """
    n = A.shape[0]
    
    # Base case: 1x1 matrix
    if n == 1:
        return A[0, 0]
    
    # Base case: 2x2 matrix
    if n == 2:
        common.counter_mul += 2
        common.counter_sub += 1
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    # Recursive case: use block determinant formula
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    # If A11 is invertible: det(A) = det(A11) * det(A22 - A21 * A11^(-1) * A12)
    from inversion import recursive_inverse
    
    try:
        A11_inv = recursive_inverse(A11)
        A21_A11_inv = strassen(A21, A11_inv)
        A21_A11_inv_A12 = strassen(A21_A11_inv, A12)
        Schur = common.mat_sub(A22, A21_A11_inv_A12)
        
        det_A11 = recursive_determinant_block(A11)
        det_Schur = recursive_determinant_block(Schur)
        
        common.counter_mul += 1
        return det_A11 * det_Schur
    except:
        # If A11 is singular, we need to use a different approach
        # For simplicity, fall back to LU-based method
        return recursive_determinant(A)

