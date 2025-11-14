import numpy as np
import common


def binet(A, B):
    n = A.shape[0]
    m = A.shape[1]
    p = B.shape[1]

    if n == 1:
        C = np.zeros((n, p))
        for col in range(p):
            for i in range(m):
                common.counter_add += 1
                common.counter_mul += 1
                C[0][col] += A[0][i] * B[i][col]
        return C
    if p == 1:
        C = np.zeros((n, p))
        for row in range(n):
            for i in range(m):
                common.counter_add += 1
                common.counter_mul += 1
                C[row][0] += A[row][i] * B[i][0]
        return C
    
    mid_n = n // 2
    mid_p = p // 2
    mid_m = m // 2
    A11 = A[:mid_n, :mid_m]
    A12 = A[:mid_n, mid_m:]
    A21 = A[mid_n:, :mid_m]
    A22 = A[mid_n:, mid_m:]
    
    B11 = B[:mid_m, :mid_p]
    B12 = B[:mid_m, mid_p:]
    B21 = B[mid_m:, :mid_p]
    B22 = B[mid_m:, mid_p:]
    
    C11 = common.mat_add(binet(A11, B11), binet(A12, B21))
    C12 = common.mat_add(binet(A11, B12), binet(A12, B22))
    C21 = common.mat_add(binet(A21, B11), binet(A22, B21))
    C22 = common.mat_add(binet(A21, B12), binet(A22, B22))

    C = np.zeros((n, p))
    C[:mid_n, :mid_p] = C11
    C[:mid_n, mid_p:] = C12
    C[mid_n:, :mid_p] = C21
    C[mid_n:, mid_p:] = C22
    
    return C


def strassen(A, B, cutoff=32):
    n = A.shape[0]
    m = A.shape[1]
    p = B.shape[1]
    
    if n != m or m != p or n != p:
        return binet(A, B)
    
    if n <= cutoff:
        return binet(A, B)
    
    if n == 1:
        common.counter_mul += 1
        return np.array([[A[0][0] * B[0][0]]])
    
    if n % 2 == 1:
        m_pad = n + 1
        A_padded = np.zeros((m_pad, m_pad))
        B_padded = np.zeros((m_pad, m_pad))
        A_padded[:n, :n] = A
        B_padded[:n, :n] = B
        
        C_padded = strassen(A_padded, B_padded, cutoff)
        return C_padded[:n, :n]
    
    k = n // 2
    
    A11 = A[:k, :k]
    A12 = A[:k, k:]
    A21 = A[k:, :k]
    A22 = A[k:, k:]
    
    B11 = B[:k, :k]
    B12 = B[:k, k:]
    B21 = B[k:, :k]
    B22 = B[k:, k:]
    
    # Compute the 7 products using Strassen's method
    # M1 = (A11 + A22) * (B11 + B22)
    M1 = strassen(common.mat_add(A11, A22), common.mat_add(B11, B22), cutoff)
    
    # M2 = (A21 + A22) * B11
    M2 = strassen(common.mat_add(A21, A22), B11, cutoff)
    
    # M3 = A11 * (B12 - B22)
    M3 = strassen(A11, common.mat_sub(B12, B22), cutoff)
    
    # M4 = A22 * (B21 - B11)
    M4 = strassen(A22, common.mat_sub(B21, B11), cutoff)
    
    # M5 = (A11 + A12) * B22
    M5 = strassen(common.mat_add(A11, A12), B22, cutoff)
    
    # M6 = (A21 - A11) * (B11 + B12)
    M6 = strassen(common.mat_sub(A21, A11), common.mat_add(B11, B12), cutoff)
    
    # M7 = (A12 - A22) * (B21 + B22)
    M7 = strassen(common.mat_sub(A12, A22), common.mat_add(B21, B22), cutoff)
    
    # Compute result blocks
    # C11 = M1 + M4 - M5 + M7
    C11 = common.mat_add(common.mat_sub(common.mat_add(M1, M4), M5), M7)
    
    # C12 = M3 + M5
    C12 = common.mat_add(M3, M5)
    
    # C21 = M2 + M4
    C21 = common.mat_add(M2, M4)
    
    # C22 = M1 - M2 + M3 + M6
    C22 = common.mat_add(common.mat_sub(common.mat_add(M1, M3), M2), M6)
    
    # Combine blocks
    C = np.zeros((n, n))
    C[:k, :k] = C11
    C[:k, k:] = C12
    C[k:, :k] = C21
    C[k:, k:] = C22
    
    return C