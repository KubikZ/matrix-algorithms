import numpy as np
from random import uniform

counter_add = 0
counter_mul = 0
counter_sub = 0


def reset_counters():
    global counter_add, counter_mul, counter_sub
    counter_add = 0
    counter_mul = 0
    counter_sub = 0


def mat_add(A, B):
    global counter_add, counter_mul
    n = A.shape[0]
    m = A.shape[1]
    C = np.zeros((n, m))
    for row in range(n):
        for col in range(m):
            counter_add += 1
            counter_mul += 2
            C[row][col] = A[row][col] + B[row][col]
    return C


def mat_sub(A, B):
    global counter_add, counter_mul, counter_sub
    n = A.shape[0]
    m = A.shape[1]
    C = np.zeros((n, m))
    for row in range(n):
        for col in range(m):
            counter_sub += 1
            counter_mul += 2
            C[row][col] = A[row][col] - B[row][col]
    return C


def randomize_matrix(n):
    return [[uniform(1e-8, 1) for _ in range(n)] for _ in range(n)]


def randomize_vec(n):
    return [uniform(1e-8, 1) for _ in range(n)]


def check_mat(C, C_test, tol=1e-9):
    return np.allclose(C, C_test, atol=tol)

