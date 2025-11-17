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

def plot_all_metrics(step, time_counters, add_counters, sub_counters, mul_counters, mem_counters, mat_mul):
    import matplotlib.pyplot as plt

    if len(time_counters) == 0:
        return

    x_values = list(range(1, len(time_counters) * step + 1, step))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Time plot
    axs[0, 0].plot(x_values, time_counters, label="Czas działania", color='blue')
    axs[0, 0].set_xlabel("Rozmiar macierzy")
    axs[0, 0].set_ylabel("Czas działania [s]")
    axs[0, 0].set_title(f"Czas działania metody rekurencyjnej ({mat_mul})")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Operations plot (add/sub/mul and total)
    total_ops = [a + s + m for a, s, m in zip(add_counters, sub_counters, mul_counters)]
    axs[0, 1].plot(x_values, total_ops, label="Liczba operacji", color='orange')
    axs[0, 1].set_xlabel("Rozmiar macierzy")
    axs[0, 1].set_ylabel("Liczba operacji")
    axs[0, 1].set_title(f"Liczba operacji ({mat_mul})")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Ops per time
    ops_per_time = []
    for a, s, m, t in zip(add_counters, sub_counters, mul_counters, time_counters):
        denom = t if t > 0 else 1e-12
        ops_per_time.append((a + s + m) / denom)
    axs[1, 0].plot(x_values, ops_per_time, label="Operacje / czas", color='brown')
    axs[1, 0].set_xlabel("Rozmiar macierzy")
    axs[1, 0].set_ylabel("Operacje / s")
    axs[1, 0].set_title("Liczba operacji / czas")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Memory plot
    x_mem = list(range(1, len(mem_counters) * step + 1, step))
    axs[1, 1].plot(x_mem, mem_counters, label="Zajęte miejsce", color='orange')
    axs[1, 1].set_xlabel("Rozmiar macierzy")
    axs[1, 1].set_ylabel("Zajęte miejsce [MB]")
    axs[1, 1].set_title(f"Pamięć używana podczas algorytmu ({mat_mul})")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()