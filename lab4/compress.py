from scipy.linalg import svd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time

eps = 1e-6

class MatrixLeaf:
    def __init__(self, A, size, U, D, V, r):
        self.size = size
        if np.all(np.abs(A) < eps):
            self.rank = 0
            self.is_zero = True
            return
        self.is_zero = False

        rows, cols = A.shape
        max_possible_rank = min(rows, cols)
        available_singular_values = len(D)
        self.rank = min(r, max_possible_rank, available_singular_values)

        self.singular_vals = D[:self.rank]
        self.U = U[:, :self.rank]
        self.V = V[:self.rank, :]
        

class MatrixNode:
    def __init__(self):
        self.children = []


def CreateTree(A, t_min, t_max, s_min, s_max, r, epsilon):
    A_part = A[t_min:t_max+1, s_min:s_max+1]
    U, D, V = svd(A_part)
    node: MatrixNode
    D_epsilon = D[np.where(D > epsilon)]
    if len(D_epsilon) <= r:
        node = MatrixLeaf(A_part, (t_min, t_max, s_min, s_max), U, D, V, r)
    else:
        node = MatrixNode()
        t_newmax = (t_min + t_max) // 2
        s_newmax = (s_min + s_max) // 2
        node.children.append(CreateTree(A, t_min, t_newmax, s_min, s_newmax, r, epsilon))
        node.children.append(CreateTree(A, t_min, t_newmax, s_newmax + 1, s_max, r, epsilon))
        node.children.append(CreateTree(A, t_newmax + 1, t_max, s_min, s_newmax, r, epsilon))
        node.children.append(CreateTree(A, t_newmax + 1, t_max, s_newmax + 1, s_max, r, epsilon))

    return node

def CompressMatrix(A, r, epsilon):
    return CreateTree(A, 0, A.shape[0] - 1, 0, A.shape[1] - 1, r, epsilon)

def ReconstructMatrix(tree_node, shape):
    result = np.zeros(shape)
    
    def reconstruct_node(node, result):
        if isinstance(node, MatrixLeaf):
            t_min, t_max, s_min, s_max = node.size
            if node.rank > 0 and not node.is_zero:
                leaf_recon = node.U @ np.diag(node.singular_vals) @ node.V
                result[t_min:t_max+1, s_min:s_max+1] = leaf_recon
        else:
            for child in node.children:
                reconstruct_node(child, result)
    
    reconstruct_node(tree_node, result)
    return result

def calculate_compression_stats(tree_root, original_shape, original_dtype=np.float64):
    def count_parameters(node):
        if isinstance(node, MatrixLeaf):
            if node.is_zero or node.rank == 0:
                return 0
            else:
                t_min, t_max, s_min, s_max = node.size
                rows = t_max - t_min + 1
                cols = s_max - s_min + 1
                r = node.rank
                return (rows * r) + r + (r * cols)
        else:
            total = 0
            for child in node.children:
                total += count_parameters(child)
            return total
    
    original_rows, original_cols = original_shape
    original_elements = original_rows * original_cols
    original_bytes = original_elements * np.dtype(original_dtype).itemsize
    
    compressed_elements = count_parameters(tree_root)
    
    compression_ratio = original_elements / compressed_elements if compressed_elements > 0 else float('inf')
    compression_percentage = (1 - (compressed_elements / original_elements)) * 100
    
    compressed_bytes = compressed_elements * np.dtype(np.float64).itemsize
    
    return {
        'original_elements': original_elements,
        'compressed_elements': compressed_elements,
        'compression_ratio': compression_ratio,
        'compression_percentage': compression_percentage,
        'space_saving_percentage': compression_percentage,
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'bytes_saved': original_bytes - compressed_bytes,
        'bits_per_element': (compressed_bytes * 8) / original_elements if original_elements > 0 else 0
    }


def draw_box(draw, x1, y1, x2, y2):
    draw.line((x1, y1, x2, y1), fill=128)
    draw.line((x1, y1, x1, y2), fill=128)
    draw.line((x2, y1, x2, y2), fill=128)
    draw.line((x1, y2, x2, y2), fill=128)

def draw_vis(matrix_root):
    def draw_lines(draw, root):
        if isinstance(root, MatrixLeaf):
            x1, x2, y1, y2 = root.size
            draw_box(draw, y1, x1, y2, x2)
        else:
            for child in root.children:
                draw_lines(draw, child)
    im = Image.new('RGB', (500,500), (255, 255, 255))
    result = ImageDraw.Draw(im)
    draw_lines(result, matrix_root)
    return im


# ============================================================================
# ZADANIE 1: MNOŻENIE SKOMPRESOWANEJ MACIERZY PRZEZ WEKTOR (10 pkt)
# ============================================================================

def matrix_vector_mult(v, X):
    if isinstance(v, MatrixLeaf):
        if v.rank == 0 or v.is_zero:
            t_min, t_max, s_min, s_max = v.size
            return np.zeros(t_max - t_min + 1)
        
        temp = v.V @ X
        temp = temp * v.singular_vals
        return v.U @ temp
    
    rows = len(X)
    mid = rows // 2

    X1 = X[:mid]
    X2 = X[mid:]
    
    Y1_1 = matrix_vector_mult(v.children[0], X1)
    Y1_2 = matrix_vector_mult(v.children[1], X2)
    Y2_1 = matrix_vector_mult(v.children[2], X1)
    Y2_2 = matrix_vector_mult(v.children[3], X2)
    
    Y1 = Y1_1 + Y1_2
    Y2 = Y2_1 + Y2_2
    
    return np.concatenate([Y1, Y2])

def create_zero_leaf(size):
    t_min, t_max, s_min, s_max = size
    rows = t_max - t_min + 1
    cols = s_max - s_min + 1
    A_zero = np.zeros((rows, cols))
    U_zero = np.zeros((rows, 1))
    D_zero = np.zeros(1)
    V_zero = np.zeros((1, cols))
    return MatrixLeaf(A_zero, size, U_zero, D_zero, V_zero, 0)


def matrix_matrix_add(v, w, target_rank):

    if isinstance(v, MatrixLeaf) and isinstance(w, MatrixLeaf):
        
        if v.rank == 0 and w.rank == 0:
            return create_zero_leaf(v.size)
        
        if v.rank == 0:
            return w
        if w.rank == 0:
            return v
        
        if v.rank > 0 and w.rank > 0:
            
            V_v_scaled = np.diag(v.singular_vals) @ v.V
            V_w_scaled = np.diag(w.singular_vals) @ w.V
            
            U_combined = np.hstack([v.U, w.U])
            V_combined = np.vstack([V_v_scaled, V_w_scaled])
            
            A_temp = U_combined @ V_combined
            
            if A_temp.size == 0 or np.all(np.abs(A_temp) < eps):
                return create_zero_leaf(v.size)
            
            U_new, D_new, V_new = svd(A_temp)
            
            k = min(target_rank, len(D_new), len([d for d in D_new if d > eps]))
            
            if k == 0:
                return create_zero_leaf(v.size)
            
            t_min, t_max, s_min, s_max = v.size
            rows = t_max - t_min + 1
            cols = s_max - s_min + 1
            A_result = U_new[:, :k] @ np.diag(D_new[:k]) @ V_new[:k, :]
            
            return MatrixLeaf(A_result, v.size, U_new, D_new, V_new, k)
    
    if isinstance(v, MatrixNode) and isinstance(w, MatrixNode):

        result_node = MatrixNode()
        for i in range(4):
            child_sum = matrix_matrix_add(v.children[i], w.children[i], target_rank)
            result_node.children.append(child_sum)
        return result_node
    
    return v


def matrix_matrix_mult(v, w, target_rank):
    
    if isinstance(v, MatrixLeaf) and isinstance(w, MatrixLeaf):
        
        if v.rank == 0 or w.rank == 0:
            return create_zero_leaf(v.size)

        
        V_v_scaled = np.diag(v.singular_vals) @ v.V
        U_w_scaled = w.U @ np.diag(w.singular_vals)
        middle = V_v_scaled @ U_w_scaled
        
        U_new = v.U @ middle
        V_new = w.V
        
        if U_new.size > 0 and V_new.size > 0:
            A_temp = U_new @ V_new
            
            if np.all(np.abs(A_temp) < eps):
                return create_zero_leaf(v.size)
            
            U_svd, D_svd, V_svd = svd(A_temp)
            
            k = min(target_rank, len(D_svd), len([d for d in D_svd if d > eps]))
            
            if k == 0:
                return create_zero_leaf(v.size)
            
            A_result = U_svd[:, :k] @ np.diag(D_svd[:k]) @ V_svd[:k, :]
            return MatrixLeaf(A_result, v.size, U_svd, D_svd, V_svd, k)
        
        return create_zero_leaf(v.size)
    
    if isinstance(v, MatrixNode) and isinstance(w, MatrixNode):
        
        result_node = MatrixNode()
        
        prod1 = matrix_matrix_mult(v.children[0], w.children[0], target_rank)
        prod2 = matrix_matrix_mult(v.children[1], w.children[2], target_rank)
        c00 = matrix_matrix_add(prod1, prod2, target_rank)
        result_node.children.append(c00)
        
        prod3 = matrix_matrix_mult(v.children[0], w.children[1], target_rank)
        prod4 = matrix_matrix_mult(v.children[1], w.children[3], target_rank)
        c01 = matrix_matrix_add(prod3, prod4, target_rank)
        result_node.children.append(c01)
        
        prod5 = matrix_matrix_mult(v.children[2], w.children[0], target_rank)
        prod6 = matrix_matrix_mult(v.children[3], w.children[2], target_rank)
        c10 = matrix_matrix_add(prod5, prod6, target_rank)
        result_node.children.append(c10)
        
        prod7 = matrix_matrix_mult(v.children[2], w.children[1], target_rank)
        prod8 = matrix_matrix_mult(v.children[3], w.children[3], target_rank)
        c11 = matrix_matrix_add(prod7, prod8, target_rank)
        result_node.children.append(c11)
        
        return result_node
    
    return v

def generate_3d_mesh_matrix(k):
    n = 2**k
    N = n * n * n
    A = np.zeros((N, N))
    
    def vertex_index(i, j, k_idx):
        return i * n * n + j * n + k_idx
    
    for i in range(n):
        for j in range(n):
            for k_idx in range(n):
                v_idx = vertex_index(i, j, k_idx)
                neighbors = []
                
                if i > 0: neighbors.append(vertex_index(i-1, j, k_idx))
                if i < n-1: neighbors.append(vertex_index(i+1, j, k_idx))
                if j > 0: neighbors.append(vertex_index(i, j-1, k_idx))
                if j < n-1: neighbors.append(vertex_index(i, j+1, k_idx))
                if k_idx > 0: neighbors.append(vertex_index(i, j, k_idx-1))
                if k_idx < n-1: neighbors.append(vertex_index(i, j, k_idx+1))
                
                for neighbor in neighbors:
                    A[v_idx, neighbor] = np.random.uniform(0.1, 1.0)
    
    return A