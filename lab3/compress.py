from scipy.linalg import svd
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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