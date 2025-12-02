from scipy.linalg import svd
import os
import numpy as np
from PIL import Image, ImageDraw

eps = 1e-6

class MatrixLeaf:
    def __init__(self, A, size, U, D, V, r):
        self.size = size
        if np.all(np.abs(A) < eps):
            self.rank = 0
            self.is_zero = True
            return
        self.is_zero = False
        self.rank = r
        self.singular_vals = D[:r]
        self.U = U[:, :r]
        self.V = V[:r, :]
        

class MatrixNode:
    def __init__(self):
        self.children = []


def CreateTree(A, t_min, t_max, s_min, s_max, r, epsilon):
    A_part = A[t_min:t_max+1, s_min:s_max+1]
    U, D, V = svd(A_part)
    node: MatrixNode
    D_epsilon = D[np.where(D > epsilon)]
    if len(D_epsilon) <= r and len(D_epsilon) <= len(A_part):
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


# from scipy.linalg import svd
# from compress import CompressMatrix, draw_vis


# IMAGEFILE = 'img.bmp'

# image = Image.open(IMAGEFILE)
# im_rgb = np.array(image)

# R = im_rgb[:,:,0]
# G = im_rgb[:,:,1]
# B = im_rgb[:,:,2]

# _, D, _ = svd(R)
# # print(D)
# R_compressed = CompressMatrix(R, 1, D[0])
# G_compressed = CompressMatrix(G, 1, D[0])
# B_compressed = CompressMatrix(B, 1, D[0])

# R_im = get_color_channel(image, 0)
# draw_vis(R_compressed, R_im)
# # B_im = draw_vis(G_compressed, image)
# # G_im = draw_vis(B_compressed, image)



