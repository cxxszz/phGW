import numpy as np
from lib.matmul import diag_matmul_np, matmul_diag_np


def sinkhorn(C: np.ndarray, p: np.ndarray, q: np.ndarray, eta: float, T=10):
    K = np.exp(-C / eta)
    u = np.ones(p.shape)
    for _ in range(T):
        v = q / (K.T @ u)
        u = p / (K @ v)
    tmp = diag_matmul_np(u, K)
    return matmul_diag_np(tmp, v)


def normalize(K: np.ndarray, p: np.ndarray, q: np.ndarray, T=10):
    u = np.ones(p.shape)
    for _ in range(T):
        v = q / (K.T @ u)
        u = p / (K @ v)
    tmp = diag_matmul_np(u, K)
    return matmul_diag_np(tmp, v)
