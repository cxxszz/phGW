import numpy as np
from numpy import linalg as la
import torch


def mem_matmul(A: torch.Tensor, B: torch.Tensor):
    dtype = A.dtype
    device = A.device
    l, m1 = A.size()
    m2, n = B.size()
    assert m1 == m2
    res = torch.zeros((l, n), dtype=dtype, device=device)
    m = m1
    slice_size_m = 1000  # corresponds to 20000 * 20000 * 20000
    size_right_m = 20000 * 20000
    size_right = m * n
    slice_size = int(size_right_m * slice_size_m / size_right)
    # print("slice_size={}".format(slice_size))
    assert slice_size > 1, "the right matrix with size {} is too large".format((m, n))
    start = 0
    while start + slice_size < l:
        end = start + slice_size
        # print("start={}, end={}".format(start, end))
        res[start: end] = torch.matmul(A[start:end], B)
        start = end
    end = l
    res[start: end] = torch.matmul(A[start:end], B)
    return res


def two_matmul(trans: torch.Tensor, X1: torch.Tensor, X2T: torch.Tensor, Y1: torch.Tensor, Y2T: torch.Tensor):
    """
    Using matrix factorization to accelerate the computation of X @ trans @ Y, where X = X1 @ X2T and Y = Y1 @ Y2T
    :param trans:
    :param X1:
    :param X2T:
    :param Y1:
    :param Y2T:
    :return:
    """
    tmp = torch.matmul(X2T, trans)
    mid = torch.matmul(tmp, Y1)
    # del tmp  # takes very few memory
    res = X1 @ mid @ Y2T
    return res


def matmul_diag(A: torch.Tensor, d: torch.Tensor):
    """

    :param A: 2-dimensional
    :param d: 1-dimensional
    :return: A @ diag(d)
    """
    return A * d


def diag_matmul(d: torch.Tensor, A: torch.Tensor):
    """

    :param d: 2-dimensional
    :param A: 1-dimensional
    :return: diag(d) @ A
    """
    return d.unsqueeze(1) * A


def matmul_diag_np(A: np.ndarray, d: np.ndarray):
    """

    :param A: 2-dimensional
    :param d: 1-dimensional
    :return: A @ diag(d)
    """
    return A * d


def diag_matmul_np(d: np.ndarray, A: np.ndarray):
    """

    :param d: 2-dimensional
    :param A: 1-dimensional
    :return: diag(d) @ A
    """
    return d.reshape(-1, 1) * A


def matrix_inv_sqrt(Y: np.ndarray):
    U, d, VT = la.svd(Y)
    return U @ np.diag(d ** -0.5) @ VT


if __name__ == '__main__':
    d = np.array([1, 2, 3])
    x = np.array([1, 2, 3])
    print(diag_matmul_np(d, x))