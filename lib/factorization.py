import torch
from lib.matmul import matmul_diag_np, diag_matmul_np, matmul_diag, diag_matmul
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm, svds


def low_rank_app(A: csr_matrix, rank: int):
    """
    Complexity O(n_row * n_col * k)
    :param A: the matrix to be approximated and has shape (n_row, n_col)
    :param rank: the rank of approximated matrix
    :return:
    """
    U, s, VT = svds(A, k=rank, which='LM', return_singular_vectors=True)  # u (n_row, k); s (k,); vt (k, n_col)
    return U, s, VT


def svd_features(A: csr_matrix, rank: int):
    """

    :param A:
    :param rank:
    :return:
    """
    U, s, VT = svds(A, k=rank, which='LM', return_singular_vectors=True)  # u (n_row, k); s (k,); vt (k, n_col)
    return matmul_diag_np(U, s)


def rect_eye_torch(A: torch.Tensor):
    n_row = A.shape[0]
    n_col = A.shape[1]
    res = torch.zeros([n_row, n_col], device=A.device, dtype=A.dtype)
    for i in range(min(n_row, n_col)):
        res[i, i] = 1

    return res


svd_prec = 1e-5


def torch_svd(A: torch.Tensor, k=None):
    if A.size(0) > 25000:
        if k is None:
            raise ValueError("torch.svd not supported to calculate the full SVD for A has size {}".format(A.size()))
        else:
            csr_A = csr_matrix(A.detach().cpu().numpy())
            Uk, dk, VkT = svds(csr_A)
            Uk = torch.from_numpy(Uk).to(A.device).type(A.dtype)
            dk = torch.from_numpy(dk).to(A.device).type(A.dtype)
            Vk = torch.from_numpy(VkT.T).to(A.device).type(A.dtype)
            return Uk, dk, Vk
    if k is None:
        try:
            U, d, V = torch.svd(A)
        except RuntimeError:
            try:
                U, d, V = torch.svd(A + svd_prec * rect_eye_torch(A))
            except RuntimeError:
                U, d, V = torch.svd(A + rect_eye_torch(A))
        return U, d, V
    else:
        try:
            U, d, V = torch.svd(A)
            Uk, dk, Vk = U[:, 0:k], d[0:k], V[:, 0:k]
        except RuntimeError:
            try:
                U, d, V = torch.svd(A + svd_prec * rect_eye_torch(A))
            except RuntimeError:
                U, d, V = torch.svd(A + rect_eye_torch(A))
            Uk, dk, Vk = U[:, 0:k], d[0:k], V[:, 0:k]
        return Uk, dk, Vk


def low_rank_torch(A: torch.Tensor, k: int):
    """
    Complexity O(n_row * n_col * k)
    :param A: the matrix to be approximated and has shape (n_row, n_col)
    :param k: the rank of approximated matrix
    :return:
    """
    # U, d, V = torch.svd(A)
    # Uk, dk, Vk = U[:, 0:k], d[0:k], V[:, 0:k]
    Uk, dk, Vk = torch_svd(A=A, k=k)
    return Uk @ diag_matmul(dk, Vk.T), Uk, diag_matmul(dk, Vk.T)  # (n_row, n_col), (n_row,k), (k, n_col)
