import numpy as np
from scipy.sparse import csr_matrix
import torch
from collections import Counter
from numpy import linalg as la
import networkx as nx
from sortedcontainers import SortedSet
from copy import deepcopy as dc
import csv
from scipy import signal as sg

div_prec = 1e-16


def set_zero_csr_sym(x: csr_matrix, indices1: SortedSet, indices2: SortedSet):
    res = dc(x)
    row_indices, col_indices = x.nonzero()
    nnz = len(row_indices)
    for i in range(nnz):
        r, c = row_indices[i], col_indices[i]
        if r in indices1 and c in indices2:
            res[r, c] = 0
            res[c, r] = 0
    res.eliminate_zeros()
    return res


def list_set_zero_csr_sym(x: csr_matrix, l1: list, l2: list):
    mask = np.ones(x.shape)
    for block_i in range(len(l1)):
        indices1, indices2 = l1[block_i], l2[block_i]
        mask[np.ix_(indices1, indices2)] = 0
        mask[np.ix_(indices2, indices1)] = 0
    res = x.multiply(mask)
    res.eliminate_zeros()
    return res


def slice_assign(big: torch.Tensor, small: torch.Tensor, indices_s: list, indices_t: list):
    # for i, index_s in enumerate(indices_s):
    #     for j, index_t in enumerate(indices_t):
    #         big[index_s, index_t] = small[i, j]
    big_flattened = big.view(-1)
    flattened_idx = [i * big.shape[1] + j for i in indices_s for j in indices_t]
    big_flattened[flattened_idx] = small.view(-1)
    return big


def trans_type_corre(trans_type_ary: np.ndarray, indices_s: list, indices_t: list, mu_s_np: np.ndarray, nt: int):
    corre_list = []
    ext_trans = np.concatenate((trans_type_ary, (mu_s_np[indices_s] - np.sum(trans_type_ary, axis=1)).reshape(-1, 1)),
                               axis=1)
    for i in range(trans_type_ary.shape[0]):
        est = np.argmax(ext_trans[i])
        if est == ext_trans.shape[1] - 1:
            corre_list.append(nt)
        else:
            corre_list.append(indices_t[est])
    return corre_list


def trans_type_corre_no_expan(trans_type_ary: np.ndarray, indices_s: list, indices_t: list):
    corre_list = []
    for i in range(trans_type_ary.shape[0]):
        est = np.argmax(trans_type_ary[i])
        corre_list.append(indices_t[est])
    return corre_list


def n_largest_indices(a: np.array, n=None):
    if n is None:
        n = len(a)
    ranked = np.argsort(a)
    largest_indices = ranked[::-1][:n]
    return largest_indices


def get_nei(w, i: int):
    if isinstance(w, np.ndarray):
        return list(np.nonzero(w[i])[0])
    elif isinstance(w, csr_matrix):
        return list(w[i].nonzero()[1])
    else:
        raise NotImplementedError


def sym_w2edge_index(w):
    """
    only processes a symmetric weight matrix
    :param w: an array, n * n
    :return: an array, 2 * num_edges
    """
    n = w.shape[0]
    l = []
    for i in range(n):
        for j in range(i + 1, n):
            if w[i][j]:
                l.append([i, j])
                l.append([j, i])
    l = np.array(l)
    return l.T


def row_nnz(x):
    """

    :param x: array or torch.Tensor, 2d shape
    :return:
    """
    # Count the number of non-zeros in each row
    if isinstance(x, np.ndarray):
        return (x != 0).sum(axis=1)
    elif isinstance(x, torch.Tensor):
        return (x != 0).sum(dim=1)
    else:
        raise TypeError
    # if isinstance(x, np.ndarray):
    #     return np.sum(x, axis=1)
    # elif isinstance(x, torch.Tensor):
    #     return torch.sum(x, dim=1)
    # else:
    #     raise TypeError


def argmax2d(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)


def most_common(l, k):
    """
    Find the k most common elements
    :param l:
    :param k:
    :return:
    """
    if len(l) == 0:
        return []
    l_c = Counter(l)
    l_tmp = l_c.most_common(k)
    freqs = [q[0] for q in l_tmp]  # frequent items
    res = []
    for i in range(k):
        res.append(freqs[i % len(freqs)])
    return res


def dist(a, b, dist_type="l2"):
    if dist_type == "l2":
        if isinstance(a, np.ndarray):
            return la.norm(a - b)
        elif isinstance(a, torch.Tensor):
            return float(torch.norm(a - b))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def array_analyze(a):
    assert isinstance(a, np.ndarray)
    a = a.flatten()
    l = []
    for i in range(len(a)):
        if a[i]:
            l.append(a[i])
    l = np.array(l)
    return np.min(l), np.median(l), np.max(l)


def array_list(a):
    res = []
    for l in a[0]:
        res.extend(l)
    return np.array(res)


def list_list(ll):
    res = []
    for l in ll:
        res.extend(l)
    res = np.array(res)
    return np.reshape(res, (1, -1))


def cos_sim(a: np.ndarray, b: np.ndarray):
    """

    :param a: (d,) array
    :param b: (d,) array
    :return:
    """
    den = la.norm(a) * la.norm(b)
    if den == 0:
        return 0
    else:
        return np.dot(a, b) / den


def row_norm(x: torch.Tensor):
    """
    normalize each row of x
    :param x: a 2d tensor
    :return:
    """
    return torch.t(torch.t(x) / (div_prec + torch.norm(x, dim=1)))


def array2g_nx(w: np.ndarray):
    """

    :param w: symmetric (n,n) array
    :return:
    """
    n = len(w)  # number of nodes
    edges = []
    for i in range(n):
        flag = False  # has not found neighbors
        for j in range(i + 1, n):
            if w[i, j] > 0:
                flag = True  # has a neighbor
                edges.append((i, j))
        if flag is False:
            while True:
                j = np.random.choice(n)
                if i != j:
                    break
            edges.append((i, j))
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


# ================================ plotting ===============================


def one_record(record):
    return float(record[2])


def smooth(x, half_size=3):
    win_size = 3 * half_size + 1
    h = np.ones(win_size) / win_size
    first_value = x[0]
    final_value = x[-1]
    first_cat = np.array([first_value] * half_size)
    final_cat = np.array([final_value] * half_size)
    catted = np.concatenate((first_cat, x, final_cat))
    unclipped = sg.convolve(catted, h, mode="same")
    return unclipped[half_size:-half_size]


def non_decreasing(x):
    y = dc(x)
    for i in range(1, len(x)):
        y[i] = np.max(x[0:i])
    return y


def read_csv(path):
    with open(path, newline="") as csvfile:
        data = list(csv.reader(csvfile))
    return np.array(list(map(
        one_record, data[1:]
    )))


if __name__ == '__main__':
    x = np.array([[1, 2, 0],
                  [3, 0, 0]])
    print(row_nnz(x))
    x = torch.from_numpy(x)
    print(row_nnz(x))
    print(argmax2d(x))
