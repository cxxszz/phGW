import torch
import time
import numpy as np
from copy import deepcopy as dc
from numpy import linalg as la
from scipy.sparse import csr_matrix
from sortedcontainers import SortedSet
from lib.srGW import md_semirelaxed_gromov_wasserstein
from lib.factorization import low_rank_app
from lib.matmul import matmul_diag_np, diag_matmul_np
from lib.util import get_nei, set_zero_csr_sym


def srgw_partition(C: torch.Tensor, mu: torch.Tensor, n_par: int, gamma_entropy: float, device='cpu',
                   dtype=torch.float32, seed=0):
    """

    :param C:
    :param mu:
    :param n_par: the number of clusters the graph is partitioned into
    :param gamma_entropy:
    :return:
    """
    C_bary = torch.eye(n_par, device=device, dtype=dtype)
    # trans, loss = md_semirelaxed_gromov_wasserstein(C1=C, p=mu, C2=C_bary, gamma_entropy=gamma_entropy, device=device,
    #                                                 dtype=dtype)
    print("seed={}".format(seed))
    trans, loss = md_semirelaxed_gromov_wasserstein(C1=C, p=mu, C2=C_bary, gamma_entropy=gamma_entropy, device=device,
                                                    dtype=dtype, init_mode="random", seed=seed)
    # print("trans={}, loss={}".format(trans, loss))
    # TODO use multiple optimization techniques to find the best partitioning (according to loss and AMI)
    tmp = torch.argmax(trans, dim=1)
    membership = [int(index) for index in tmp]
    return np.array(membership)


def obtain_indices_d(C: torch.Tensor, mu: torch.Tensor, n_par: int, gamma_entropy: float, device='cpu',
                     dtype=torch.float32):
    while True:
        seed = np.random.randint(0, 10000)
        membership = srgw_partition(C=C, mu=mu, n_par=n_par, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                    seed=seed)
        indices_d = {}
        for i in range(n_par):
            indices_d[i] = np.where(membership == i)[0]
            print("cluster {} has shape {}".format(i, len(indices_d[i])))
            if len(indices_d[i]) <= 1:
                indices_d.pop(i)
                break
        if len(indices_d) == n_par:  # the partition is successful
            break
    return indices_d


def obtain_indices_d_recur(A: torch.Tensor, n_par: int, par_level: int, gamma_entropy: float, device='cpu',
                           dtype=torch.float32, node_limit=200):
    def is_large(X: tuple, node_limit=node_limit):
        sub_n = len(X[2])
        return sub_n > node_limit

    def insert(X: tuple, sub_A_final: list, sub_p_final: list, sub_indices_final: list):
        sub_A_final.append(X[0])  # sub_A
        sub_p_final.append(X[1])  # sub_p
        sub_indices_final.append(X[2])  # sub_indices
        return sub_A_final, sub_p_final, sub_indices_final

    def partition(X: tuple, n_par=n_par):
        A, p, real_indices = X
        while True:
            seed = np.random.randint(0, 10000)
            membership = srgw_partition(A, p, n_par=n_par, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                        seed=seed)
            indices_d = {}
            for i in range(n_par):
                indices_d[i] = np.where(membership == i)[0]
                print("cluster {} has shape {}".format(i, len(indices_d[i])))
                if len(indices_d[i]) <= 1:
                    indices_d.pop(i)
                    break
            if len(indices_d) == n_par:  # the partition is successful
                break
        X_list = []
        for i in range(n_par):
            indices = indices_d[i]
            sub_A = A[indices][:, indices]
            sub_p = p[indices]
            sub_p = sub_p / torch.sum(sub_p)
            sub_indices = real_indices[indices]
            X_list.append((
                sub_A, sub_p, sub_indices
            ))
        return X_list

    sub_A_final, sub_p_final, sub_indices_final = [], [], []
    p = torch.sum(A, dim=1)
    p = p / torch.sum(p)
    n = len(p)
    que = [(A, p, np.arange(n))]
    cur_par_level = 0
    while cur_par_level < par_level:
        que_new = []
        while len(que):
            # print("que_new length {}, final length {}".format(len(que_new), len(sub_As_final)))
            X_cur = que[0]
            que.pop(0)
            if is_large(X_cur, node_limit=node_limit):
                que_new.extend(partition(X_cur, n_par=n_par))
            else:
                sub_A_final, sub_p_final, sub_indices_final = insert(X_cur, sub_A_final, sub_p_final, sub_indices_final)
        if len(que_new) == 0:
            break
        else:
            que = dc(que_new)
            cur_par_level += 1
    for X in que:
        sub_A_final.append(X[0])
        sub_p_final.append(X[1])
        sub_indices_final.append(X[2])
    indices_d = {}
    for i, indices in enumerate(sub_indices_final):
        indices_d[i] = indices
    return indices_d


class h_matrix(object):
    """
    one-level block tree for the adjcancy data
    sparsity ==>> inter-cluster
    low-rankness ==>> intra-cluster
    """

    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32):
        """save particles and matrix"""
        self.n = w.shape[0]
        self.gamma_entropy, self.device, self.dtype = gamma_entropy, device, dtype
        self.n_par, self.rank = n_par, rank
        self.build(w)

    def build(self, w: np.ndarray):
        w_torch = torch.from_numpy(w).to(self.device).type(self.dtype)
        mu = torch.sum(w_torch, dim=1).to(self.device).type(self.dtype)
        mu = mu / torch.sum(mu)
        self.indices_d = obtain_indices_d(C=w_torch, mu=mu, n_par=self.n_par, gamma_entropy=self.gamma_entropy,
                                          device=self.device, dtype=self.dtype)
        self.n_total_par = len(self.indices_d)
        self.matrix = {}
        for i in range(self.n_total_par):
            for j in range(self.n_total_par):
                key = "{}_{}".format(i, j)
                indices_i, indices_j = self.indices_d[i], self.indices_d[j]
                if i == j:
                    self.matrix[key] = low_rank_app(w[indices_i][:, indices_j], rank=self.rank)
                else:
                    self.matrix[key] = csr_matrix(w[indices_i][:, indices_j])
        return self


class h_matrix1csr(h_matrix):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32):
        super(h_matrix1csr, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype, n_par=n_par,
                                           rank=rank)

    def build(self, w: np.ndarray):
        w_torch = torch.from_numpy(w).to(self.device).type(self.dtype)
        mu = torch.sum(w_torch, dim=1).to(self.device).type(self.dtype)
        mu = mu / torch.sum(mu)
        self.indices_d = obtain_indices_d(C=w_torch, mu=mu, n_par=self.n_par, gamma_entropy=self.gamma_entropy,
                                          device=self.device, dtype=self.dtype)
        self.n_total_par = len(self.indices_d)
        self.matrix = {}
        for i in range(self.n_total_par):
            indices_i = self.indices_d[i]
            k_local = min(self.rank, len(indices_i) - 1)
            U, s, VT = low_rank_app(w[indices_i][:, indices_i], rank=k_local)
            self.matrix[i] = (U, diag_matmul_np(s, VT))
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for i in range(self.n_total_par):
            indices_i = self.indices_d[i]
            tmp[np.ix_(indices_i, indices_i)] = 0
        # print("count_nonzero", np.count_nonzero(tmp))
        self.inter_csr = csr_matrix(tmp)
        del w_torch
        return self

    def rdot_matrix(self, X: np.ndarray):
        """
        self @ X
        :param X: (n, n1) array
        :return: (n, n1) array
        """
        res = self.inter_csr @ X
        # res = np.zeros(X.shape)
        for i in range(self.n_total_par):
            X_i = X[self.indices_d[i]]
            U, sVT = self.matrix[i]
            tmp = sVT @ X_i
            res[self.indices_d[i]] += U @ tmp
        return res

    def ldot_matrix(self, X: np.ndarray):
        res = X @ self.inter_csr
        for i in range(self.n_total_par):
            X_i = X[:, self.indices_d[i]]
            U, sVT = self.matrix[i]
            tmp = X_i @ U
            res[:, self.indices_d[i]] += tmp @ sVT
        return res


def h_todense(h: h_matrix1csr):
    x = h.inter_csr.todense()
    for i in range(h.n_par):
        indices_i = h.indices_d[i]
        U, sVT = h.matrix[i]
        x[np.ix_(indices_i, indices_i)] = U @ sVT
    return x


class h_matrix_recur(h_matrix1csr):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200):
        self.par_level = par_level
        self.node_limit = node_limit
        super(h_matrix_recur, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype, n_par=n_par,
                                             rank=rank)

    def build(self, w: np.ndarray):
        w_torch = torch.from_numpy(w).to(self.device).type(self.dtype)
        self.indices_d = obtain_indices_d_recur(A=w_torch, n_par=self.n_par, par_level=self.par_level,
                                                gamma_entropy=self.gamma_entropy, device=self.device, dtype=self.dtype,
                                                node_limit=self.node_limit)
        self.n_total_par = len(self.indices_d)
        del w_torch
        self.matrix = {}
        for i in range(self.n_total_par):
            indices_i = self.indices_d[i]
            k_local = min(self.rank, len(indices_i) - 1)
            U, s, VT = low_rank_app(w[indices_i][:, indices_i], rank=k_local)
            self.matrix[i] = (U, diag_matmul_np(s, VT))
        self.calc_inter_csr(w)
        return self

    def calc_inter_csr(self, w: np.ndarray):
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for i in range(self.n_total_par):
            indices_i = self.indices_d[i]
            tmp[np.ix_(indices_i, indices_i)] = 0
        # print("count_nonzero", np.count_nonzero(tmp))
        self.inter_csr = csr_matrix(tmp)
        del tmp
        return self


class h_matrix_prune(h_matrix_recur):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200, prune_lda=1.0):
        self.prune_lda = prune_lda
        super(h_matrix_prune, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype, n_par=n_par,
                                             rank=rank, par_level=par_level, node_limit=node_limit)

    def calc_inter_csr(self, w: np.ndarray):
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for c in range(self.n_total_par):
            indices_c = self.indices_d[c]
            tmp[np.ix_(indices_c, indices_c)] = 0
        # ============================================= prune ==================================================
        start = time.time()
        nei_d, nei_d_prune = {}, {}
        for i in range(w.shape[0]):  # obtaining neighbors
            nei_d[i] = SortedSet(get_nei(w, i))
            nei_d_prune[i] = {}
            for c in range(self.n_total_par):
                indi_other = self.indices_d[c]
                if i not in indi_other:
                    nei_d_prune[i][c] = dc(nei_d[i])
                    for j in indi_other:
                        if w[i][j]:
                            nei_d_prune[i][c].remove(j)
                    # nei_d_prune[i][c] = nei_d[i].difference(indi_other)
        print("obtaining neighbors takes time {}".format(time.time() - start))
        start = time.time()
        prune_p = {}  # prune probabilities
        for c1 in range(self.n_total_par):  # calculaing the prune probabilities for each inadmissible block
            prune_p[c1] = {}
            for c2 in range(c1 + 1, self.n_total_par):
                prune_p[c1][c2] = 0
                indices1, indices2 = self.indices_d[c1], self.indices_d[c2]
                for node11 in range(len(indices1)):
                    for node12 in range(node11 + 1, len(indices1)):
                        i1, i2 = indices1[node11], indices1[node12]
                        try:
                            sim = len(nei_d_prune[i1][c2].intersection(nei_d_prune[i2][c2])) / len(
                                nei_d_prune[i1][c2].union(nei_d_prune[i2][c2]))
                        except ZeroDivisionError:
                            sim = 0
                        try:
                            sim0 = len(nei_d[i1].intersection(nei_d[i2])) / len(nei_d[i1].union(nei_d[i2]))
                        except ZeroDivisionError:
                            sim0 = 0
                        if sim - sim0 > prune_p[c1][c2]:
                            prune_p[c1][c2] = sim - sim0
                for node21 in range(len(indices2)):
                    for node22 in range(node21 + 1, len(indices2)):
                        j1, j2 = indices2[node21], indices2[node22]
                        try:
                            sim = len(nei_d_prune[j1][c1].intersection(nei_d_prune[j2][c1])) / len(
                                nei_d_prune[j1][c1].union(nei_d_prune[j2][c1]))
                        except ZeroDivisionError:
                            sim = 0
                        try:
                            sim0 = len(nei_d[j1].intersection(nei_d[j2])) / len(nei_d[j1].union(nei_d[j2]))
                        except ZeroDivisionError:
                            sim0 = 0
                        if sim - sim0 > prune_p[c1][c2]:
                            prune_p[c1][c2] = sim - sim0
        print("calculating pruning probabilities takes time {}".format(time.time() - start))
        for c1 in range(self.n_total_par):  # all preparations finished
            for c2 in range(c1 + 1, self.n_total_par):
                prune_p[c1][c2] = np.exp(-self.prune_lda * prune_p[c1][c2] ** 2)
                x = np.random.uniform(0, 1)
                if x <= prune_p[c1][c2]:
                    indices1, indices2 = self.indices_d[c1], self.indices_d[c2]
                    tmp[np.ix_(indices1, indices2)] = 0
                    tmp[np.ix_(indices2, indices1)] = 0
        print("prune_p={}".format(prune_p))
        self.inter_csr = csr_matrix(tmp)
        return self


class h_matrix_prune_dm(h_matrix_prune):
    """
    The significance of an inadmissible block is determined by the sum of the degree products of end points of each edge
    """

    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200, prune_lda=1.0):
        super(h_matrix_prune_dm, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                                n_par=n_par, rank=rank, par_level=par_level, node_limit=node_limit,
                                                prune_lda=prune_lda)

    def calc_inter_csr(self, w: np.ndarray):
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for c in range(self.n_total_par):
            indices_c = self.indices_d[c]
            tmp[np.ix_(indices_c, indices_c)] = 0
        # ============================================= prune ==================================================
        start = time.time()
        nei_d, deg = {}, {}
        for i in range(w.shape[0]):  # obtaining neighbors
            nei_d[i] = SortedSet(get_nei(w, i))
            deg[i] = len(nei_d[i])
        print("obtaining neighbors takes time {}".format(time.time() - start))
        start = time.time()
        prune_p = {}  # prune probabilities
        for c1 in range(self.n_total_par):  # calculaing the prune probabilities for each inadmissible block
            prune_p[c1] = {}
            for c2 in range(c1 + 1, self.n_total_par):
                total_edges = 0
                prune_p[c1][c2] = 0.0
                indices1, indices2 = self.indices_d[c1], self.indices_d[c2]
                for i in indices1:
                    checkset = nei_d[i].intersection(indices2)
                    total_edges += len(checkset)
                    for j in checkset:
                        prune_p[c1][c2] += deg[i] * deg[j]
                prune_p[c1][c2] /= total_edges
        print("calculating pruning probabilities takes time {}".format(time.time() - start))
        print("prune_p exponents={}".format(prune_p))
        for c1 in range(self.n_total_par):  # all preparations finished
            for c2 in range(c1 + 1, self.n_total_par):
                prune_p[c1][c2] = np.exp(-self.prune_lda * prune_p[c1][c2])
                x = np.random.uniform(0, 1)
                if x <= prune_p[c1][c2]:
                    indices1, indices2 = self.indices_d[c1], self.indices_d[c2]
                    tmp[np.ix_(indices1, indices2)] = 0
                    tmp[np.ix_(indices2, indices1)] = 0
        print("prune_p={}".format(prune_p))
        self.inter_csr = csr_matrix(tmp)
        return self


class h_matrix_prune_edge(h_matrix_prune):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200, prune_lda=1.0):
        super(h_matrix_prune_edge, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                                  n_par=n_par, rank=rank, par_level=par_level, node_limit=node_limit,
                                                  prune_lda=prune_lda)

    def calc_inter_csr(self, w: np.ndarray):
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for c in range(self.n_total_par):
            indices_c = self.indices_d[c]
            tmp[np.ix_(indices_c, indices_c)] = 0
        # ============================================= prune ==================================================
        nei_d, deg = {}, {}
        for i in range(w.shape[0]):  # obtaining neighbors
            nei_d[i] = SortedSet(get_nei(w, i))
            deg[i] = len(nei_d[i])
        for c1 in range(self.n_total_par):
            indices1 = self.indices_d[c1]
            for c2 in range(c1 + 1, self.n_total_par):
                indices2 = self.indices_d[c2]
                for i in indices1:
                    checkset = nei_d[i].intersection(indices2)
                    for j in checkset:
                        prune_p = np.exp(-self.prune_lda * (deg[i] * deg[j]) ** 2)
                        x = np.random.uniform(0, 1)
                        if x < prune_p:
                            tmp[i][j] = 0
                            tmp[j][i] = 0
        self.inter_csr = csr_matrix(tmp)
        return self


class h_matrix_random_prune(h_matrix_prune_dm):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200, prune_lda=1.0):
        super(h_matrix_random_prune, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                                    n_par=n_par, rank=rank, par_level=par_level, node_limit=node_limit,
                                                    prune_lda=prune_lda)

    def reset_inter_csr(self):
        tmp = dc(self.inter_csr_store)
        for c1 in range(self.n_total_par):
            for c2 in range(c1 + 1, self.n_total_par):
                x = np.random.uniform(0, 1)
                if x <= self.prune_p[c1][c2]:
                    indices1, indices2 = self.indices_d_sort[c1], self.indices_d_sort[c2]
                    tmp = set_zero_csr_sym(tmp, indices1, indices2)
        self.inter_csr = tmp
        return self

    def calc_inter_csr(self, w: np.ndarray):
        self.indices_d_sort = {}
        for c in range(self.n_total_par):
            self.indices_d_sort[c] = SortedSet(self.indices_d[c])
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for c in range(self.n_total_par):
            indices_c = self.indices_d[c]
            tmp[np.ix_(indices_c, indices_c)] = 0
        self.inter_csr_store = csr_matrix(tmp)
        # ============================================= prune ==================================================
        start = time.time()
        nei_d, deg = {}, {}
        for i in range(w.shape[0]):  # obtaining neighbors
            nei_d[i] = SortedSet(get_nei(w, i))
            deg[i] = len(nei_d[i])
        print("obtaining neighbors takes time {}".format(time.time() - start))
        start = time.time()
        prune_p = {}  # prune probabilities
        for c1 in range(self.n_total_par):  # calculaing the prune probabilities for each inadmissible block
            prune_p[c1] = {}
            for c2 in range(c1 + 1, self.n_total_par):
                total_edges = 0
                prune_p[c1][c2] = 0.0
                indices1, indices2 = self.indices_d[c1], self.indices_d[c2]
                for i in indices1:
                    checkset = nei_d[i].intersection(indices2)
                    total_edges += len(checkset)
                    for j in checkset:
                        prune_p[c1][c2] += deg[i] * deg[j]
                try:
                    prune_p[c1][c2] /= total_edges
                except ZeroDivisionError:
                    prune_p[c1][c2] = 0
                prune_p[c1][c2] = np.exp(-self.prune_lda * prune_p[c1][c2])
        self.prune_p = prune_p
        print("calculating pruning probabilities takes time {}".format(time.time() - start))
        print("prune probabilities {}".format(self.prune_p))
        self.reset_inter_csr()
        return self


class h_matrix_random_prune_edge(h_matrix_prune_edge):
    def __init__(self, w: np.ndarray, gamma_entropy=0.1, device='cpu', dtype=torch.float32, n_par=2, rank=32,
                 par_level=3, node_limit=200, prune_lda=1.0):
        super(h_matrix_random_prune_edge, self).__init__(w=w, gamma_entropy=gamma_entropy, device=device, dtype=dtype,
                                                         n_par=n_par, rank=rank, par_level=par_level,
                                                         node_limit=node_limit, prune_lda=prune_lda)

    def reset_inter_csr(self):
        self.inter_csr = dc(self.inter_csr_store)
        for c1 in range(self.n_total_par):
            indices1 = self.indices_d[c1]
            for c2 in range(c1 + 1, self.n_total_par):
                indices2 = self.indices_d[c2]
                for i in indices1:
                    checkset = self.nei_d[i].intersection(indices2)
                    for j in checkset:
                        prune_p = np.exp(-self.prune_lda * self.deg[i] * self.deg[j])
                        x = np.random.uniform(0, 1)
                        if x < prune_p:
                            self.inter_csr[i, j] = 0
                            self.inter_csr[j, i] = 0
        self.inter_csr.eliminate_zeros()
        return self

    def calc_inter_csr(self, w: np.ndarray):
        tmp = dc(w)
        # print("count_nonzero", np.count_nonzero(tmp))
        for c in range(self.n_total_par):
            indices_c = self.indices_d[c]
            tmp[np.ix_(indices_c, indices_c)] = 0
        self.inter_csr_store = csr_matrix(tmp)
        self.nei_d, self.deg = {}, {}
        for i in range(w.shape[0]):  # obtaining neighbors
            self.nei_d[i] = SortedSet(get_nei(w, i))
            self.deg[i] = len(self.nei_d[i])
        self.reset_inter_csr()
        return self
