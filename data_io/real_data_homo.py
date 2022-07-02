"""
Real-world data, but synthetic noise.
"""

import numpy as np
from numpy import linalg as la
import pickle
from copy import deepcopy as dc
from scipy.sparse import csc_matrix
from sortedcontainers import SortedList
import os

# seed = 1
# seed = 2
# seed = 3
seed = 4
np.random.seed(seed)


class graph(object):
    def __init__(self, w: np.ndarray, lying=None):
        self.my_init(w=w, lying=lying)

    def my_init(self, w: np.ndarray, lying=None):
        self.w = w
        self.n = len(w)
        self.lying = lying  # underlying node, i.e., the node order in the full graph

    def get_nei(self, id: int):
        """

        :param id: node id, starting from 0
        :return: a neighbour list
        """
        res = []
        for i in range(self.n):
            if self.w[id][i]:
                res.append(i)
        return res


def overlap_sample_list(l, n_overlap: int, ns: int, nt: int):
    """
    BE CAREFUL with the DUPLICATEs
    get two lists that have n_overlap common elements, and have length ns and nt respectively.
    :param l:
    :param n_overlap:
    :param ns:
    :param nt:
    :return:
    """
    n_total = len(l)
    overlap_ary = np.random.choice(n_total, n_overlap, replace=False)
    l_unique = dc(l)
    for i in range(n_overlap):
        # remove(value) removes a value from the list. ValueError raised if value is not a member
        l_unique.remove(overlap_ary[i])
    unique_s_ary = np.random.choice(l_unique, ns - n_overlap, replace=False)
    for i in range(ns - n_overlap):
        # remove(value) removes a value from the list. ValueError raised if value is not a member
        l_unique.remove(unique_s_ary[i])
    unique_t_ary = np.random.choice(l_unique, nt - n_overlap, replace=False)
    ary_s = np.concatenate((overlap_ary, unique_s_ary))
    ary_t = np.concatenate((overlap_ary, unique_t_ary))
    np.random.shuffle(ary_s)
    np.random.shuffle(ary_t)
    return ary_s, ary_t


def subset_rand(full_graph: graph, sub_ary: np.ndarray):
    # n_sub = len(sub_list)
    w = full_graph.w[sub_ary, :][:, sub_ary]
    return graph(w=w, lying=sub_ary)


def subset_rw(full_graph: graph, n_sub: int, sub_list_in=None, exclu_so=None, overlap=True):
    """

    :param full_graph:
    :param n_sub: the number of nodes in the sub graph
    :return: sub_graph
    """
    n_super = len(full_graph.w)
    if sub_list_in is None:  # finding overlapping nodes
        sub_list = []
    else:
        sub_list = dc(sub_list_in)
    cur_num = 0  # current number of nodes in the sub
    sub_so = SortedList(sub_list)
    unique_so = SortedList()  # ensures (1-over_ratio) nodes belong to graph_s exclusively

    #####################################################################################################
    ## choosing the first node in the subgraph
    if len(sub_so) == 0:  # sub_list is empty
        while (True):
            i = np.random.choice(n_super)
            if np.sum(full_graph.w[i]) > 0:  # not isolated node
                break
    else:
        # random walk
        while (True):
            start_point = np.random.choice(sub_list)
            try:
                i = np.random.choice(full_graph.get_nei(start_point))  # node i is possibly not sampled
            except ValueError:
                continue
            else:
                break
    ####################################################################################################
    ## getting the subgraph
    while len(sub_so) < n_sub:
        if i not in sub_so:
            if exclu_so is None or i not in exclu_so:
                sub_list.append(i)
                sub_so.add(i)
                unique_so.add(i)
                print("sub_list={},\nsub_so={},\nunique_so={}".format(sub_list, sub_so, unique_so))
            # random walk
            while (True):
                start_point = np.random.choice(sub_list)
                try:
                    i = np.random.choice(full_graph.get_nei(start_point))  # node i is possibly not sampled
                except ValueError:
                    continue
                else:
                    break

    if overlap:
        return sub_list, unique_so

    else:
        sub_ary = np.array(sub_list)
        return sub_ary, unique_so


class graph_pair(object):
    def __init__(self, full_graph: graph, n_overlap: int, ns: int, nt: int, anch_ratio=0.2, dist_value=0.0):
        """
        non-attributed graphs
        :param full_graph: a non-attributed graph
        :param n_overlap: int
        :param ns: int
        :param nt: int
        :param anch_ratio: float
        :param dist_value: whether adding edge noise
        """
        self.my_init(full_graph=full_graph, n_overlap=n_overlap, ns=ns, nt=nt, anch_ratio=anch_ratio,
                     dist_value=dist_value)

    def my_init(self, full_graph: graph, n_overlap: int, ns: int, nt: int, anch_ratio=0.2, dist_value=0.0):
        print("number of nodes:")
        print(n_overlap, ns, nt)
        # assert n_overlap < ns
        # assert n_overlap < nt
        # assert ns <= nt
        n_total = full_graph.n
        total_node_list = list(range(n_total))
        self.n_overlap = n_overlap
        self.ns, self.nt = ns, nt
        self.dist_value = dist_value  # the ratio of noisy edges

        # taking subset of the full graph
        if ns == n_total and nt == n_total:
            ary_s = np.arange(n_total)
            ary_t = np.arange(n_total)
            np.random.shuffle(ary_s)
            np.random.shuffle(ary_t)
        else:
            ary_s, ary_t = overlap_sample_list(l=total_node_list, n_overlap=self.n_overlap, ns=self.ns, nt=self.nt)
        self.graph_s = subset_rand(full_graph=full_graph, sub_ary=ary_s)
        self.graph_t = subset_rand(full_graph=full_graph, sub_ary=ary_t)
        assert self.ns == self.graph_s.n
        assert self.nt == self.graph_t.n

        # calculating has_cp and gt that will be used to evaluate the estimated node correspondence
        self.has_cp = np.zeros(self.ns, dtype=bool)  # False initially
        self.gt = np.zeros(self.ns, dtype=np.int) + self.nt
        for i in range(self.ns):
            for j in range(self.nt):
                if self.graph_s.lying[i] == self.graph_t.lying[j]:
                    #  if node i of the source graph and node j of the target graph is actually the same node
                    self.has_cp[i] = True
                    self.gt[i] = j
                    break
        assert np.sum(self.gt < self.nt) == self.n_overlap

        # ----------------------------------- dermine anchor nodes -------------------------------------------
        assert anch_ratio > 0  # we know the counterparts of anch_ratio of nodes in graph_s
        self.sample_anch(anch_ratio=anch_ratio)
        self.add_edge_noise()
        return self

    def add_edge_noise(self):
        if self.dist_value > 0:  ## adding noise
            # n_add_edge = int(self.nt * (self.nt - 1) / 2 * self.dist_value)
            n_total_edge = np.sum((self.graph_t.w > 0).astype(np.float)) / 2
            n_add_edge = int(n_total_edge * self.dist_value)
            i = 0
            while i < n_add_edge:
                j1, j2 = np.random.choice(self.nt, 2, replace=False)
                if self.graph_t.w[j1][j2] == 0:
                    self.graph_t.w[j1][j2] = np.random.uniform(1e-3, 2e-3)
                    self.graph_t.w[j2][j1] = self.graph_t.w[j1][j2]
                    i += 1
        return self

    def result_eval(self, corre, print_resu=False):
        """

        :param corre: The estimated target node for each node in the source graph
        :return:
        """
        cm, wm, tm, m_dummy = 0, 0, 0, 0  # the number of correct matching, wrong matching, total matching,
        # and matching to the dummy node, respectively
        for i in range(self.ns):
            if self.has_cp[i]:
                if corre[i] == self.gt[i]:
                    cm += 1
                    tm += 1
                elif corre[i] == self.nt:
                    m_dummy += 1
                else:
                    wm += 1
                    tm += 1
            else:
                if corre[i] == self.nt:
                    m_dummy += 1
                else:
                    wm += 1
                    tm += 1
        if tm > 0:
            return cm / self.n_overlap, cm / tm, cm, wm, m_dummy, tm, self.n_overlap, self.ns
        else:
            return cm / self.n_overlap, 0, cm, wm, m_dummy, tm, self.n_overlap, self.ns

    def sample_anch(self, anch_ratio):
        anch_s = np.random.choice(range(self.ns), int(self.ns * anch_ratio + 0.5), replace=False)
        anch_t = np.zeros(anch_s.shape, dtype=np.int)
        anch_cp = np.zeros(anch_s.shape, dtype=np.int)
        for i, index in enumerate(anch_s):
            if self.has_cp[index]:
                anch_t[i] = self.gt[index]
                anch_cp[i] = 1
            else:
                anch_t[i] = self.nt
                anch_cp[i] = 0
        self.anch_s, self.anch_t = anch_s, anch_t
        self.anch_cp = anch_cp
        return self


class graph_pair_rw(graph_pair):
    def __init__(self, full_graph: graph, n_overlap: int, ns: int, nt: int, anch_ratio=0.2, dist_value=0.0):
        super(graph_pair_rw, self).__init__(full_graph=full_graph, n_overlap=n_overlap, ns=ns, nt=nt,
                                            anch_ratio=anch_ratio, dist_value=dist_value)

    def my_init(self, full_graph: graph, n_overlap: int, ns: int, nt: int, anch_ratio=0.2, dist_value=0.0):
        print("number of nodes:")
        print(n_overlap, ns, nt)
        # assert n_overlap < ns
        # assert n_overlap < nt
        # assert ns <= nt
        n_total = full_graph.n
        total_node_list = list(range(n_total))
        self.n_overlap = n_overlap
        self.ns, self.nt = ns, nt

        # taking subset of the full graph
        if ns == n_total and nt == n_total:
            ary_s = np.arange(n_total)
            ary_t = np.arange(n_total)
            np.random.shuffle(ary_s)
            np.random.shuffle(ary_t)
        else:
            overlap_list, _ = subset_rw(full_graph=full_graph, n_sub=n_overlap, sub_list_in=None, overlap=True)
            print("overlap_list={}".format(overlap_list))
            ary_s, unique_s_so = subset_rw(full_graph=full_graph, n_sub=ns, sub_list_in=overlap_list, exclu_so=None,
                                           overlap=False)
            ary_t, unique_t_so = subset_rw(full_graph=full_graph, n_sub=nt, sub_list_in=overlap_list,
                                           exclu_so=unique_s_so, overlap=False)
            for i in unique_s_so:
                assert i not in unique_t_so
            # ary_s, ary_t = overlap_sample_list(l=total_node_list, n_overlap=self.n_overlap, ns=self.ns, nt=self.nt)

        self.graph_s = subset_rand(full_graph=full_graph, sub_ary=ary_s)
        self.graph_t = subset_rand(full_graph=full_graph, sub_ary=ary_t)
        assert self.ns == self.graph_s.n
        assert self.nt == self.graph_t.n

        # calculating has_cp and gt that will be used to evaluate the estimated node correspondence
        self.has_cp = np.zeros(self.ns, dtype=bool)  # False initially
        self.gt = np.zeros(self.ns, dtype=np.int) + self.nt
        for i in range(self.ns):
            for j in range(self.nt):
                if self.graph_s.lying[i] == self.graph_t.lying[j]:
                    #  if node i of the source graph and node j of the target graph is actually the same node
                    self.has_cp[i] = True
                    self.gt[i] = j
                    break
        assert np.sum(self.gt < self.nt) == self.n_overlap

        # dermine anchor nodes
        assert anch_ratio > 0  # we know the counterparts of anch_ratio of nodes in graph_s
        self.sample_anch(anch_ratio=anch_ratio)
        return self


def generate_store(full_graph, n_overlap=800, ns=900, nt=904, anch_ratio=0.2, dist_value=0.0, store=True,
                   dataset="ppi"):
    graph_st = graph_pair(full_graph=full_graph, n_overlap=n_overlap, ns=ns, nt=nt,
                          anch_ratio=anch_ratio, dist_value=dist_value)
    # graph_st = graph_pair_rw(full_graph=full_graph, n_overlap=n_overlap, ns=ns, nt=nt, anch_ratio=anch_ratio,
    #                          dist_value=dist_value)
    ## random walk based subsetting may not walk for some graphs including PPI
    if store:
        data_path = os.path.join("data", dataset,
                                 "{}_{}_{}_{}_{}_{}.p".format(n_overlap, ns, nt, int(100 * anch_ratio), seed,
                                                              dist_value))
        with open(data_path, "wb") as f:
            pickle.dump(graph_st, f)
        return graph_st, data_path
    else:
        return graph_st, None
