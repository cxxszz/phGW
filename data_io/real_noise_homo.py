"""
Real-world correlated graphs, i.e., real noise
"""
import numpy as np
from numpy import linalg as la
import pickle
from copy import deepcopy as dc
from scipy.sparse import csc_matrix
from sortedcontainers import SortedList
import os
from data_io.real_data_homo import graph


class graph_pair_rn(object):
    def __init__(self, graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2):
        self.my_init(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)

    def my_init(self, graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2):
        self.graph_s = graph_s
        self.graph_t = graph_t
        self.ns = self.graph_s.n
        self.nt = self.graph_t.n
        self.n_overlap = n_overlap
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

        # --------------------------------- dermine anchor nodes -----------------------------
        assert anch_ratio > 0  # we know the counterparts of anch_ratio of nodes in graph_s
        self.sample_anch(anch_ratio=anch_ratio)
        return self

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


def generate_store(graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2, store=True, dataset="ppi_rn",
                   noise_level=0):
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)
    if store:
        data_path = os.path.join("data/data_bench/", dataset, "{}.p".format(noise_level))
        with open(data_path, "wb") as f:
            pickle.dump(graph_st, f)
        return graph_st, data_path
    else:
        return graph_st, None


def generate_store_oregon(graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2, store=True, dataset="oregon",
                          noise_level=0):
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)
    if store:
        data_path = os.path.join("data/data_bench/", dataset, "{}.p".format(noise_level))
        with open(data_path, "wb") as f:
            pickle.dump(graph_st, f)
        return graph_st, data_path
    else:
        return graph_st, None


def generate_store_dbp(graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2, store=True, category="zh_en"):
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)
    if store:
        data_path = os.path.join("data/data_bench/dbp", "{}.p".format(category))
        with open(data_path, "wb") as f:
            pickle.dump(graph_st, f)
        return graph_st, data_path
    else:
        return graph_st, None


def edge_correctness(graph_st: graph_pair_rn, corre: np.ndarray):
    num = 0
    ws, wt = graph_st.graph_s.w, graph_st.graph_t.w
    for i in range(len(corre)):
        iprime = corre[i]
        for j in range(i + 1, len(corre)):
            jprime = corre[j]
            if ws[i, j] > 0 and wt[iprime, jprime] > 0:
                num += 1
    return num / graph_st.ES


def symm_substru_score(graph_st: graph_pair_rn, corre: np.ndarray):
    num = 0
    ws, wt = graph_st.graph_s.w, graph_st.graph_t.w
    for i in range(len(corre)):
        iprime = corre[i]
        for j in range(i + 1, len(corre)):
            jprime = corre[j]
            if ws[i, j] > 0 and wt[iprime, jprime] > 0:
                num += 1
    den = 0
    for i in range(len(corre)):
        iprime = corre[i]
        for j in range(i + 1, len(corre)):
            jprime = corre[j]
            if wt[iprime, jprime] > 0:
                den += 1
    den = den + graph_st.ES - num
    return num / den


def report(corre: list, indices: list, graph_st: graph_pair_rn):
    assert len(corre) == len(indices), "len(corre)={}, len(indices)={}".format(len(corre), len(corre))
    cm, wm, tm, m_dummy = 0, 0, 0, 0  # the number of correct matching, wrong matching, total matching,
    # and matching to the dummy node, respectively

    for corre_index, i in enumerate(indices):
        # print("corre_index={}, i={}, corre[corre_index]={}".format(corre_index, i, corre[corre_index]))
        # print("gt[i]={}".format(graph_st.gt[i]))
        if graph_st.has_cp[i]:
            if corre[corre_index] == graph_st.gt[i]:
                cm += 1
                tm += 1
            elif corre[corre_index] == graph_st.nt:
                m_dummy += 1
            else:
                wm += 1
                tm += 1
        else:
            if corre[corre_index] == graph_st.nt:
                m_dummy += 1
            else:
                wm += 1
                tm += 1
    if tm > 0:
        result = [cm / graph_st.n_overlap, cm / tm, cm, wm, m_dummy, tm, graph_st.n_overlap, graph_st.ns]
    else:
        result = [cm / graph_st.n_overlap, 0, cm, wm, m_dummy, tm, graph_st.n_overlap, graph_st.ns]

    # ==================================== calculate EC and S3 ===========================================
    ws, wt = graph_st.graph_s.w, graph_st.graph_t.w
    # --------------------------------------- calculate EC -----------------------------------------------
    ec_num = 0
    for corre_index in range(len(indices)):
        i = indices[corre_index]
        iprime = corre[corre_index]
        for corre_index1 in range(corre_index + 1, len(indices)):
            j = indices[corre_index1]
            jprime = corre[corre_index1]
            if ws[i, j] > 0 and wt[iprime, jprime] > 0:
                ec_num += 1
    ec = ec_num / graph_st.ES
    # -------------------------------------- calculate S3 ------------------------------------------------
    s3_den = 0
    for corre_index in range(len(indices)):
        iprime = corre[corre_index]
        for corre_index1 in range(corre_index + 1, len(indices)):
            jprime = corre[corre_index1]
            if wt[iprime, jprime] > 0:
                s3_den += 1
    s3_den = s3_den + graph_st.ES - ec_num
    s3 = ec_num / s3_den
    return result, ec, s3
