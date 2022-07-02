"""
IO for Amazon
"""

import numpy as np
import os
import pickle
from copy import deepcopy as dc
from sortedcontainers import SortedDict
from scipy.sparse import csr_matrix, save_npz, load_npz
from data_io.real_noise_homo import graph, graph_pair_rn
from data_io.part import part_dynamic, subgraph
import csv


def main_api_type():
    # ====================== first completely load the data ====================
    N = 10099
    n = 1000
    # ------------------------------ edges and edge_types ----------------------------
    dataset = "amazon"
    weights = []
    row = []
    col = []
    edge_types = SortedDict()
    with open("data/raw/amazon/link.dat", "r") as f:
        for line_id, x in enumerate(f):
            y = x.split('\t')
            print(y, len(y))
            weights.append(1)
            weights.append(1)
            i, j = int(y[0]), int(y[1])
            row.append(i)
            row.append(j)
            col.append(j)
            col.append(i)
            edge_types["{}_{}".format(i, j)] = int(y[2])
            edge_types["{}_{}".format(j, i)] = int(y[2])
    w_csr = csr_matrix((weights, (row, col)), shape=(N, N))
    # ==================== now extract a sub-graph with reasonable size =========
    indices = subgraph(w=w_csr, max_nodes=n)
    np.random.shuffle(indices)
    print(indices)

    # =================== generate graph_s and graph_t ==========================
    w = w_csr[indices][:, indices].toarray()
    ratios = (1.0, 0.0, 0.0, 0.01)
    # ratios = (0.9, 0.04, 0.06, 0.01)
    # ratios = (0.8, 0.09, 0.11, 0.01)
    # ratios = (0.7, 0.14, 0.16, 0.01)
    # ratios = (0.6, 0.18, 0.22, 0.01)
    # ratios = (0.5, 0.24, 0.26, 0.01)
    # ratios = (0.4, 0.3, 0.3, 0.01)
    if ratios[0] < 1:
        ws, lying_s, wt, lying_t, n_overlap = part_dynamic(w=w, over_ratio=ratios[0], s_ratio=ratios[1],
                                                           t_ratio=ratios[2])
    else:
        n_overlap = n
        lying_s = indices
        ws = w
        perm = np.arange(n)
        lying_t = indices[perm]
        wt = w[perm][:, perm]

    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    ns, nt = graph_s.n, graph_t.n
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=ratios[3])
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    edge_types_s, edge_types_t = SortedDict(), SortedDict()
    for i in range(ns):
        for j in range(i + 1, ns):
            if ws[i, j]:
                lying_i, lying_j = lying_s[i], lying_s[j]
                edge_types_s["{}_{}".format(i, j)] = edge_types["{}_{}".format(lying_i, lying_j)]
                edge_types_s["{}_{}".format(j, i)] = edge_types["{}_{}".format(lying_j, lying_i)]
    for i1 in range(nt):
        for j1 in range(i1 + 1, nt):
            if wt[i1, j1]:
                lying_i1, lying_j1 = lying_t[i1], lying_t[j1]
                edge_types_t["{}_{}".format(i1, j1)] = edge_types["{}_{}".format(lying_i1, lying_j1)]
                edge_types_t["{}_{}".format(j1, i1)] = edge_types["{}_{}".format(lying_j1, lying_i1)]
    data_path = os.path.join("data", dataset, "{}_{}_{}_{}.p".format(n_overlap, ns, nt, int(100 * ratios[3])))
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f)
    edge_types_s_path = os.path.join("data", dataset,
                                     "{}_{}_{}_{}_edge_types_s.p".format(n_overlap, ns, nt, int(100 * ratios[3])))
    edge_types_t_path = os.path.join("data", dataset,
                                     "{}_{}_{}_{}_edge_types_t.p".format(n_overlap, ns, nt, int(100 * ratios[3])))
    with open(edge_types_s_path, "wb") as f:
        pickle.dump(edge_types_s, f)
    with open(edge_types_t_path, "wb") as f:
        pickle.dump(edge_types_t, f)
    return graph_st, data_path
