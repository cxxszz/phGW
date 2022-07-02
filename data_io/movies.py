"""
IO for Movies
"""

import numpy as np
import os
import pickle
from sortedcontainers import SortedList
from scipy.sparse import csr_matrix, save_npz, load_npz
from data_io.real_noise_hete import graph, graph_pair_rn
from data_io.part import part_static, part_dynamic, subgraph, subgraph_type, part_dynamic_type
import csv


def main_api_type():
    # ====================== first completely load the data ====================
    N = 21435
    start_d = {"a": 0, "m": 10789, "d": 10789 + 7332, "c": 10789 + 7332 + 1741}
    E = 89038
    n = 1000
    convert_d = {"a": 0, "m": 1, "d": 2, "c": 3}
    # --------------------------- determine the node indices -------------------------
    sl_d = {"a": SortedList(), "m": SortedList(), "d": SortedList(), "c": SortedList()}
    with open("data/raw/Movies.txt", "r") as f:
        for line_id, x in enumerate(f):
            y = x.split('\t')
            node0 = y[0]
            if line_id == E - 1:
                node1 = y[1]
            else:
                node1 = y[1][:-1]
            i, j = int(node0[1:]), int(node1[1:])
            if i not in sl_d[node0[0]]:
                sl_d[node0[0]].add(i)
            if j not in sl_d[node1[0]]:
                sl_d[node1[0]].add(j)
    # ------------------------------ edges and node_types ----------------------------
    dataset = "movies"
    node_types_full = np.zeros(N, dtype=np.int)
    type_d = {}
    weights = []
    row = []
    col = []
    with open("data/raw/Movies.txt", "r") as f:
        for line_id, x in enumerate(f):
            y = x.split('\t')
            weights.append(1)
            weights.append(1)
            # print(line_id, type(y), y)
            node0 = y[0]
            if line_id == E - 1:
                node1 = y[1]
            else:
                node1 = y[1][:-1]
            # print(node0, node1)
            i = sl_d[node0[0]].index(int(node0[1:])) + start_d[node0[0]]
            j = sl_d[node1[0]].index(int(node1[1:])) + start_d[node1[0]]
            row.append(i)
            row.append(j)
            col.append(j)
            col.append(i)
            type_i, type_j = convert_d[node0[0]], convert_d[node1[0]]
            for node_type in [type_i, type_j]:
                node_types_full[i] = type_i
                node_types_full[j] = type_j
                if node_type not in type_d.keys():
                    type_d[node_type] = 1
                else:
                    type_d[node_type] += 1
    w_csr = csr_matrix((weights, (row, col)), shape=(N, N))
    # ---------------------------- node_types ----------------------------------
    for node_type in type_d.keys():
        print("The original graph, Type-{} has {} nodes".format(node_type, type_d[node_type]))
    # ==================== now extract a sub-graph with reasonable size =========
    indices = subgraph_type(w=w_csr, node_types=node_types_full, type_d=type_d, max_nodes=n)
    np.random.shuffle(indices)
    print(indices)
    node_types = node_types_full[indices]
    for node_type in type_d.keys():
        print("Type-{} has {} nodes".format(node_type, np.sum((node_types == node_type).astype(np.int))))

    # =================== generate graph_s and graph_t ==========================
    w = w_csr[indices][:, indices].toarray()
    # ratios = (1.0, 0.0, 0.0, 0.1)
    # ratios = (0.9, 0.04, 0.06, 0.1)
    # ratios = (0.8, 0.09, 0.11, 0.1)
    # ratios = (0.7, 0.14, 0.16, 0.1)
    ratios = (0.6, 0.18, 0.22, 0.1)
    # ratios = (0.5, 0.24, 0.26, 0.1)
    # ratios = (0.4, 0.3, 0.3, 0.1)
    if ratios[0] < 1:
        ws, lying_s, wt, lying_t, n_overlap = part_dynamic_type(w=w, node_types=node_types, type_d=type_d,
                                                                over_ratio=ratios[0], s_ratio=ratios[1],
                                                                t_ratio=ratios[2])
    else:
        n_overlap = n
        lying_s = np.arange(n)
        ws = w
        lying_t = np.arange(n)
        np.random.shuffle(lying_t)
        wt = w[lying_t][:, lying_t]
    node_types_s = node_types[lying_s]
    node_types_t = node_types[lying_t]

    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    graph_s.set_node_types(node_types=node_types_s)
    graph_t.set_node_types(node_types=node_types_t)
    ns, nt = graph_s.n, graph_t.n
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=ratios[3])
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))

    data_path = os.path.join("data", dataset, "{}_{}_{}_{}.p".format(n_overlap, ns, nt, int(100 * ratios[3])))
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f)
    return graph_st, data_path
