"""
IO for DBLP
"""

import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from data_io.real_noise_hete import graph, graph_pair_rn
from data_io.part import part_static, part_dynamic, subgraph, subgraph_type, part_dynamic_type
import csv


def main_api_type():
    # ====================== first completely load the data ====================
    N = 1989077
    d0 = 300
    n = 8000
    # n = 100
    # ------------------------------- edges ------------------------------------
    dataset = "dblp"
    csr_path = os.path.join("data", dataset, "csr.npz")
    if not os.path.exists(csr_path):
        weights = []
        row = []
        col = []
        with open("data/raw/DBLP/link.dat", "r") as f:
            for line_id, x in enumerate(f):
                print("line_id={}".format(line_id))
                y = x.split('\t')
                weights.append(float(y[3]))
                weights.append(float(y[3]))
                row.append(int(y[0]))
                row.append(int(y[1]))
                col.append(int(y[1]))
                col.append(int(y[0]))
        w_csr = csr_matrix((weights, (row, col)), shape=(N, N))
        save_npz(csr_path, w_csr)
        print("saved the original csr_matrix")
    else:
        w_csr = load_npz(csr_path)
        print("loaded the original csr_matrix")
    # -------------------- node_types and node attributes ---------------------------
    type_d = {}
    node_types_full = np.zeros(N, dtype=np.int)
    attributes_full = np.zeros([N, d0])
    with open("data/raw/DBLP/node.dat", "r") as f:
        for line_id, x in enumerate(f):
            y = x.split('\t')
            node_type = int(y[2])
            node_types_full[line_id] = node_type
            att_str = y[3]
            att_list = att_str.split(',')
            attributes_full[line_id] = np.array([float(att) for att in att_list])
            if node_type not in type_d.keys():
                type_d[node_type] = 1
            else:
                type_d[node_type] += 1
    for node_type in type_d.keys():
        print("The original graph, Type-{} has {} nodes".format(node_type, type_d[node_type]))
    # ==================== now extract a sub-graph with reasonable size =========
    indices = subgraph_type(w=w_csr, node_types=node_types_full, type_d=type_d, max_nodes=n)
    print(indices)
    node_types = node_types_full[indices]
    for node_type in type_d.keys():
        print("Type-{} has {} nodes".format(node_type, np.sum((node_types == node_type).astype(np.int))))

    # =================== generate graph_s and graph_t ==========================
    w = w_csr[indices][:, indices].toarray()
    # ratios = (1.0, 0.0, 0.0, 0.01)
    # ratios = (0.9, 0.04, 0.06, 0.01)
    ratios = (0.8, 0.09, 0.11, 0.01)
    # ratios = (0.7, 0.14, 0.16, 0.01)
    # ratios = (0.6, 0.19, 0.21, 0.01)
    # ratios = (0.5, 0.24, 0.26, 0.01)
    # ratios = (0.4, 0.3, 0.3, 0.01)
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
    values_s = attributes_full[lying_s]
    # values_t = attributes_full[lying_t]
    values_t = attributes_full[lying_t] + np.random.normal(0, 1, [len(node_types_t), d0])

    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    graph_s.set_node_types(node_types=node_types_s)
    graph_t.set_node_types(node_types=node_types_t)
    graph_s.set_values(values=values_s)
    graph_t.set_values(values=values_t)
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
