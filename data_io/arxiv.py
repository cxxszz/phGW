"""
IO for Arxiv data http://snap.stanford.edu/data/ca-GrQc.html
"""

import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from data_io.real_noise_homo import graph, graph_pair_rn
from data_io.part import part_static, part_dynamic


def main_api():
    n = 18772
    w = np.zeros((n, n))
    id_d = {}  # id in the graph
    with open("data/raw/ca-AstroPh.txt", "r") as f:
        for line_id, x in enumerate(f):
            if line_id > 4:
                y = x.split('\t')
                key0 = y[0]
                key1 = y[1][:-1]
                if key0 not in id_d:
                    i = len(id_d)
                    id_d[key0] = i
                else:
                    i = id_d[key0]

                if key1 not in id_d:
                    j = len(id_d)
                    id_d[key1] = j
                else:
                    j = id_d[key1]

                w[i][j] = 1
                w[j][i] = 1

    # ratios = (0.9, 0.04, 0.06, 0.1)
    # ratios = (0.8, 0.09, 0.11, 0.1)
    # ratios = (0.7, 0.14, 0.16, 0.1)
    # ratios = (0.6, 0.19, 0.21, 0.1)
    ratios = (0.5, 0.24, 0.26, 0.1)
    # ratios = (0.4, 0.3, 0.3, 0.1)

    # ws, lying_s, wt, lying_t, n_overlap = part_static(w=w, over_ratio=ratios[0], s_ratio=ratios[1],
    #                                                   t_ratio=ratios[2])
    ws, lying_s, wt, lying_t, n_overlap = part_dynamic(w=w, over_ratio=ratios[0], s_ratio=ratios[1],
                                                       t_ratio=ratios[2])

    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    ns, nt = graph_s.n, graph_t.n
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=ratios[3])
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    dataset = "arxiv"
    data_path = os.path.join("data", dataset, "{}_{}_{}_{}.p".format(n_overlap, ns, nt, int(100 * ratios[3])))
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f)
    return graph_st, data_path
