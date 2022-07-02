"""
Stochastic Block Model
"""

import numpy as np
import networkx as nx
from data_io.real_data_homo import graph, generate_store


def main_api(n=400, n_com=2, intra_prob=0.02, inter_prob=0.94):
    """

    :param n:
    :param n_com: n_community
    :param intra_prob:
    :param inter_prob:
    :return:
    """
    sizes = int(n / n_com) * np.ones(n_com, dtype=np.int)
    probs = 0.02 * np.ones((n_com, n_com)) + 0.94 * np.eye(n_com)
    g = nx.stochastic_block_model(sizes, probs, sparse=False)
    w = np.zeros((n, n))
    for e in g.edges:
        i = e[0]
        j = e[1]
        w[i, j] = 1
        w[j, i] = 1
    full_graph = graph(w=w)
    dataset = "sbm"
    # ## =================datasets used in the paper below======================================
    # graph_st, data_path = generate_store(full_graph=full_graph, n_overlap=n, ns=n, nt=n, anch_ratio=0.1,
    #                                      dist_value=0, store=True, dataset=dataset)
    graph_st, data_path = generate_store(full_graph=full_graph, n_overlap=n, ns=n, nt=n, anch_ratio=0.1,
                                         dist_value=0, store=False, dataset=dataset)

    print(graph_st.result_eval(graph_st.gt))
    # print(graph_st.anch_cp)
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))

    ## ==================calculating graphlets features=======================================
    # name = data_path.split('/')[-1][:-2]
    #
    # sim = sim_calc(graph_st=graph_st, dataset=dataset, name=name)
    return graph_st
