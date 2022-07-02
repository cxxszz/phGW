"""
IO for the counter-example
"""
import numpy as np
import os
import pickle
from data_io.real_noise_hete import graph, graph_pair_rn


def main_api_type():
    ws = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])
    wt = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])
    lying_s = np.array([1, 2, 3])
    lying_t = np.array([4, 3, 2])
    node_types_s = np.array([0, 0, 1])
    node_types_t = np.array([1, 1, 0])
    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    graph_s.set_node_types(node_types=node_types_s)
    graph_t.set_node_types(node_types=node_types_t)
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=2, anch_ratio=1.0)
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))

    data_path = os.path.join("data", "counterexample.p")
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f)
    return graph_st, data_path
