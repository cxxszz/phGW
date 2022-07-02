import numpy as np
from copy import deepcopy as dc
from data_io.real_data_hete import graph
from data_io.real_noise_hete import graph_pair_rn


def rm_isolated(g: graph):
    # ----------------- isolated nodes included ----------------
    lying_full = g.lying
    w_full = g.w
    node_types_full = g.node_types
    values_full = g.values

    # ----------------- remove isolated nodes ------------------
    deg_full = np.sum(w_full, axis=1)
    indices_sub = np.nonzero(deg_full > 0)[0]
    if len(indices_sub) == g.n:
        return g
    if isinstance(lying_full, list):
        lying = [lying_full[i] for i in indices_sub]
    elif isinstance(lying_full, np.ndarray):
        lying = lying_full[indices_sub]
    else:
        raise TypeError(type(lying_full))
    if isinstance(node_types_full, list):
        node_types = [node_types_full[i] for i in indices_sub]
    elif isinstance(node_types_full, np.ndarray):
        node_types = node_types_full[indices_sub]
    else:
        raise TypeError(type(node_types_full))
    w = w_full[indices_sub][:, indices_sub]
    values = values_full[indices_sub]
    g_sub = graph(w=w, lying=lying)
    g_sub.set_node_types(node_types=node_types)
    g_sub.set_values(values=values)
    return g_sub


def oag_switch(g_st: graph_pair_rn):
    anch_ratio = 0.01
    g_s = g_st.graph_s
    g_t = g_st.graph_t

    graph_s = rm_isolated(g_t)
    graph_t = dc(g_s)
    assert graph_s.n <= graph_t.n
    n_overlap = 0
    for a in graph_s.lying:
        for b in graph_t.lying:
            if a == b:
                n_overlap += 1
                break
    return graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)
