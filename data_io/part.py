import numpy as np
from sortedcontainers import SortedList
from lib.util import get_nei
from scipy.sparse import csr_matrix


def subgraph(w: csr_matrix, max_nodes=4000):
    """
    State-of-the-art methods mainly take O(n^3) memory. Thus, n cannot be too large.
    :param w: has shape (N, N) where N may be a large number
    :param max_nodes:
    :return: indices as an nparray
    """
    N = w.shape[0]
    sl = SortedList()
    cand = SortedList()
    # ------------ The expansion starts from a node that has many neighbors ------------
    deg = np.asarray(w.sum(axis=0)).flatten()
    # print("deg={}".format(deg), type(deg))
    norm_deg = deg / np.sum(deg)  # normalized degree
    index = np.random.choice(N, p=norm_deg)
    # print("norm_deg={}, max(norm_deg)={}".format(norm_deg, np.max(norm_deg)))
    # print(deg[index], norm_deg[index])

    # index = np.random.choice(N)
    sl.add(index)
    all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
    # print(index)
    # print("all_nei={}".format(all_nei))
    # print(w[index].nonzero()[1])
    for i in all_nei:
        if i not in cand and i not in sl:
            cand.add(i)
    while len(sl) < max_nodes and len(cand) > 0:
        # print("len(sl)={}".format(len(sl)))
        index = np.random.choice(list(cand))
        sl.add(index)
        cand.discard(index)
        all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
        for i in all_nei:
            if i not in cand and i not in sl:
                cand.add(i)
    # assert len(sl) > 1000
    return np.array(list(sl))


def part_static(w: np.ndarray, over_ratio=0.9, s_ratio=0.05, t_ratio=0.05):
    """
    Expansion-based sub-graph generation.
    :param w:
    :param over_ratio: |Vs \cap Vt| / |Vs \cup Vt|
    :param s_ratio: |Vs - Vs \cap Vt| / |Vs \cup Vt|
    :param t_ratio: |Vt - Vs \cap Vt| / |Vs \cup Vt|
    :return:
    """

    def cand_lists(w: np.ndarray, sl_o, sl_s, sl_t):
        # ----------- Candidate lists for sl_o, sl_s, and sl_t respectively ------------
        cand_o, cand_s, cand_t = SortedList(), SortedList(), SortedList()
        # ------------------------ Calculating cand_o ----------------------------------
        for j in list(sl_o):
            all_nei = get_nei(w=w, i=j)  # all nodes neighboring to node j
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    cand_o.add(i)
        # ------------------------ Calculating cand_s ----------------------------------
        cand_s.update(cand_o)
        for j in list(sl_s):
            all_nei = get_nei(w=w, i=j)  # all nodes neighboring to node j
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    cand_s.add(i)
        # ------------------------ Calculating cand_t ----------------------------------
        cand_t.update(cand_o)
        for j in list(sl_t):
            all_nei = get_nei(w=w, i=j)  # all nodes neighboring to node j
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    cand_t.add(i)
        return cand_o, cand_s, cand_t

    n = w.shape[0]
    sl_o = SortedList()  # Vs \cap Vt
    sl_s = SortedList()  # Vs - Vs \cap Vt
    sl_t = SortedList()  # Vt - Vs \cap Vt

    # ------------ The expansion starts from a node that has a large degree ------------
    deg = np.sum(w, axis=1)
    norm_deg = deg / np.sum(deg)  # normalized degree
    index = np.random.choice(n, p=norm_deg)
    sl_o.add(index)
    cand_o, cand_s, cand_t = cand_lists(w=w, sl_o=sl_o, sl_s=sl_s, sl_t=sl_t)
    # --------------------------- expansion iterations ---------------------------------
    while len(cand_o) > 0 and len(cand_s) > 0 and len(cand_t) > 0:
        # determine the part we should add nodes into
        part_to_add = np.random.choice(["o", "s", "t"], p=np.array([over_ratio, s_ratio, t_ratio]))
        if part_to_add == "s":
            index = np.random.choice(list(cand_s))
            sl_s.add(index)
        elif part_to_add == "t":
            index = np.random.choice(list(cand_t))
            sl_t.add(index)
        else:
            index = np.random.choice(list(cand_o))
            sl_o.add(index)
        cand_o, cand_s, cand_t = cand_lists(w=w, sl_o=sl_o, sl_s=sl_s, sl_t=sl_t)
        print("sl_o {}, sl_s {}, sl_t {}".format(len(sl_o), len(sl_s), len(sl_t)))
    n_overlap = len(sl_o)
    indices_s, indices_t = list(sl_o) + list(sl_s), list(sl_o) + list(sl_t)
    lying_s, lying_t = np.array(indices_s), np.array(indices_t)
    np.random.shuffle(lying_t)
    ws = w[lying_s, :][:, lying_s]
    wt = w[lying_t, :][:, lying_t]
    return ws, lying_s, wt, lying_t, n_overlap


def part_dynamic(w: np.ndarray, over_ratio=0.9, s_ratio=0.05, t_ratio=0.05):
    """
    Expansion-based sub-graph generation.
    :param w:
    :param over_ratio: |Vs \cap Vt| / |Vs \cup Vt|
    :param s_ratio: |Vs - Vs \cap Vt| / |Vs \cup Vt|
    :param t_ratio: |Vt - Vs \cap Vt| / |Vs \cup Vt|
    :return:
    """

    n = w.shape[0]
    sl_o = SortedList()  # Vs \cap Vt
    sl_s = SortedList()  # Vs - Vs \cap Vt
    sl_t = SortedList()  # Vt - Vs \cap Vt
    cand_o, cand_s, cand_t = SortedList(), SortedList(), SortedList()
    # ------------ The expansion starts from a node that has a large degree ------------
    deg = np.sum(w, axis=1)
    norm_deg = deg / np.sum(deg)  # normalized degree
    index = np.random.choice(n, p=norm_deg)
    sl_o.add(index)
    all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
    for i in all_nei:
        if i not in sl_o and i not in sl_s and i not in sl_t:
            cand_o.add(i)
            cand_s.add(i)
            cand_t.add(i)
    # --------------------------- expansion iterations ---------------------------------
    while len(cand_o) > 0 and len(cand_s) > 0 and len(cand_t) > 0:
        # determine the part we should add nodes into
        part_to_add = np.random.choice(["o", "s", "t"], p=np.array([over_ratio, s_ratio, t_ratio]))
        if part_to_add == "s":
            index = np.random.choice(list(cand_s))
            sl_s.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_s:
                        cand_s.add(i)
        elif part_to_add == "t":
            index = np.random.choice(list(cand_t))
            sl_t.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_t:
                        cand_t.add(i)
        else:
            index = np.random.choice(list(cand_o))
            sl_o.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_o:
                        cand_o.add(i)
                    if i not in cand_s:
                        cand_s.add(i)
                    if i not in cand_t:
                        cand_t.add(i)
        print("sl_o {}, sl_s {}, sl_t {}".format(len(sl_o), len(sl_s), len(sl_t)))
    n_overlap = len(sl_o)
    indices_s, indices_t = list(sl_o) + list(sl_s), list(sl_o) + list(sl_t)
    lying_s, lying_t = np.array(indices_s), np.array(indices_t)
    np.random.shuffle(lying_t)
    ws = w[lying_s, :][:, lying_s]
    wt = w[lying_t, :][:, lying_t]
    return ws, lying_s, wt, lying_t, n_overlap


def calc_weight(cand: SortedList, node_types: np.ndarray, wei_d: dict):
    weights = []
    for i in cand:
        k = node_types[i]
        weights.append(wei_d[k])
    return np.array(weights)


def type_sample(cand: SortedList, node_types: np.ndarray, wei_d: dict):
    weights = calc_weight(cand=cand, node_types=node_types, wei_d=wei_d)
    p = weights / np.sum(weights)
    return np.random.choice(list(cand), p=p)


def subgraph_type(w: csr_matrix, node_types: np.ndarray, type_d: dict, max_nodes=4000):
    """
    State-of-the-art methods mainly take O(n^3) memory. Thus, n cannot be too large.
    :param w: has shape (N, N) where N may be a large number
    :param max_nodes:
    :return: indices as an nparray
    """
    wei_d = {}
    for k in type_d.keys():
        wei_d[k] = 1 / type_d[k]
    N = w.shape[0]
    sl = SortedList()
    cand = SortedList()
    # ------------ The expansion starts from a node that has many neighbors ------------
    deg = np.asarray(w.sum(axis=0)).flatten()
    norm_deg = deg / np.sum(deg)  # normalized degree
    index = np.random.choice(N, p=norm_deg)
    sl.add(index)
    all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
    for i in all_nei:
        if i not in cand and i not in sl:
            cand.add(i)
    while len(sl) < max_nodes and len(cand) > 0:
        print("len(sl)={}".format(len(sl)))
        index = type_sample(cand=cand, node_types=node_types, wei_d=wei_d)
        sl.add(index)
        cand.discard(index)
        all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
        for i in all_nei:
            if i not in cand and i not in sl:
                cand.add(i)
    assert len(sl) > 20
    return np.array(list(sl))


def part_dynamic_type(w: np.ndarray, node_types: np.ndarray, type_d: dict, over_ratio=0.9, s_ratio=0.05, t_ratio=0.05):
    """
    Expansion-based sub-graph generation.
    :param w:
    :param over_ratio: |Vs \cap Vt| / |Vs \cup Vt|
    :param s_ratio: |Vs - Vs \cap Vt| / |Vs \cup Vt|
    :param t_ratio: |Vt - Vs \cap Vt| / |Vs \cup Vt|
    :return:
    """
    wei_d = {}
    for k in type_d.keys():
        wei_d[k] = 1 / type_d[k]
    n = w.shape[0]
    sl_o = SortedList()  # Vs \cap Vt
    sl_s = SortedList()  # Vs - Vs \cap Vt
    sl_t = SortedList()  # Vt - Vs \cap Vt
    cand_o, cand_s, cand_t = SortedList(), SortedList(), SortedList()
    # ------------ The expansion starts from a node that has a large degree ------------
    deg = np.sum(w, axis=1)
    norm_deg = deg / np.sum(deg)  # normalized degree
    index = np.random.choice(n, p=norm_deg)
    sl_o.add(index)
    all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
    for i in all_nei:
        if i not in sl_o and i not in sl_s and i not in sl_t:
            cand_o.add(i)
            cand_s.add(i)
            cand_t.add(i)
    # --------------------------- expansion iterations ---------------------------------
    while len(cand_o) > 0 and len(cand_s) > 0 and len(cand_t) > 0:
        # determine the part we should add nodes into
        part_to_add = np.random.choice(["o", "s", "t"], p=np.array([over_ratio, s_ratio, t_ratio]))
        if part_to_add == "s":
            # index = np.random.choice(list(cand_s))
            index = type_sample(cand=cand_s, node_types=node_types, wei_d=wei_d)
            sl_s.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_s:
                        cand_s.add(i)
        elif part_to_add == "t":
            # index = np.random.choice(list(cand_t))
            index = type_sample(cand=cand_t, node_types=node_types, wei_d=wei_d)
            sl_t.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_t:
                        cand_t.add(i)
        else:
            # index = np.random.choice(list(cand_o))
            index = type_sample(cand=cand_o, node_types=node_types, wei_d=wei_d)
            sl_o.add(index)
            cand_o.discard(index)
            cand_s.discard(index)
            cand_t.discard(index)
            all_nei = get_nei(w=w, i=index)  # all nodes neighboring to node index
            for i in all_nei:
                if i not in sl_o and i not in sl_s and i not in sl_t:
                    if i not in cand_o:
                        cand_o.add(i)
                    if i not in cand_s:
                        cand_s.add(i)
                    if i not in cand_t:
                        cand_t.add(i)
        print("sl_o {}, sl_s {}, sl_t {}, cand_o {}, cand_s {}, cand_t {}".format(len(sl_o), len(sl_s), len(sl_t),
                                                                                  len(cand_o), len(cand_s),
                                                                                  len(cand_t)))
    n_overlap = len(sl_o)
    indices_s, indices_t = list(sl_o) + list(sl_s), list(sl_o) + list(sl_t)
    lying_s, lying_t = np.array(indices_s), np.array(indices_t)
    np.random.shuffle(lying_t)
    ws = w[lying_s, :][:, lying_s]
    wt = w[lying_t, :][:, lying_t]
    return ws, lying_s, wt, lying_t, n_overlap
