import numpy as np
import os
import pickle
from sortedcontainers import SortedList
from data_io.real_data_homo import graph
from data_io.real_noise_homo import graph_pair_rn


def main_api():
    # ================================ source graph =================================
    # ns = 137517
    # path_s = "data/raw/askubuntu-a2q.txt"
    ns = 75555
    path_s = "data/raw/askubuntu-c2a.txt"
    with open(path_s, "r") as f:
        data = f.readlines()
    sl_s = SortedList()
    for record in data:
        s = record.split(" ")
        i = int(s[0])
        j = int(s[1])
        if i not in sl_s:
            sl_s.add(i)
        if j not in sl_s:
            sl_s.add(j)
    print(sl_s)
    lying_s = np.zeros(ns, dtype=np.int)
    for i, item in enumerate(sl_s):
        lying_s[i] = item
    print(lying_s)
    ws = np.zeros((ns, ns), dtype=np.int8)
    for record in data:
        s = record.split(" ")
        i = sl_s.index(int(s[0]))
        j = sl_s.index(int(s[1]))
        ws[i, j] = 1
        ws[j, i] = 1
    graph_s = graph(w=ws, lying=lying_s)
    # =============================== target graph ==================================
    # path_t = "data/raw/askubuntu.txt"
    # nt = 159316
    path_t = "data/raw/askubuntu-c2q.txt"
    nt = 79155
    with open(path_t, "r") as f:
        data = f.readlines()
    sl_t = SortedList()
    for record in data:
        s = record.split(" ")
        i = int(s[0])
        j = int(s[1])
        if i not in sl_t:
            sl_t.add(i)
        if j not in sl_t:
            sl_t.add(j)
    print(sl_t)
    lying_t = np.zeros(nt, dtype=np.int)
    for i, item in enumerate(sl_t):
        lying_t[i] = item
    print(lying_t)
    wt = np.zeros((nt, nt), dtype=np.int8)
    for record in data:
        s = record.split(" ")
        i = sl_t.index(int(s[0]))
        j = sl_t.index(int(s[1]))
        wt[i, j] = 1
        wt[j, i] = 1
    graph_t = graph(w=wt, lying=lying_t)
    n_overlap = 0
    for item in sl_s:
        if item in sl_t:
            n_overlap += 1
    graph_st = graph_pair_rn(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.2)
    data_path = os.path.join("data/ubuntu/1.p")
    print(graph_st.result_eval(graph_st.gt))
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    with open(data_path, "wb") as f:
        pickle.dump(graph_st, f, protocol=4)
    return graph_st
