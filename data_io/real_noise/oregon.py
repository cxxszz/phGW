import numpy as np
import pickle
from sortedcontainers import SortedList
from data_io.real_data_homo import graph
from data_io.real_noise_homo import generate_store_oregon


def main_api():
    # -------------------------------------------- source graph --------------------------------------------
    path = "data/raw/oregon1_010331.txt"
    ns = 10670
    with open(path, "r") as f:
        data = f.readlines()
    sl_s = SortedList()
    for record in data[4:]:
        s = record.split("\t")
        i = int(s[0])
        j = int(s[1][:-1])
        if i not in sl_s:
            sl_s.add(i)
        if j not in sl_s:
            sl_s.add(j)
    print(sl_s)
    lying_s = np.zeros(ns, dtype=np.int)
    for i, item in enumerate(sl_s):
        lying_s[i] = item
    print(lying_s)
    ws = np.zeros((ns, ns))
    for record in data[4:]:
        s = record.split("\t")
        i = sl_s.index(int(s[0]))
        j = sl_s.index(int(s[1][:-1]))
        ws[i, j] = 1
        ws[j, i] = 1
    graph_s = graph(w=ws, lying=lying_s)

    # -------------------------------------------- target graph --------------------------------------------
    # path = "data/raw/oregon1_010331.txt"
    # nt = 10670
    # noise_level = 0
    path = "data/raw/oregon1_010407.txt"
    nt = 10729
    noise_level = 1
    # path = "data/raw/oregon1_010414.txt"
    # nt = 10790
    # noise_level = 2
    # path = "data/raw/oregon1_010421.txt"
    # nt = 10859
    # noise_level = 3
    # path = "data/raw/oregon1_010428.txt"
    # nt = 10886
    # noise_level = 4
    # path = "data/raw/oregon1_010505.txt"
    # nt = 10943
    # noise_level = 5
    with open(path, "r") as f:
        data = f.readlines()
    sl_t = SortedList()
    for record in data[4:]:
        s = record.split("\t")
        i = int(s[0])
        j = int(s[1][:-1])
        if i not in sl_t:
            sl_t.add(i)
        if j not in sl_t:
            sl_t.add(j)
    print(sl_t)
    lying_t = np.zeros(nt, dtype=np.int)
    for i, item in enumerate(sl_t):
        lying_t[i] = item
    print(lying_t)
    wt = np.zeros((nt, nt))
    for record in data[4:]:
        s = record.split("\t")
        i = sl_t.index(int(s[0]))
        j = sl_t.index(int(s[1][:-1]))
        wt[i, j] = 1
        wt[j, i] = 1
    graph_t = graph(w=wt, lying=lying_t)

    n_overlap = 0
    for item in sl_s:
        if item in sl_t:
            n_overlap += 1

    graph_st, data_path = generate_store_oregon(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.2,
                                                noise_level=noise_level)
    print("data_path={}".format(data_path))
    print(graph_st.result_eval(graph_st.gt))
    print(graph_st.anch_cp)
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    # ------------------------ calculating graphlets features --------------------------------------
    # dataset = "ppi"
    # name = data_path.split('/')[-1][:-2]
    #
    # sim = sim_calc(graph_st=graph_st, dataset=dataset, name=name)
    return graph_st

# def main_api():
#     # -------------------------------------------- source graph --------------------------------------------
#     path = "data/raw/oregon1_010331.txt"
#     ns = 10670
#     lying_s = np.arange(ns)
#     ws = np.zeros((ns, ns))
#     with open(path, "r") as f:
#         data = f.readlines()
#     for record in data[4:]:
#         s = record.split("\t")
#         i = int(s[0]) - 1  # starting from 1
#         j = int(s[1][:-1]) - 1
#         ws[i, j] = 1
#         ws[j, i] = 1
#     for i in range(ns):
#         if np.sum(ws[i]) == 0:
#             j = np.random.randint(0, ns, 1)[0]  # without [0], the result will be an array of shape (1,)
#             ws[i, j] = 1
#             ws[j, i] = 1
#     graph_s = graph(w=ws, lying=lying_s)
#
#     # -------------------------------------------- target graph --------------------------------------------
#     path = "data/raw/oregon1_010407.txt"
#     nt = 10729
#     noise_level = 1
#     shi_order = np.arange(nt)  # shifted order
#     np.random.shuffle(shi_order)
#     wt = np.zeros((nt, nt))
#     with open(path, "r") as f:
#         data = f.readlines()
#     for record in data[4:]:
#         s = record.split("\t")
#         i = shi_order[int(s[0]) - 1]
#         j = shi_order[int(s[1][:-1]) - 1]
#         wt[i, j] = 1
#         wt[j, i] = 1
#     lying_t = np.zeros(nt)
#     for index in range(len(lying_t)):
#         lying_t[shi_order[index]] = index
#     for i in range(nt):
#         if np.sum(wt[i]) == 0:
#             j = np.random.randint(0, nt, 1)[0]
#             wt[i, j] = 1
#             wt[j, i] = 1
#     graph_t = graph(w=wt, lying=lying_t)
#
#     n_overlap = np.min([graph_s.n, graph_t.n])
#     graph_st, data_path = generate_store_oregon(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.2,
#                                                 noise_level=noise_level)
#     print("data_path={}".format(data_path))
#     print(graph_st.result_eval(graph_st.gt))
#     print(graph_st.anch_cp)
#     mu_s = np.sum(graph_st.graph_s.w, axis=1)
#     mu_t = np.sum(graph_st.graph_t.w, axis=1)
#     print(np.sum(mu_s == 0))
#     print(np.sum(mu_t == 0))
#     # ------------------------ calculating graphlets features --------------------------------------
#     # dataset = "ppi"
#     # name = data_path.split('/')[-1][:-2]
#     #
#     # sim = sim_calc(graph_st=graph_st, dataset=dataset, name=name)
#     return graph_st
