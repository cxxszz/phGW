import numpy as np
import pickle
import os
import torch
from torch_geometric.datasets import DBP15K
from data_io.real_data_homo import graph
from data_io.real_noise_homo import generate_store_dbp


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


def main_api():
    category = "zh_en"
    path = os.path.join('data', 'DBP15K')
    data = DBP15K(path, category, transform=SumEmbedding())[0]
    ns = data.x1.size(0)
    nt = data.x2.size(0)
    assert ns <= nt
    print("ns={}, nt={}".format(ns, nt))

    print(torch.sum(torch.abs(data.train_y[0] - data.train_y[1])))
    print(torch.sum(torch.abs(data.test_y[0] - data.test_y[1])))
    print(len(data.train_y[0]))
    print(len(data.test_y[0]))
    n_overlap = len(data.train_y[0]) + len(data.test_y[0])
    # print(data.edge_index1)
    # print(data.edge_index2)
    # print("Es={}, Et={}".format(data.edge_index1.size(1), data.edge_index2.size(1)))

    lying_s = np.zeros(ns) - 1
    lying_t = np.zeros(nt) - 2
    for nn in range(data.train_y.size(1)):
        i = data.train_y[0, nn]
        lying_s[i] = i
        lying_t[i] = i

    for nn in range(data.test_y.size(1)):
        i = data.test_y[0, nn]
        lying_s[i] = i
        lying_t[i] = i

    ws = np.zeros((ns, ns))
    for ei in range(data.edge_index1.size(1)):  # the index of the edge
        i = data.edge_index1[0, ei]
        j = data.edge_index1[1, ei]
        ws[i, j] = 1
        ws[j, i] = 1

    wt = np.zeros((nt, nt))
    for ei in range(data.edge_index2.size(1)):
        i = data.edge_index2[0, ei]
        j = data.edge_index2[1, ei]
        wt[i, j] = 1
        wt[j, i] = 1

    graph_s = graph(w=ws, lying=lying_s)
    graph_t = graph(w=wt, lying=lying_t)
    graph_st, data_path = generate_store_dbp(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.2,
                                             category=category)
    print("data_path={}".format(data_path))
    print(graph_st.result_eval(graph_st.gt))
    print(graph_st.anch_cp)
    mu_s = np.sum(graph_st.graph_s.w, axis=1)
    mu_t = np.sum(graph_st.graph_t.w, axis=1)
    print(np.sum(mu_s == 0))
    print(np.sum(mu_t == 0))
    return graph_st
