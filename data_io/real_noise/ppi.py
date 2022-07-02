import numpy as np
import pickle
from data_io.real_data_homo import graph
from data_io.real_noise_homo import generate_store


def main_api():
    with open("data/raw/PPI_syn_database.pkl", "rb") as f:
        database = pickle.load(f)  # a dict

    ws = 0.5 * (database["costs"][0].toarray() + database["costs"][0].toarray().T)
    lying_s = database["idx2nodes"][0]
    graph_s = graph(w=ws, lying=lying_s)
    for noise_level in range(6):
        print("noise_level={}".format(noise_level))
        wt = 0.5 * (database["costs"][noise_level].toarray() + database["costs"][noise_level].toarray().T)
        lying_t = database["idx2nodes"][noise_level]
        graph_t = graph(w=wt, lying=lying_t)
        n_overlap = np.min([graph_s.n, graph_t.n])
        graph_st, data_path = generate_store(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=0.2,
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
