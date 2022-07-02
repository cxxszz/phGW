import numpy as np
from data_io.real_data_homo import graph, generate_store

def rand_prob_node(A):
    """

    :param A: adjacency matrix
    :return:
    """
    nodes_proba = np.sum(A, axis=1)
    nodes_proba = nodes_proba / np.sum(nodes_proba)
    return np.random.choice(range(A.shape[0]), p=nodes_proba)


def add_one_edge(A, cur_node):
    random_proba_node = rand_prob_node(A=A)
    if A[random_proba_node][cur_node] == 0:
        A[random_proba_node][cur_node] = 1
        A[cur_node][random_proba_node] = 1
        return A
    else:
        A = add_one_edge(A=A, cur_node=cur_node)
        return A


def add_edge_ba(n_final, n_init=100, m=5):
    assert m < n_init
    A = np.zeros((n_final, n_final))
    A[0:n_init, 0:n_init] = 1  # initial nodes constitute a complete graph
    for cur_node in range(n_init, n_final):
        for _ in range(m):
            A = add_one_edge(A=A, cur_node=cur_node)
    return A


def main_api():
    n = 200
    w = add_edge_ba(n_final=n)
    full_graph = graph(w=w)
    dataset = "ba"
    # ## =================datasets used in the paper below======================================
    graph_st, data_path = generate_store(full_graph=full_graph, n_overlap=200, ns=200, nt=200, anch_ratio=0.1,
                                         dist_value=0.15, store=True, dataset=dataset)

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