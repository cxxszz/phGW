"""
Real-world correlated heterogeneous graphs, i.e., real noise
"""
import numpy as np
from data_io.real_data_hete import graph


class graph_pair_rn(object):
    def __init__(self, graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2):
        self.my_init(graph_s=graph_s, graph_t=graph_t, n_overlap=n_overlap, anch_ratio=anch_ratio)

    def my_init(self, graph_s: graph, graph_t: graph, n_overlap: int, anch_ratio=0.2):
        self.graph_s = graph_s
        self.graph_t = graph_t
        self.ns = self.graph_s.n
        self.nt = self.graph_t.n
        self.n_overlap = n_overlap
        # calculating has_cp and gt that will be used to evaluate the estimated node correspondence
        self.has_cp = np.zeros(self.ns, dtype=bool)  # False initially
        self.gt = np.zeros(self.ns, dtype=np.int) + self.nt
        for i in range(self.ns):
            for j in range(self.nt):
                if self.graph_s.lying[i] == self.graph_t.lying[j]:
                    #  if node i of the source graph and node j of the target graph is actually the same node
                    self.has_cp[i] = True
                    self.gt[i] = j
                    break
        assert np.sum(self.gt < self.nt) == self.n_overlap

        # --------------------------------- dermine anchor nodes -----------------------------
        assert anch_ratio > 0  # we know the counterparts of anch_ratio of nodes in graph_s
        self.sample_anch(anch_ratio=anch_ratio)
        return self

    def sample_anch(self, anch_ratio):
        anch_s = np.random.choice(range(self.ns), int(self.ns * anch_ratio + 0.5), replace=False)
        anch_t = np.zeros(anch_s.shape, dtype=np.int)
        anch_cp = np.zeros(anch_s.shape, dtype=np.int)
        for i, index in enumerate(anch_s):
            if self.has_cp[index]:
                anch_t[i] = self.gt[index]
                anch_cp[i] = 1
            else:
                anch_t[i] = self.nt
                anch_cp[i] = 0
        self.anch_s, self.anch_t = anch_s, anch_t
        self.anch_cp = anch_cp
        return self

    def result_eval(self, corre, print_resu=False):
        """

        :param corre: The estimated target node for each node in the source graph
        :return:
        """
        cm, wm, tm, m_dummy = 0, 0, 0, 0  # the number of correct matching, wrong matching, total matching,
        # and matching to the dummy node, respectively
        mis_type = 0
        for i in range(self.ns):
            if self.has_cp[i]:
                if corre[i] == self.gt[i]:
                    cm += 1
                    tm += 1
                elif corre[i] == self.nt:
                    m_dummy += 1
                else:
                    wm += 1
                    tm += 1
                    if self.graph_s.node_types[i] != self.graph_t.node_types[corre[i]]:
                        mis_type += 1
            else:
                if corre[i] == self.nt:
                    m_dummy += 1
                else:
                    wm += 1
                    tm += 1
                    if self.graph_s.node_types[i] != self.graph_t.node_types[corre[i]]:
                        mis_type += 1
        if tm > 0:
            return cm / self.n_overlap, cm / tm, cm, wm, m_dummy, tm, self.n_overlap, self.ns, mis_type / self.ns
        else:
            return cm / self.n_overlap, 0, cm, wm, m_dummy, tm, self.n_overlap, self.ns, mis_type / self.ns


def report(corre: list, indices: list, graph_st: graph_pair_rn):
    cm, wm, tm, m_dummy = 0, 0, 0, 0  # the number of correct matching, wrong matching, total matching,
    # and matching to the dummy node, respectively
    mis_type = 0
    for corre_index, i in enumerate(indices):
        # print("corre_index={}, i={}, corre[corre_index]={}".format(corre_index, i, corre[corre_index]))
        # print("corre_index={}, i={}, len(corre)={}, len(indices)={}".format(corre_index, i, len(corre), len(indices)))
        if graph_st.has_cp[i]:
            if corre[corre_index] == graph_st.gt[i]:
                cm += 1
                tm += 1
            elif corre[corre_index] == graph_st.nt:
                m_dummy += 1
            else:
                wm += 1
                tm += 1
                if graph_st.graph_s.node_types[i] != graph_st.graph_t.node_types[corre[corre_index]]:
                    mis_type += 1
        else:
            if corre[corre_index] == graph_st.nt:
                m_dummy += 1
            else:
                wm += 1
                tm += 1
                if graph_st.graph_s.node_types[i] != graph_st.graph_t.node_types[corre[corre_index]]:
                    mis_type += 1
    if tm > 0:
        return cm / graph_st.n_overlap, cm / tm, cm, wm, m_dummy, tm, graph_st.n_overlap, graph_st.ns, mis_type / graph_st.ns
    else:
        return cm / graph_st.n_overlap, 0, cm, wm, m_dummy, tm, graph_st.n_overlap, graph_st.ns, mis_type / graph_st.ns
