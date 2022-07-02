"""
A pair of correlated heterogeneous graphs, but with simulated noise.
"""
import numpy as np
from data_io.real_data_homo import graph as graph_base, graph_pair as graph_pair_base


class graph(graph_base):
    def __init__(self, w: np.ndarray, lying=None):
        super(graph, self).__init__(w=w, lying=lying)

    def my_init(self, w: np.ndarray, lying=None):
        self.w = w
        self.n = len(w)
        self.lying = lying  # underlying node, i.e., the node order in the full graph
        self.values = None
        self.node_types = None
        self.type_nodes = {}
        return self

    def set_values(self, values: np.ndarray):
        assert len(values) == self.n
        self.values = values
        return self

    def set_node_types(self, node_types: np.ndarray):
        assert len(node_types) == self.n
        self.node_types = node_types
        self.get_type_nodes()
        return self

    def get_type_nodes(self):
        self.type_nodes = {}
        for i in range(self.n):
            node_type = self.node_types[i]
            if node_type not in self.type_nodes:
                self.type_nodes[node_type] = [i]
            else:
                self.type_nodes[node_type].append(i)
        return self
