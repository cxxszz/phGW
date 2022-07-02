import networkx as nx
import matplotlib.pyplot as plt
G = nx.star_graph(3)
L = nx.line_graph(G)
print(G.nodes, G.edges)
print(L.nodes, L.edges)
nx.draw(L, with_labels=True)
plt.show()