import numpy as np
import torch
from copy import deepcopy as dc
from data_io.real_noise_homo import graph_pair_rn, edge_correctness, symm_substru_score
import methods.xnetmf as xnetmf
from methods.regal_alignments import get_embedding_similarities
import time

use_cuda = True
device = torch.device('cuda:0' if use_cuda else 'cpu')

beta = 0.025
outer_iteration = 40000
iter_bound = 1e-30
# fused_wei = 0.3
fused_wei = 0

############################################################
## REGAL parameters
dimensions = 128  # Number of dimensions
k = 100  # Controls of landmarks to sample
untillayer = 2  # Calculation until the layer for xNetMF
alpha = 0.05  # Discount factor for further layers
num_buckets = 2  # base of log for degree (node feature) binning
gammastruc = 1  # Weight on structural similarity
gammaattr = 1  # Weight on attribute similarity
numtop = 10  # Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities


class RepMethod(object):
    def __init__(self, align_info=None, p=None, k=10, max_layer=None, alpha=0.1, num_buckets=None, normalize=True,
                 gammastruc=1, gammaattr=1):
        self.p = p  # sample p points
        self.k = k  # control sample size
        self.max_layer = max_layer  # furthest hop distance up to which to compare neighbors
        self.alpha = alpha  # discount factor for higher layers
        self.num_buckets = num_buckets  # number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
        self.normalize = normalize  # whether to normalize node embeddings
        self.gammastruc = gammastruc  # parameter weighing structural similarity in node identity
        self.gammaattr = gammaattr  # parameter weighing attribute similarity in node identity


class Graph(object):
    # Undirected, unweighted
    def __init__(self, adj, num_buckets=None, node_labels=None, edge_labels=None, graph_label=None,
                 node_attributes=None, true_alignments=None):
        self.G_adj = adj  # adjacency matrix
        self.N = self.G_adj.shape[0]  # number of nodes
        self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
        self.max_degree = max(self.node_degrees)
        self.num_buckets = num_buckets  # how many buckets to break node features into

        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.node_attributes = node_attributes  # N x A matrix, where N is # of nodes, and A is # of attributes
        self.kneighbors = None  # dict of k-hop neighbors for each node
        self.true_alignments = true_alignments  # dict of true alignments, if this graph is a combination of multiple graphs


def learn_representations(adj):
    print("got adj matrix")

    graph = Graph(adj)
    max_layer = untillayer
    rep_method = RepMethod(max_layer=max_layer, alpha=alpha, k=k, num_buckets=num_buckets, normalize=True,
                           gammastruc=gammastruc, gammaattr=gammaattr)
    if max_layer is None:
        max_layer = 1000
    print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
    representations = xnetmf.get_representations(graph, rep_method)
    return representations


def match_regal(graph_st: graph_pair_rn, dataset="ppi", name="800_900_900_10_4", seed=1, rounds=2):
    print("Running REGAL")

    ns = graph_st.graph_s.n
    nt = graph_st.graph_t.n

    ws = dc(graph_st.graph_s.w)
    wt = dc(graph_st.graph_t.w)
    # for i in range(ns):
    #     if np.sum(ws[i]) < 0.5:
    #         j = np.random.choice(ns)
    #         print("ws[{}]={}".format(i, ws[i]))
    #         print("ws[{}]={}".format(j, ws[j]))
    #         assert ws[i][j] == 0 and ws[j][i] == 0
    #         ws[i][j] = 1
    #         ws[j][i] = 1
    # for i in range(nt):
    #     if np.sum(wt[i]) < 0.5:
    #         j = np.random.choice(nt)
    #         print("wt[{}]={}".format(i, wt[i]))
    #         print("wt[{}]={}".format(j, wt[j]))
    #         assert wt[i][j] == 0 and wt[j][i] == 0
    #         wt[i][j] = 1
    #         wt[j][i] = 1
    for t in range(rounds):
        start = time.time()
        w_comb = np.concatenate((
            np.concatenate((ws, np.zeros((ns, nt))), axis=1),
            np.concatenate((np.zeros((nt, ns)), wt), axis=1)
        ), axis=0)
        embed = learn_representations(w_comb)
        embed_s = embed[0:ns]
        embed_t = embed[ns:]
        # print(embed_s.shape)

        alignment_matrix = get_embedding_similarities(embed_s, embed_t, num_top=numtop)
        alignment_matrix = alignment_matrix.todense()
        # print(alignment_matrix)
        print(alignment_matrix.shape)
        corre = np.argmax(alignment_matrix, axis=1)
        result = graph_st.result_eval(corre)
        ec = edge_correctness(graph_st, corre)
        s3 = symm_substru_score(graph_st, corre)
        print("t={}, result={}, ec={}, s3={}, takes time {}".format(t, result, ec, s3, time.time() - start))
    return corre
