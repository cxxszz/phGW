from data_io.real_data_homo import graph_pair
import numpy as np
import sklearn.metrics.pairwise
import os
from copy import deepcopy as dc
import time
from scipy import sparse
import theano
from theano import tensor as T
import scipy.sparse as sps
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import ot
from lapsolver import solve_dense

dim = 128
window = 10
negative = 1.0
embsim = "euclidean"
niter_init = 10
reg_init = 1.0
lr = 1.0
bsz = 10
nepoch = 5
niter_align = 10
reg_align = 0.05
alignmethod = "greedy"
numtop = 10
rel_reg = 0.1


# Full NMF matrix (which NMF factorizes with SVD)
# Taken from MILE code
def netmf_mat_full(A, window=10, b=1.0):
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    # print "A shape", A.shape
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = sparse.csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        # print "Compute matrix %d-th power" % (i + 1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


# Used in NetMF, AROPE
def svd_embed(prox_sim, dim):
    u, s, v = sparse.linalg.svds(prox_sim, dim, return_singular_vectors="u")
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf(A, dim=128, window=10, b=1.0, normalize=True):
    prox_sim = netmf_mat_full(A, window, b)
    embed = svd_embed(prox_sim, dim)
    if normalize:
        norms = np.linalg.norm(embed, axis=1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    return embed


def objective(X, Y, R, n=5):
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    if isinstance(C, np.matrix):
        C = np.asarray(C)
    # P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    this_reg = rel_reg * np.max(C)
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, this_reg, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def procrustes(X_src, Y_tgt):
    '''
    print "procrustes:", Y_tgt, X_src
    print np.isnan(Y_tgt).any(), np.isinf(Y_tgt).any()
    print np.isnan(X_src).any(), np.isinf(X_src).any()
    print np.min(Y_tgt), np.max(Y_tgt)
    print np.min(X_src), np.max(X_src)
    dot = np.dot(Y_tgt.T, X_src)
    print np.isnan(dot).any(), np.isinf(dot).any()
    print np.min(dot), np.max(dot)
    '''
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)


def align(X, Y, R, lr=1.0, bsz=10, nepoch=5, niter=10,
          nmax=10, reg=0.05, verbose=True, project_every=True):
    for epoch in range(1, nepoch + 1):
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            # print bsz, C.shape
            # print("type(C)={}".format(type(C)))
            if isinstance(C, np.matrix):
                C = np.asarray(C)
            # P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            this_reg = rel_reg * np.max(C)
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, this_reg, stopThr=1e-3)
            # print P.shape, C.shape
            # compute gradient
            # print "random values from embeddings:", xt, yt
            # print "sinkhorn", np.isnan(P).any(), np.isinf(P).any()
            # Pyt = np.dot(P, yt)
            # print "Pyt", np.isnan(Pyt).any(), np.isinf(Pyt).any()
            G = - np.dot(xt.T, np.dot(P, yt))
            # print "G", np.isnan(G).any(), np.isinf(G).any()
            update = lr / bsz * G
            print(("Update: %.3f (norm G %.3f)" % (np.linalg.norm(update), np.linalg.norm(G))))
            R -= update

            # project on orthogonal matrices
            if project_every:
                U, s, VT = np.linalg.svd(R)
                R = np.dot(U, VT)
        niter //= 4
        if verbose:
            print(("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R))))
    if not project_every:
        U, s, VT = np.linalg.svd(R)
        R = np.dot(U, VT)
    return R, P


def convex_init(X, Y, niter=10, reg=1.0, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        # q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        this_reg = rel_reg * np.max(G)
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, this_reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    print(obj)
    return procrustes(np.dot(P, X), Y).T, P


def peri_proj_np(trans, p, q, total_mass=0.9, n_iter=100):
    """

    :param trans:
    :param p: array of shape (ns,)
    :param q: array of shape (nt,)
    :param total_mass:
    :param n_iter:
    :return:
    """
    div_prec = 1e-16

    def cw_min(x1, x2):
        """
        component-wise minimum
        :param x1: array of shape (n,)
        :param x2: array of shape (n,)
        :return:
        """
        x1c = np.reshape(x1, (-1, 1))  # (n, 1)
        x2c = np.reshape(x2, (-1, 1))
        print("x1c has size {}, x2c has size {}".format(x1c.shape, x2c.shape))
        X = np.concatenate((x1c, x2c), axis=1)
        return np.min(X, axis=1)

    for _ in range(n_iter):
        print("trans has size {}, p has size {}, q has size {}".format(trans.shape, p.shape, q.shape))
        print("div_prec + np.sum(trans, axis=1)={}".format(div_prec + np.sum(trans, axis=1)))
        print(
            "p / (div_prec + np.sum(trans, axis=1)) has size {}".format((p / (div_prec + np.sum(trans, axis=1))).shape))
        P_p_d = cw_min(p / (div_prec + np.sum(trans, axis=1)), np.ones(p.shape))
        trans *= np.reshape(P_p_d, (-1, 1))

        P_q_d = cw_min(q / (div_prec + np.sum(trans, axis=0)), np.ones(q.shape))
        trans *= P_q_d

        trans /= np.sum(trans)
        trans *= total_mass
    return trans


def sparsen(x, thre=1e-3):
    return (x > thre).astype(np.float) * x


def convex_init_sparse(X, Y, K_X=None, K_Y=None, niter=10, reg=1.0, apply_sqrt=False, P=None):
    if P is not None:  # already given initial correspondence--then just procrustes
        return procrustes(P.dot(X), Y).T, P
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    if K_X is None:
        K_X = np.dot(X, X.T)
    if K_Y is None:
        K_Y = np.dot(Y, Y.T)
    print((type(K_X), K_X.shape))
    K_Y *= sparse.linalg.norm(K_X) / sparse.linalg.norm(K_Y)
    K2_X, K2_Y = K_X.dot(K_X), K_Y.dot(K_Y)
    # print K_X, K_Y, K2_X, K2_Y
    K_X, K_Y, K2_X, K2_Y = K_X.toarray(), K_Y.toarray(), K2_X.toarray(), K2_Y.toarray()
    # P = np.ones([n, n]) / float(n)
    nX = X.shape[0]
    nY = Y.shape[0]
    P = np.ones([nY, nX]) / float(nX)

    for it in range(1, niter + 1):
        if it % 10 == 0: print(it)
        G = P.dot(K2_X) + K2_Y.dot(P) - 2 * K_Y.dot(P.dot(K_X))
        # G = G.todense() #TODO how to get around this??
        if nX == nY:
            # ------------------------- Approach 1, solve_dense based -----------------------------
            # rids, cids = solve_dense(G)
            # q = np.zeros((nY, nX))
            # for r, c in zip(rids, cids):
            #     q[r, c] = 1
            # ---------------- Approach 2 (original), might be unstable on some datasets-----------
            if isinstance(G, np.matrix):
                G = np.asarray(G)
            this_reg = rel_reg * np.max(G)
            q = ot.sinkhorn(np.ones(n), np.ones(n), G, this_reg, stopThr=1e-3)
        else:
            # ------------------------- Approach 1, solve_dense based -----------------------------
            rids, cids = solve_dense(G)
            q = np.zeros((nY, nX))
            for r, c in zip(rids, cids):
                q[r, c] = 1
            # ------------------------- Approach 2, unstable ----------------------------------------
            # q = ot.sinkhorn(np.ones(nY), np.ones(nX), G, reg, stopThr=1e-3)
            # ------------------------- Approach 3, partial OT --------------------------------------
            # mu_s = np.ones(nY)
            # mu_t = np.ones(nX)
            # M = np.exp(-G / reg)
            # M = M / np.max(M)
            # print(M.shape, mu_s.shape, mu_t.shape)
            # trans = peri_proj_np(M, p=mu_s, q=mu_t, total_mass=np.min([nX, nY]), n_iter=10)
            # q = sparsen(trans)
            # q = ot.partial.partial_wasserstein(np.ones(nY), np.ones(nX), G, m=np.min([nX, nY]))
        q = sparse.csr_matrix(q)
        # print q.shape
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(P.dot(K_X) - K_Y.dot(P))
    print(obj)
    return procrustes(P.dot(X), Y).T, P


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def align_embeddings(embed1, embed2, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if embsim == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1): adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2): adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = convex_init_sparse(embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False,
                                                niter=niter_init, reg=reg_init, P=corr)
    else:
        init_sim, corr_mat = convex_init(embed1, embed2, apply_sqrt=False, niter=niter_init, reg=reg_init)
    print(corr_mat)
    print(np.max(corr_mat, axis=0))
    print(np.max(corr_mat, axis=1))

    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = align(embed1, embed2, init_sim, lr=lr, bsz=bsz,
                                       nepoch=nepoch, niter=niter_align, reg=reg_align)
    print(dim_align_matrix.shape, corr_mat.shape)

    # Step 3: Match Nodes with Similar Embeddings
    # Align embedding spaces
    aligned_embed1 = embed1.dot(dim_align_matrix)
    # Greedily match nodes
    if alignmethod == 'greedy':  # greedily align each embedding to most similar neighbor
        # KD tree with only top similarities computed
        if numtop is not None:
            alignment_matrix = kd_align(aligned_embed1, embed2, distance_metric=embsim, num_top=numtop)
        # All pairwise distance computation
        else:
            if embsim == "cosine":
                alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(aligned_embed1, embed2)
            else:
                alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
                alignment_matrix = np.exp(-alignment_matrix)
    return alignment_matrix


def match_conealign(graph_st: graph_pair, dataset="ppi", name="800_900_900_10_4", seed=1):
    print("Running CONE-Align")
    start = time.time()

    ns = graph_st.graph_s.n
    nt = graph_st.graph_t.n

    ws = dc(graph_st.graph_s.w)
    wt = dc(graph_st.graph_t.w)
    # -------- step1: obtain normalized proximity-preserving node embeddings ---------------
    emb_matrixA = netmf(ws, dim=dim, window=window, b=negative, normalize=True)
    emb_matrixB = netmf(wt, dim=dim, window=window, b=negative, normalize=True)

    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    alignment_matrix = align_embeddings(emb_matrixA, emb_matrixB, adj1=csr_matrix(ws), adj2=csr_matrix(wt),
                                        struc_embed=None, struc_embed2=None)

    # print(alignment_matrix)
    print(alignment_matrix.shape)
    corre = np.argmax(alignment_matrix, axis=1)
    result = graph_st.result_eval(corre)
    print("result={}".format(result))
    print("CONE-Align takes time {}".format(time.time() - start))
    save_path = os.path.join("data", dataset, name + "_ini_cone_{}.npy".format(seed))
    np.save(save_path, corre)
    return corre
