import numpy as np
import os
import torch
import time
from scipy.sparse import csr_matrix
from lib.logger import Logger
from datetime import datetime
from methods.gw_cost import cor_io
from methods.regal import match_regal
from lib.matmul import mem_matmul, two_matmul, matmul_diag
from lib.factorization import torch_svd
from lapsolver import solve_dense
import torch.nn.functional as F
from data_io.real_noise_homo import edge_correctness, symm_substru_score
from methods.conealign import netmf, align_embeddings

prec = 1e-16
embed_d = 32


# embed_d = 512


def procustes(C1: torch.Tensor, C2: torch.Tensor, W: torch.Tensor, rho: float):
    M = C1 @ C2.T + rho * W
    X, Sigma, Y = torch_svd(M)
    # print("X.size={}, Y.size={}".format(X.size(), Y.size()))
    return X @ Y.T


def columnwise_simplex_projection(U: torch.Tensor):
    tmp = F.relu(U)
    column_sum = torch.sum(tmp, dim=0) + prec
    return tmp / column_sum


def match_pivot(graph_st, dataset: str, name: str, seed=1, use_cuda=False, ini="uni"):
    n_iter = 1000
    lda = 10
    rho = 0.1
    # rho = 1
    torch.set_grad_enabled(False)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    dtype = torch.float
    print("Running Pivot, assigning nodes non-uniform measures")
    start = time.time()
    # -------------------------------- tensorboard path -------------------------------------------
    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    board_path = os.path.join("results", dataset, name, "sd{}".format(seed),
                              timestamp + "_pivot_{}_{}_{}".format(lda, rho, embed_d))
    print(board_path)
    logger = Logger(board_path)
    # ------------------------------- tensorboard path defined ------------------------------------
    cor_s_tmp, cor_t_tmp, mu_s, mu_t = cor_io(graph_st=graph_st, device=device)
    del cor_s_tmp, cor_t_tmp
    # As, At = graph_st.graph_s.w, graph_st.graph_t.w  # this option considers edge weights
    As, At = (graph_st.graph_s.w > 0).astype(np.float), (graph_st.graph_t.w > 0).astype(np.float)
    As = torch.from_numpy(As).to(device).type(dtype)
    At = torch.from_numpy(At).to(device).type(dtype)
    ns, nt = graph_st.graph_s.n, graph_st.graph_t.n

    # ---------------------------- initialization ------------------------------------
    print("running CONE-Align")
    dim = 128
    window = 10
    negative = 1.0
    emb_matrixA = netmf(graph_st.graph_s.w, dim=dim, window=window, b=negative, normalize=True)
    emb_matrixB = netmf(graph_st.graph_t.w, dim=dim, window=window, b=negative, normalize=True)
    U = align_embeddings(emb_matrixA, emb_matrixB, adj1=csr_matrix(graph_st.graph_s.w),
                         adj2=csr_matrix(graph_st.graph_t.w), struc_embed=None, struc_embed2=None)
    U = U.todense()
    if isinstance(U, np.matrix):
        U = np.asarray(U)
    U = torch.from_numpy(U).to(device).type(dtype)
    U1, Sigma1, V1 = torch_svd(As)
    del As
    E1 = U1[:, 0: embed_d]
    del U1
    Q1 = matmul_diag(V1[:, 0: embed_d], 1 / Sigma1[0: embed_d])
    del V1
    C1 = torch.cat([E1, Q1], dim=1)
    inv_term = torch.inverse(At @ At + lda * torch.eye(nt, device=device))

    for t in range(n_iter):
        Q2 = inv_term @ (At @ U.T @ E1 + lda * U.T @ Q1)
        C2 = torch.cat([At @ Q2, lda * Q2], dim=1)
        W = columnwise_simplex_projection(U)
        U = procustes(C1, C2, W, rho)
        if t % 10 == 0:
            U_ary = U.to(torch.device("cpu")).numpy()
            rids, cids = solve_dense(-U_ary)
            corre = np.zeros(ns, dtype=np.int)
            for r, c in zip(rids, cids):
                corre[r] = c
            # print("corre={}".format(corre))
            result = graph_st.result_eval(corre)
            rec, prec = result[0], result[1]
            if rec == 0 or prec == 0:
                F1 = 0
            else:
                F1 = 2 * rec * prec / (rec + prec)
            if len(result) == 9:
                info = {'rec': rec, "prec": prec, "F1": F1, "type_mis": result[8]}
            else:
                info = {'rec': rec, "prec": prec, "F1": F1}
            ec = edge_correctness(graph_st, corre)
            s3 = symm_substru_score(graph_st, corre)
            info["ec"] = ec
            info["s3"] = s3
            info["runtime"] = time.time() - start
            print("iter_num={}, result={}, ec={}, s3={}, takes time {}".format(t, result, ec, s3, info["runtime"]))
            for tag, value in info.items():
                logger.scalar_summary(tag, value, t)

    return 0
