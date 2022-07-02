from data_io.real_noise_homo import graph_pair_rn
from data_io.real_noise_homo import edge_correctness, symm_substru_score
import numpy as np
from numpy import linalg as la
import os
import time
from numba import jit
from multiprocessing import Pool
from lib.logger import Logger
from datetime import datetime
from lib.partition import h_matrix1csr as h_matrix
# from lib.partition import h_matrix_recur as h_matrix
from lib.sinkhorn import normalize

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
symmetric = True
verbose = True

n_iter = 800
add_prec = 1e-30
stop_prec = 1e-7


def peyre_expon_h2(ws: np.ndarray, wt: np.ndarray, trans: np.ndarray, ps: np.ndarray, pt: np.ndarray, h2s: h_matrix,
                   h2t: h_matrix):
    """

    :param ws:
    :param wt:
    :param trans:
    :param ps:
    :param pt:
    :return:
    """
    ns, nt = len(ps), len(pt)
    deg_terms = np.outer(ws @ ps, np.ones(nt))
    deg_terms += np.outer(np.ones(ns), wt @ pt)

    tmp = h2s.rdot_matrix(trans)
    num = h2t.ldot_matrix(tmp)

    # num = ws @ trans @ wt
    return deg_terms - 2 * num


def func(data1, rows, data2, columns):
    return data1.matrix[rows][:, columns]


def match_hi(graph_st: graph_pair_rn, dataset="ppi", name="800_900_900_10_4", seed=1, use_cuda=False, ini="uni",
             n_par=2, par_level=3, rank=2, md_lr=1e2, node_limit=200):
    print("Running hi")
    start = time.time()
    # ============================ tensorboard path =============================

    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    board_path = os.path.join("results", dataset, name, "sd{}".format(seed),
                              timestamp + "_hi_{}_{}_{}_{}_{}_{}".format(ini, n_par, rank, md_lr, par_level,
                                                                         node_limit))
    print(board_path)
    logger = Logger(board_path)
    ws, wt = graph_st.graph_s.w, graph_st.graph_t.w
    ps, pt = np.sum(ws, axis=1), np.sum(wt, axis=1)
    ps = ps / np.sum(ps)
    pt = pt / np.sum(pt)
    ns, nt = len(ps), len(pt)
    h2s = h_matrix(ws, n_par=n_par, rank=rank)
    h2t = h_matrix(wt, n_par=n_par, rank=rank)
    # h2s = h_matrix(ws, n_par=n_par, rank=rank, par_level=par_level, node_limit=node_limit)
    # h2t = h_matrix(wt, n_par=n_par, rank=rank, par_level=par_level, node_limit=node_limit)
    trans = np.outer(ps, pt)
    for iter_num in range(n_iter):
        print("calculating grad")
        grad = peyre_expon_h2(ws=ws, wt=wt, trans=trans, ps=ps, pt=pt, h2s=h2s, h2t=h2t)
        print("grad calculated")
        tmp = trans * np.exp(-1 - md_lr * grad) + add_prec
        trans_new = normalize(tmp, ps, pt)
        if iter_num % 20 == 0:
            corre = np.argmax(trans_new, axis=1)
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
            print(
                "iter_num={}, result={}, ec={}, s3={}, takes time {}".format(iter_num, result, ec, s3, info["runtime"]))
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter_num)
        if np.max(np.abs(trans_new - trans)) < stop_prec:
            corre = np.argmax(trans_new, axis=1)
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
            print(
                "iter_num={}, result={}, ec={}, s3={}, takes time {}".format(iter_num, result, ec, s3, info["runtime"]))
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter_num)
            break
        else:
            trans = trans_new
    return trans
