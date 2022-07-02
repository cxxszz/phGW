import numpy as np
from numpy import linalg as la
import os
import time
import torch
from lib.logger import Logger
from lib.factorization import low_rank_app, low_rank_torch
from lib.torch_sinkhorn import peyre_expon_low_rank, peri_proj, peyre_expon
from datetime import datetime
from data_io.real_noise_homo import graph_pair_rn, edge_correctness, symm_substru_score
from methods.gw_cost import cor_io

n_iter = 2000
add_prec = 1e-30
stop_prec = 1e-9
proj_steps = 30


def match_qegw(graph_st: graph_pair_rn, dataset="ppi", name="800_900_900_10_4", seed=1, use_cuda=False, rank=2,
               md_lr=1e2):
    print("Running Quadratic Entropic-GW")
    start = time.time()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # dtype = torch.double
    dtype = torch.float
    # ============================ tensorboard path =============================

    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    board_path = os.path.join("results", dataset, name, "sd{}".format(seed),
                              timestamp + "_qegw_{}_{}".format(rank, md_lr))
    print(board_path)
    logger = Logger(board_path)
    Bs, Bt, mu_s, mu_t = cor_io(graph_st=graph_st, device=device, dtype=dtype)
    _, Bs1, Bs2T = low_rank_torch(Bs, k=rank)
    _, Bt1, Bt2T = low_rank_torch(Bt, k=rank)
    trans = torch.matmul(mu_s, torch.t(mu_t))
    for iter_num in range(n_iter):
        grad = peyre_expon_low_rank(Bs, Bt, trans, Bs1, Bs2T, Bt1, Bt2T)
        # grad = peyre_expon(Bs, Bt, trans)
        tmp = trans * np.exp(-1 - md_lr * grad) + add_prec
        trans_new = peri_proj(trans=tmp, mu_s=mu_s, mu_t=mu_t, total_mass=1.0, n_iter=proj_steps)
        if iter_num % 1 == 0:
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
        if torch.max(torch.abs(trans_new - trans)) < stop_prec:
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
