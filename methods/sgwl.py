from data_io.real_noise_homo import graph_pair_rn, report, edge_correctness, symm_substru_score
import numpy as np
import os
from copy import deepcopy as dc
import torch
import torch.optim as optim
import time
from scipy.sparse import csr_matrix
from scipy.special import softmax
import ot
from lib.logger import Logger
from datetime import datetime
from methods.gw_cost import cor_io, sinkhorn_iteration, GromovWassersteinEmbedding
from lib.torch_sinkhorn import normalize, peyre_expon, peri_proj
from lib.matmul import mem_matmul
from lib.util import trans_type_corre_no_expan

beta = 0.01
outer_iteration = 200
iter_bound = 1e-30
dtype = torch.float32

div_prec = 1e-16
log_prec = 1e-16
add_prec = 1e-16
stop_criterion = 1e-7

epochs = 50
sgd_iter = 400
proj_steps = 10


def estimate_target_distribution(probs: list, dim_t=2, balanced=False):
    """
    Estimate target distribution via the average of sorted source probabilities
    :param probs: a list of the distribution of source nodes (n_s, 1)
    :param dim_t: the dimension of target distribution
    :return: (dim_t, 1) vector representing a distribution
    """
    if balanced:
        return np.ones(dim_t) / dim_t
    else:
        p_t = np.zeros(dim_t)
        x_t = np.linspace(0, 1, p_t.shape[0])
        for n in range(len(probs)):
            p_s = probs[n]
            p_s = np.sort(p_s)[::-1]
            x_s = np.linspace(0, 1, p_s.shape[0])
            p_t_n = np.interp(x_t, x_s, p_s)
            p_t += p_t_n
        p_t /= np.sum(p_t)
        return p_t


def partition(X: tuple, clu_num=2, balanced=False):
    As, At, ps, pt, real_indices_s, real_indices_t = X
    p_bary = estimate_target_distribution([ps, pt], clu_num, balanced=balanced)
    A_bary = ot.gromov.gromov_barycenters(clu_num, [As, At], [ps, pt], p_bary, np.ones(2) / 2, "square_loss")
    trans_s = ot.gromov.gromov_wasserstein(As, A_bary, ps, p_bary, loss_fun='square_loss')
    trans_t = ot.gromov.gromov_wasserstein(At, A_bary, pt, p_bary, loss_fun='square_loss')
    decide_s = trans_s / p_bary
    decide_t = trans_t / p_bary
    member_s = np.argmax(decide_s, axis=1)
    member_t = np.argmax(decide_t, axis=1)
    X_list = []
    for i in range(clu_num):
        indices_s = np.where(member_s == i)[0]
        indices_t = np.where(member_t == i)[0]
        sub_As = As[indices_s][:, indices_s]
        sub_At = At[indices_t][:, indices_t]
        sub_ps, sub_pt = ps[indices_s], pt[indices_t]
        sub_ps /= np.sum(sub_ps)
        sub_pt /= np.sum(sub_pt)
        sub_indices_s, sub_indices_t = real_indices_s[indices_s], real_indices_t[indices_t]  # np.ndarray
        X_list.append((
            sub_As, sub_At, sub_ps, sub_pt, sub_indices_s, sub_indices_t
        ))
    return X_list


def recursive_graph_partition(As: np.ndarray, At: np.ndarray, ps: np.ndarray, pt: np.ndarray, clu_num=2,
                              partition_level=3, node_limit=200, balanced=False):
    """

    :param As and At: Adjacency matrices
    :param ps and pt: Marginal distributions
    :param clu_num: the number of clus when doing graph partition
    :param partition_level: the number of partition levels
    :param max_node_num: the maximum number of nodes in a sub-graph
    :return:
        par_s and par_t. par_s is a list and each of its elements are indices corresponding to one cluster.
        par_t is defined similarly.
    """

    def is_large(X: tuple, node_limit=200):
        sub_ns, sub_nt = len(X[2]), len(X[3])
        if sub_ns > node_limit and sub_nt > node_limit:
            return True
        else:
            return False

    def insert(X: tuple, sub_As_final: list, sub_At_final: list, sub_ps_final: list, sub_pt_final: list,
               sub_indices_s_final: list, sub_indices_t_final: list):
        sub_As_final.append(X[0])  # sub_As
        sub_At_final.append(X[1])  # sub_At
        sub_ps_final.append(X[2])  # sub_ps
        sub_pt_final.append(X[3])  # sub_pt
        sub_indices_s_final.append(X[4])  # sub_indices_s
        sub_indices_t_final.append(X[5])  # sub_indices_t
        return sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final, sub_indices_t_final

    sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final, sub_indices_t_final = [], [], [], [], [], []
    ns, nt = len(ps), len(pt)
    que = [(
        As, At, ps, pt, np.arange(ns), np.arange(nt)
    )]
    cur_par_level = 0
    while cur_par_level < partition_level:
        que_new = []
        while len(que):
            # print("que_new length {}, final length {}".format(len(que_new), len(sub_As_final)))
            X_cur = que[0]
            que.pop(0)
            if is_large(X_cur, node_limit=node_limit):
                que_new.extend(partition(X_cur, clu_num=clu_num, balanced=balanced))
            else:
                sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final, sub_indices_t_final = insert(
                    X_cur, sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final,
                    sub_indices_t_final)
        if len(que_new) == 0:
            break
        else:
            que = dc(que_new)
            cur_par_level += 1
    for X in que:
        sub_As_final.append(X[0])
        sub_At_final.append(X[1])
        sub_ps_final.append(X[2])
        sub_pt_final.append(X[3])
        sub_indices_s_final.append(X[4])
        sub_indices_t_final.append(X[5])
    return sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final, sub_indices_t_final


def match_sgwl(graph_st: graph_pair_rn, dataset="ppi", name="800_900_900_10_4", seed=1, use_cuda=False, ini="uni",
               clu_num=2, partition_level=3, node_limit=200, md_lr=1e2, balanced=True):
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("Running S-GWL")
    start = time.time()
    # ============================ tensorboard path =============================

    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    board_path = os.path.join("results", dataset, name, "sd{}".format(seed),
                              timestamp + "_sgwl_{}_{}_{}_{}_{}".format(ini, clu_num, partition_level, node_limit,
                                                                        md_lr))
    print(board_path)
    logger = Logger(board_path)

    # ================================ IO start =================================
    ws, wt = graph_st.graph_s.w, graph_st.graph_t.w
    ns, nt = graph_st.graph_s.n, graph_st.graph_t.n
    ps, pt = np.sum(ws, axis=1), np.sum(wt, axis=1)
    ps = ps / np.sum(ps)
    pt = pt / np.sum(pt)

    # =========================== recursive partition =============================
    print("Partitioning")
    sub_As_final, sub_At_final, sub_ps_final, sub_pt_final, sub_indices_s_final, sub_indices_t_final = recursive_graph_partition(
        ws, wt, ps, pt, clu_num=clu_num, partition_level=partition_level, node_limit=node_limit, balanced=balanced)
    assert len(sub_As_final) == len(sub_At_final)
    # print(sub_indices_s_final)
    for l in sub_indices_s_final:
        print(len(l))
    As_list = [torch.from_numpy(sub_As).to(device).type(dtype) for sub_As in sub_As_final]
    At_list = [torch.from_numpy(sub_At).to(device).type(dtype) for sub_At in sub_At_final]
    mu_s_list = [torch.from_numpy(sub_ps).to(device).type(dtype).unsqueeze_(1) for sub_ps in sub_ps_final]
    mu_t_list = [torch.from_numpy(sub_pt).to(device).type(dtype).unsqueeze_(1) for sub_pt in sub_pt_final]
    # ================== main iterations, estimating the node correspondence for each cluster ==============
    indices_fi = []  # the indices of nodes that we finish matching
    corre_fi = []  # estimated node correspondence
    log_iter_num = 0
    corre_init = np.random.choice(nt, ns)
    for clu_id in range(len(sub_indices_s_final)):
        print("clu_id={}".format(clu_id))
        indices_s, indices_t = list(sub_indices_s_final[clu_id]), list(sub_indices_t_final[clu_id])
        if len(indices_s) < 2 or len(indices_t) < 2:
            continue
        print("indices_s shape {}, indices_t shape {}".format(len(indices_s), len(indices_t)))
        md_iter = max(len(indices_s), len(indices_t), outer_iteration)
        trans_clu = torch.matmul(mu_s_list[clu_id], torch.t(mu_t_list[clu_id]))
        for _ in range(md_iter):
            g = peyre_expon(As_list[clu_id], At_list[clu_id], trans_clu)
            tmp = trans_clu * torch.exp(-1 - md_lr * g) + add_prec
            trans_clu_new = normalize(tmp, mu_s=mu_s_list[clu_id], mu_t=mu_t_list[clu_id], n_iter=proj_steps)
            log_iter_num += 1
            if log_iter_num % 1 == 1:
                trans_clu_array = trans_clu_new.detach_().to(torch.device("cpu")).numpy()
                corre_tmp = trans_type_corre_no_expan(trans_clu_array, indices_s=indices_s, indices_t=indices_t)
                # result, ec, s3 = report(corre=corre_fi + corre_tmp, indices=indices_fi + indices_s, graph_st=graph_st)
                corre = dc(corre_init)
                corre[indices_fi + indices_s] = np.array(corre_fi + corre_tmp)
                result = graph_st.result_eval(corre)
                ec = edge_correctness(graph_st, corre)
                s3 = symm_substru_score(graph_st, corre)
                rec, prec = result[0], result[1]
                if rec == 0 or prec == 0:
                    F1 = 0
                else:
                    F1 = 2 * rec * prec / (rec + prec)
                if len(result) == 9:
                    info = {'rec': rec, "prec": prec, "F1": F1, "type_mis": result[8]}
                else:
                    info = {'rec': rec, "prec": prec, "F1": F1}
                info["ec"] = ec
                info["s3"] = s3
                info["runtime"] = time.time() - start
                print("result={}, ec={}, s3={}, runtime={}".format(result, ec, s3, info["runtime"]))
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, log_iter_num)
            if torch.max(torch.abs(trans_clu - trans_clu_new)) < stop_criterion:
                trans_clu_array = trans_clu_new.detach_().to(torch.device("cpu")).numpy()
                corre_tmp = trans_type_corre_no_expan(trans_clu_array, indices_s=indices_s, indices_t=indices_t)
                # result, ec, s3 = report(corre=corre_fi + corre_tmp, indices=indices_fi + indices_s, graph_st=graph_st)
                corre = dc(corre_init)
                corre[indices_fi + indices_s] = np.array(corre_fi + corre_tmp)
                result = graph_st.result_eval(corre)
                ec = edge_correctness(graph_st, corre)
                s3 = symm_substru_score(graph_st, corre)
                rec, prec = result[0], result[1]
                if rec == 0 or prec == 0:
                    F1 = 0
                else:
                    F1 = 2 * rec * prec / (rec + prec)
                if len(result) == 9:
                    info = {'rec': rec, "prec": prec, "F1": F1, "type_mis": result[8]}
                else:
                    info = {'rec': rec, "prec": prec, "F1": F1}
                info["ec"] = ec
                info["s3"] = s3
                info["runtime"] = time.time() - start
                print("result={}, ec={}, s3={}, runtime={}".format(result, ec, s3, info["runtime"]))
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, log_iter_num)
                break
            else:
                trans_clu = trans_clu_new
        corre_fi = corre_fi + corre_tmp
        indices_fi = indices_fi + indices_s
    return 0
