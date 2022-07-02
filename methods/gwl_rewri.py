from data_io.real_data_homo import graph_pair
import numpy as np
import os
import torch
import torch.optim as optim
import time
from lib.logger import Logger
from datetime import datetime
from data_io.real_noise_homo import edge_correctness, symm_substru_score
from methods.gw_cost import cor_io, sinkhorn_iteration, GromovWassersteinEmbedding
from lib.torch_sinkhorn import normalize, peyre_expon, peri_proj
from methods.regal import match_regal
from lib.matmul import mem_matmul

beta = 0.01
outer_iteration = 200
iter_bound = 1e-30

div_prec = 1e-16
log_prec = 1e-16

epochs = 50
sgd_iter = 400


def match_gwl(graph_st: graph_pair, dataset="ppi", name="800_900_900_10_4", seed=1, use_cuda=False, ini="uni",
              wei_stru=0.95):
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("Running GWL similarity")
    start = time.time()
    # ============================ tensorboard path =============================

    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    board_path = os.path.join("results", dataset, name, "sd{}".format(seed),
                              timestamp + "_GWL_similarity_{}".format(ini))
    print(board_path)
    logger = Logger(board_path)

    # ================================ IO start =================================
    cor_s_tmp, cor_t_tmp, mu_s, mu_t = cor_io(graph_st=graph_st, device=device)
    cor_s = (cor_s_tmp > 0).float()
    cor_t = (cor_t_tmp > 0).float()
    del cor_s_tmp, cor_t_tmp
    ns = mu_s.size(0)
    nt = mu_t.size(0)
    # ---------------------------- initialization ------------------------------------
    if ini == "regal":  # REGAL-based initialization
        ini_path = os.path.join("data", dataset, name + "_ini_regal_{}.npy".format(seed))
        if os.path.exists(ini_path):
            init_corre = np.load(ini_path)
        else:
            init_corre = match_regal(graph_st, dataset=dataset, name=name, seed=seed)
        trans = torch.zeros((ns, nt), device=device)
        for i in range(ns):
            iprime = init_corre[i]
            trans[i, iprime] = 1
    elif ini == "uni":  # uniform initialization
        trans = torch.matmul(mu_s, torch.t(mu_t))
    else:  # random initialization
        init_corre = torch.arange(ns)
        perm = torch.randperm(ns)
        init_corre = init_corre[perm]
        trans = torch.zeros((ns, nt), device=device)
        for i in range(ns):
            iprime = init_corre[i]
            trans[i, iprime] = mu_s[i, 0]
    # trans = normalize(sim=trans + 1e-32, mu_s=mu_s, mu_t=mu_t, n_iter=2)
    trans = normalize(sim=trans + 1e-32, mu_s=mu_s, mu_t=mu_t,
                      n_iter=30)  # adding 1e-32 enhances numerical stability and avoiding ZeroDivision
    # trans = peri_proj(trans=trans + 1e-32, mu_s=mu_s, mu_t=mu_t, total_mass=1.0, n_iter=30)
    # print("trans={}".format(trans))
    # input()

    index_s = torch.from_numpy(
        np.array(range(ns))
    )
    index_t = torch.from_numpy(
        np.array(range(nt))
    )
    index_s = index_s.type(torch.LongTensor)
    index_t = index_t.type(torch.LongTensor)
    index_s, index_t = index_s.to(device), index_t.to(device)
    # IO finished
    ####################################################################################

    emb_model = GromovWassersteinEmbedding(ns, nt, dim=32, device=device)
    emb_model.to(device)
    op = optim.Adam(emb_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        alpha = max([(epochs - epoch) / epochs, wei_stru])
        cor_s_emb = 1 - emb_model.self_cost_mat(index_s, 0).data
        cor_t_emb = 1 - emb_model.self_cost_mat(index_t, 1).data
        cost_st_12 = emb_model.mutual_cost_mat(index_s, index_t).data
        cor1 = alpha * cor_s + (1 - alpha) * cor_s_emb
        cor2 = alpha * cor_t + (1 - alpha) * cor_t_emb
        cost_mutual = (1 - alpha) * cost_st_12
        del cor_s_emb, cor_t_emb, cost_st_12

        # f1_st = torch.matmul(cor1 ** 2, mu_s).repeat(1, nt)
        # f2_st = torch.matmul(torch.t(mu_t), torch.t(cor2 ** 2)).repeat(ns, 1)
        # cost_st = f1_st + f2_st
        ####################################################################################
        # learning transport plan
        ####################################################################################
        for t in range(outer_iteration):
            # cost = cost_st - 2 * mem_matmul(mem_matmul(cor1, trans), torch.t(cor2)) + 0.1 * cost_mutual
            cost = peyre_expon(cor1, cor2, trans) + 0.1 * cost_mutual
            # cost = cost_st - 2 * torch.matmul(torch.matmul(cor1, trans), torch.t(cor2)) + 0.1 * cost_mutual
            # trans_new = sinkhorn_iteration(cost=cost, mu_s=mu_s, mu_t=mu_t, trans0=trans, beta=beta)
            # trans_new = sinkhorn_iteration(cost=cost, mu_s=mu_s, mu_t=mu_t, trans0=trans, beta=beta, inner_iteration=10)
            trans_new = sinkhorn_iteration(cost=cost, mu_s=mu_s, mu_t=mu_t, trans0=trans, beta=beta, inner_iteration=30)
            # print("trans={}".format(trans_new))
            # input()
            rela_error = torch.sum(torch.abs(trans_new - trans)) / torch.sum(torch.abs(trans))
            del cost, trans
            trans = trans_new
            if rela_error < iter_bound:
                break
            if t % 2 == 0:
                print('sinkhorn iter {}/{}'.format(t, outer_iteration))
                if use_cuda:
                    trans_tmp = trans.detach_().cpu().numpy()
                else:
                    trans_tmp = trans.detach_().numpy()
                corre = np.argmax(trans_tmp, axis=1)
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
                print("t={}, result={}, ec={}, s3={}".format(t, result, ec, s3))
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch * outer_iteration + t)
        ####################################################################################
        # learning embedding
        ####################################################################################
        # trans_tmp = np.zeros((ns, nt))
        # trans_np = trans.cpu().data.numpy()
        # index_s_np = index_s.cpu().data.numpy()
        # index_t_np = index_t.cpu().data.numpy()
        # patch = trans[index_s_np, :]
        # patch = patch[:, index_t_np]
        # energy = np.sum(patch) + 1
        # for row in range(trans_np.shape[0]):
        #     for col in range(trans_np.shape[1]):
        #         trans_tmp[index_s_np[row], index_t_np[col]] += (energy * trans_np[row, col])
        cost1, cost2 = 1 - cor1, 1 - cor2
        del cor1, cor2, cost_mutual
        for num in range(sgd_iter):
            op.zero_grad()

            # loss_gw, loss_w, regularizer = emb_model(index_s, index_t, trans, mu_s, mu_t, cost1, cost2,
            #                                          prior=cost_mutual, mask1=None, mask2=None, mask12=None)
            loss_gw, loss_w, regularizer = emb_model(index_s, index_t, trans, mu_s, mu_t, cost1, cost2,
                                                     prior=None, mask1=None, mask2=None, mask12=None)
            loss = 1e3 * loss_gw + 1e3 * loss_w + regularizer
            loss.backward()
            op.step()
            if num % 10 == 0:
                print('inner {}/{}: loss={:.6f}.'.format(num, sgd_iter, loss.data))
    return 0
