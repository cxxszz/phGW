import pickle
import numpy as np
import torch
import os
import argparse
from data_io.real_noise_homo import graph_pair_rn
from methods.regal import match_regal
from methods.gwl_rewri import match_gwl
from methods.hi import match_hi
from methods.hi_recur import match_hi as match_hi_recur
from methods.qegw import match_qegw
from methods.qgw import match_qgw
from methods.sgwl import match_sgwl
from methods.SpectralPivot import match_pivot


def main(args):
    alg = args.alg
    seed = args.seed
    ini = args.ini
    n_par = args.npar
    par_level = args.level
    rank = args.rank
    md_lr = args.lr
    node_limit = args.limit
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # print("prune=", args.prune)
    # ------------------------------ Arenas --------------------------------
    if args.dataset == "arenas" and args.noise == 0:
        data_path = "data/arenas/1133_1133_1133_10_4_0.p"
    elif args.dataset == "arenas" and args.noise == 0.05:
        data_path = "data/arenas/1133_1133_1133_10_4_0.05.p"
    elif args.dataset == "arenas" and args.noise == 0.1:
        data_path = "data/arenas/1133_1133_1133_10_4_0.1.p"
    elif args.dataset == "arenas" and args.noise == 0.15:
        data_path = "data/arenas/1133_1133_1133_10_4_0.15.p"
    elif args.dataset == "arenas" and args.noise == 0.2:
        data_path = "data/arenas/1133_1133_1133_10_4_0.2.p"
    elif args.dataset == "arenas" and args.noise == 0.25:
        data_path = "data/arenas/1133_1133_1133_10_4_0.25.p"
    # ------------------------------ FM -------------------------------------
    elif args.dataset == "fm" and args.noise == 0:
        data_path = "data/fm/7624_7624_7624_10_4_0.p"
    elif args.dataset == "fm" and args.noise == 0.05:
        data_path = "data/fm/7624_7624_7624_10_4_0.05.p"
    elif args.dataset == "fm" and args.noise == 0.1:
        data_path = "data/fm/7624_7624_7624_10_4_0.1.p"
    elif args.dataset == "fm" and args.noise == 0.15:
        data_path = "data/fm/7624_7624_7624_10_4_0.15.p"
    elif args.dataset == "fm" and args.noise == 0.2:
        data_path = "data/fm/7624_7624_7624_10_4_0.2.p"
    elif args.dataset == "fm" and args.noise == 0.25:
        data_path = "data/fm/7624_7624_7624_10_4_0.25.p"

    # ---------------------------Arxiv --------------------------------------
    elif args.dataset == "arxiv" and args.noise == 0.0:
        data_path = "data/arxiv/18772_18772_18772_10_4_0.p"
    elif args.dataset == "arxiv" and args.noise == 0.2:
        data_path = "data/arxiv/18772_18772_18772_10_4_0.2.p"

    elif args.dataset == "ppi" and args.noise == 0.2:
        data_path = "data/ppi_rn/1.p"

    elif args.dataset == "oregon" and args.noise == 0.2:
        data_path = "data/oregon/1.p"

    elif args.dataset == "ubuntu" and args.noise == 0.2:
        data_path = "data/ubuntu/1.p"
    # ----------------------------- SBM ------------------------------------
    elif args.dataset == "sbm" and args.noise == 0.0:
        data_path = "data/sbm/400_400_400_10_4_0.p"

    else:
        raise NotImplementedError

    print("data_path is {}".format(data_path))
    with open(data_path, "rb") as f:
        graph_st = pickle.load(f)
    # -------------------------- added two attributes for evaluation -------------------------
    if isinstance(graph_st.graph_s.w, np.ndarray):
        graph_st.ES = np.count_nonzero(graph_st.graph_s.w) // 2  # each undirected edge is counted once
    else:
        graph_st.ES = graph_st.graph_s.w.count_nonzero() // 2
    if isinstance(graph_st.graph_t.w, np.ndarray):
        graph_st.ET = np.count_nonzero(graph_st.graph_t.w) // 2
    else:
        graph_st.ET = graph_st.graph_t.w.count_nonzero() // 2
    dataset = data_path.split('/')[1]
    name = data_path.split('/')[-1][:-2]
    ns = graph_st.graph_s.n

    print("Matching undirected graphs using algorithm {}".format(alg))
    if alg == "regal":
        match_regal(graph_st, dataset=dataset, name=name, seed=seed, rounds=1)
    elif alg == "gwl":
        match_gwl(graph_st, dataset=dataset, name=name, seed=seed, ini=ini)
    elif alg == "hi_recur" or alg == "hi" or alg == "phgw":
        match_hi_recur(graph_st, dataset=dataset, name=name, seed=seed, ini=ini, n_par=n_par, par_level=par_level,
                       node_limit=node_limit, rank=rank, md_lr=md_lr, prune_lda=args.lda, prune=args.prune)
    elif alg == "pivot":
        match_pivot(graph_st, dataset=dataset, name=name, seed=seed, ini=ini)
    elif alg == "qegw":
        match_qegw(graph_st, dataset=dataset, name=name, seed=seed, rank=rank, md_lr=md_lr)
    elif alg == "qgw":
        match_qgw(graph_st, dataset=dataset, name=name, seed=seed, ini=ini, clu_num=n_par,
                  partition_level=par_level)
    elif alg == "sgwl":
        match_sgwl(graph_st, dataset=dataset, name=name, seed=seed, ini=ini, clu_num=n_par,
                   partition_level=par_level, node_limit=node_limit, md_lr=md_lr)
    else:
        raise NotImplementedError
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alg", help="The method to run", default="qgw")
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--no-prune', dest='feature', action='store_false')
    parser.add_argument("--ini", help="Initialization method", default="uni")
    parser.add_argument("-d", "--dataset", help="The method to run", default="arenas")
    parser.add_argument("-n", "--noise", help="Noise level", default=0.2, type=float)
    parser.add_argument("-s", "--seed", help="Random seed", default=1, type=int)
    parser.add_argument("-p", "--npar", help="number of clusters", default=2, type=int)
    parser.add_argument("-l", "--level", help="level of partitions", default=3, type=int)
    parser.add_argument("-r", "--rank", help="rank", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=100, type=float)
    parser.add_argument("--lda", help="prune lda", default=1.0, type=float)
    parser.add_argument("--limit", help="node limit", default=200, type=int)
    args = parser.parse_args()
    main(args)
