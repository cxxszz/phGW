from data_io.real_data_homo import graph_pair
import torch
import torch.nn as nn
from lib.lp import cw_min
from lib.matmul import mem_matmul
from lib.torch_sinkhorn import peri_proj

div_prec = 1e-16
log_prec = 1e-16

sk_bound = 1e-30


def outer(a: torch.Tensor, b: torch.Tensor, device):
    m, n = a.size(0), b.size(0)
    res = torch.zeros(m, n).to(device)
    for i in range(m):
        res[i] = a[i] * b

    return res


def cor_io(graph_st: graph_pair, device=torch.device("cuda:0"), dtype=torch.float):
    """

    :param graph_st:
    :return: correlation matrices R^{ns*ns} and R^{nt*nt}
    """
    cor_s = torch.from_numpy(graph_st.graph_s.w)
    cor_t = torch.from_numpy(graph_st.graph_t.w)

    mu_s = torch.sum(cor_s, dim=1).unsqueeze_(1).type(dtype)
    mu_t = torch.sum(cor_t, dim=1).unsqueeze_(1).type(dtype)
    mu_s = mu_s.to(device)
    mu_t = mu_t.to(device)
    # mu_s += 1e-7
    # mu_t += 1e-7
    mu_s /= torch.sum(mu_s)
    mu_t /= torch.sum(mu_t)

    cor_s = cor_s.to(device).type(dtype)
    cor_t = cor_t.to(device).type(dtype)

    return cor_s, cor_t, mu_s, mu_t


def cor_io_double(graph_st: graph_pair, device=torch.device("cuda:0")):
    """

    :param graph_st:
    :return: correlation matrices R^{ns*ns} and R^{nt*nt}
    """
    cor_s = torch.from_numpy(graph_st.graph_s.w).to(device)  # w is double
    cor_t = torch.from_numpy(graph_st.graph_t.w).to(device)

    mu_s = torch.sum(cor_s, dim=1).unsqueeze_(1)
    mu_t = torch.sum(cor_t, dim=1).unsqueeze_(1)
    # mu_s += 1e-7
    # mu_t += 1e-7
    mu_s /= torch.sum(mu_s)
    mu_t /= torch.sum(mu_t)

    return cor_s, cor_t, mu_s, mu_t


def sinkhorn_iteration(cost, mu_s, mu_t, trans0, beta=1e-1, inner_iteration=2):
    ns = mu_s.size(0)
    nt = mu_t.size(0)

    a = mu_s.sum().repeat(ns, 1)
    # print("a={}".format(a))
    a /= a.sum()
    b = 0
    if trans0 is None:
        kernel = torch.exp(-cost / beta)
    else:
        kernel = torch.exp(-cost / beta) * trans0

    # for l in range(inner_iteration):
    #     b = mu_t / (torch.matmul(torch.t(kernel), a) + div_prec)
    #     a_new = mu_s / (torch.matmul(kernel, b) + div_prec)
    #     rela_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
    #     a = a_new
    #     if rela_error <= sk_bound:
    #         break
    # trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
    trans = peri_proj(trans=kernel, mu_s=mu_s, mu_t=mu_t, total_mass=1.0, n_iter=inner_iteration)
    return trans


def ext_sinkhorn_iteration(cost, mu_s, mu_t, trans0, beta=1e-1, total_mass=0.9, device=torch.device("cuda:0")):
    mu_s_ext = torch.cat((mu_s, torch.FloatTensor([1 - total_mass]).unsqueeze_(0).to(device)), dim=0)
    mu_t_ext = torch.cat((mu_t, torch.FloatTensor([1 - total_mass]).unsqueeze_(0).to(device)), dim=0)

    #############################################################################################################
    ## extending the cost matrix
    ns = cost.size(0)
    nt = cost.size(1)

    bins0 = torch.ones((ns, 1)).to(device)
    bins1 = torch.ones((1, nt)).to(device)
    bin = torch.FloatTensor([[float("inf")]]).to(device)

    cost_ext = torch.cat([torch.cat([cost, bins0], -1),
                          torch.cat([bins1, bin], -1)], 0)

    #############################################################################################################
    ## extending the transport plan
    trans0_bins0 = torch.clamp(mu_s.squeeze(1) - torch.sum(trans0, dim=1), min=0).unsqueeze_(1)
    trans0_bins1 = torch.clamp(mu_t.squeeze(1) - torch.sum(trans0, dim=0), min=0).unsqueeze_(0)
    trans0_bin = torch.FloatTensor([[0]]).to(device)
    trans0_ext = torch.cat([torch.cat([trans0, trans0_bins0], 1),
                            torch.cat([trans0_bins1, trans0_bin], 1)], 0)

    #############################################################################################################
    ## calculating the extended transport plan
    trans_ext = sinkhorn_iteration(cost=cost_ext, mu_s=mu_s_ext, mu_t=mu_t_ext, trans0=trans0_ext, beta=beta,
                                   inner_iteration=200)
    trans_new = trans_ext[0:-1, 0:-1]
    # print("torch.sum(trans_ext)=", torch.sum(trans_ext))
    # input()
    return trans_new


def entropic_wasserstein(cost, mu_s, mu_t, reg=1e-1, inner_iteration=2, device=torch.device("cuda:0")):
    ns = mu_s.size(0)
    nt = mu_t.size(0)

    a = mu_s.sum().repeat(ns, 1)
    # print("a={}".format(a))
    a /= a.sum()
    b = 0
    kernel = torch.exp(-cost / reg)
    del cost
    for l in range(inner_iteration):
        b = mu_t / (torch.matmul(torch.t(kernel), a) + div_prec)
        a_new = mu_s / (torch.matmul(kernel, b) + div_prec)
        rela_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
        a = a_new
        if rela_error <= sk_bound:
            break
    tmp = a * kernel
    trans = torch.t(b * torch.t(tmp))
    del tmp
    # trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
    return trans


def entropic_partial_wasserstein(cost, mu_s, mu_t, reg=1e-1, inner_iteration=2, total_mass=0.9,
                                 device=torch.device("cuda:0")):
    trans = torch.exp(-cost / reg).to(device)
    p = mu_s.squeeze(1)
    q = mu_t.squeeze(1)
    for _ in range(inner_iteration):
        # torch.diagflat() builds a diagonal matrix
        P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), torch.ones(p.size()).to(device))
        P_p = torch.diagflat(P_p_d)
        trans = torch.matmul(P_p, trans)

        P_q_d = cw_min(q / (div_prec + torch.sum(trans, dim=0)), torch.ones(q.size()).to(device))
        P_q = torch.diagflat(P_q_d)
        trans = torch.matmul(trans, P_q)

        trans = trans / torch.sum(trans) * total_mass
    return trans


class GromovWassersteinEmbedding(nn.Module):
    """
    Learning embeddings from Cosine similarity
    """

    def __init__(self, num1: int, num2: int, dim: int, cost_type: str = 'cosine', loss_type: str = 'L2', emb_s=None,
                 emb_t=None, device=torch.device("cuda:0")):
        super(GromovWassersteinEmbedding, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.dim = dim
        self.cost_type = cost_type
        self.loss_type = loss_type
        self.device = device
        emb1 = nn.Embedding(self.num1, self.dim)
        if emb_s is None:
            emb1.weight = nn.Parameter(
                torch.FloatTensor(self.num1, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        else:
            emb1.weight = nn.Parameter(emb_s)
        emb2 = nn.Embedding(self.num2, self.dim)
        if emb_t is None:
            emb2.weight = nn.Parameter(
                torch.FloatTensor(self.num2, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        else:
            emb2.weight = nn.Parameter(emb_t)
        self.emb_model = nn.ModuleList([emb1, emb2])

    def orthogonal(self, index, idx):
        embs = self.emb_model[idx](index)
        orth = mem_matmul(torch.t(embs), embs)
        orth -= torch.eye(embs.size(1)).to(self.device)
        return (orth ** 2).sum()

    def self_cost_mat(self, index, idx):
        embs = self.emb_model[idx](index)  # (batch_size, dim)
        if self.cost_type == 'cosine':
            # cosine similarity
            energy = torch.sqrt(torch.sum(embs ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cost = 1 - torch.exp(
                -5 * (1 - mem_matmul(embs, torch.t(embs)) / (
                        mem_matmul(energy, torch.t(energy)) + div_prec)))
        else:
            # Euclidean distance
            embs = mem_matmul(embs, torch.t(embs))  # (batch_size, batch_size)
            embs_diag = torch.diag(embs).view(-1, 1).repeat(1, embs.size(0))  # (batch_size, batch_size)
            cost = 1 - torch.exp(-(embs_diag + torch.t(embs_diag) - 2 * embs) / embs.size(1))
        return cost

    def mutual_cost_mat(self, index1, index2):
        embs1 = self.emb_model[0](index1)  # (batch_size1, dim)
        embs2 = self.emb_model[1](index2)  # (batch_size2, dim)
        if self.cost_type == 'cosine':
            # cosine similarity
            energy1 = torch.sqrt(torch.sum(embs1 ** 2, dim=1, keepdim=True))  # (batch_size1, 1)
            energy2 = torch.sqrt(torch.sum(embs2 ** 2, dim=1, keepdim=True))  # (batch_size2, 1)
            cost = 1 - torch.exp(
                -(1 - mem_matmul(embs1, torch.t(embs2)) / (mem_matmul(energy1, torch.t(energy2)) + div_prec)))
        else:
            # Euclidean distance
            embs = mem_matmul(embs1, torch.t(embs2))  # (batch_size1, batch_size2)
            # (batch_size1, batch_size2)
            embs_diag1 = torch.diag(mem_matmul(embs1, torch.t(embs1))).view(-1, 1).repeat(1, embs2.size(0))
            # (batch_size2, batch_size1)
            embs_diag2 = torch.diag(mem_matmul(embs2, torch.t(embs2))).view(-1, 1).repeat(1, embs1.size(0))
            cost = 1 - torch.exp(-(embs_diag1 + torch.t(embs_diag2) - 2 * embs) / embs1.size(1))
        return cost

    def tensor_times_mat(self, cost_s, cost_t, trans, mu_s, mu_t):
        if self.loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s ** 2, mu_s).repeat(1, trans.size(1))
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t ** 2)).repeat(trans.size(0), 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * mem_matmul(mem_matmul(cost_s, trans), torch.t(cost_t))
        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s * torch.log(cost_s + log_prec) - cost_s, mu_s).repeat(1, trans.size(1))
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t)).repeat(trans.size(0), 1)
            cost_st = f1_st + f2_st
            cost = cost_st - mem_matmul(mem_matmul(cost_s, trans), torch.t(torch.log(cost_t + log_prec)))
        return cost

    def similarity(self, cost_pred, cost_truth, mask=None):
        if mask is None:
            if self.loss_type == 'L2':
                loss = ((cost_pred - cost_truth) ** 2) * torch.exp(-cost_truth)
            else:
                loss = cost_pred * torch.log(cost_pred / (cost_truth + log_prec))
        else:
            if self.loss_type == 'L2':
                # print(mask.size())
                # print(cost_truth.size())
                # print(cost_pred.size())
                loss = mask.data * ((cost_pred - cost_truth) ** 2) * torch.exp(-cost_truth)
            else:
                loss = mask.data * (cost_pred * torch.log(cost_pred / (cost_truth + log_prec)))
        loss = loss.sum()
        return loss

    def forward(self, index1, index2, trans, mu_s, mu_t, cost1, cost2, prior=None, mask1=None, mask2=None, mask12=None):
        """
        cost_s = self.self_cost_mat(index1, 0)
        cost_t = self.self_cost_mat(index2, 1)
        cost_st = self.mutual_cost_mat(index1, index2)
        cost = self.tensor_times_mat(cost_s, cost_t, trans, mu_s, mu_t)
        d_gw = (cost * trans).sum()
        d_w = (cost_st * trans).sum()
        regularizer = self.similarity(cost_s, cost1, mask1) + self.similarity(cost_t, cost2, mask2)
        regularizer += self.orthogonal(index1, 0) + self.orthogonal(index2, 1)
        if prior is not None:
            regularizer += self.similarity(cost_st, prior, mask12)
        return d_gw, d_w, regularizer
        """
        cost_st = self.mutual_cost_mat(index1, index2)
        d_w = (cost_st * trans).sum()
        regularizer = self.orthogonal(index1, 0) + self.orthogonal(index2, 1)
        if prior is not None:
            regularizer += self.similarity(cost_st, prior, mask12)
        return 0, d_w, regularizer


class Embedding_learned(nn.Module):
    def __init__(self, ns: int, nt: int, dim: int, emb_s=None, emb_t=None, device=torch.device("cuda:0")):
        super(Embedding_learned, self).__init__()
        self.ns, self.nt = ns, nt
        self.dim = dim
        self.device = device
        if emb_s is None:
            self.emb_s = nn.Parameter(torch.FloatTensor(self.ns, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        else:
            self.emb_s = nn.Parameter(emb_s)
        if emb_t is None:
            self.emb_t = nn.Parameter(torch.FloatTensor(self.nt, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        else:
            self.emb_t = nn.Parameter(emb_t)

    def orthogonal(self):
        orth_s = mem_matmul(torch.t(self.emb_s), self.emb_s)
        orth_t = mem_matmul(torch.t(self.emb_t), self.emb_t)
        orth_s = orth_s - torch.eye(self.dim).to(self.device)
        orth_t = orth_t - torch.eye(self.dim).to(self.device)
        return torch.sum(orth_s ** 2) + torch.sum(orth_t ** 2)

    def similarity(self, cost_s_emb, cost_t_emb, cost_s, cost_t):
        loss_s = torch.sum((cost_s_emb - cost_s) ** 2)
        loss_t = torch.sum((cost_t_emb - cost_t) ** 2)
        return loss_s + loss_t

    def intra_cost_mat(self):
        energy_s = torch.norm(self.emb_s, dim=1)
        energy_s = energy_s.unsqueeze(1)
        cost_s_emb = 1 - mem_matmul(self.emb_s, self.emb_s.T) / (div_prec + mem_matmul(energy_s, energy_s.T))

        energy_t = torch.norm(self.emb_t, dim=1)
        energy_t = energy_t.unsqueeze(1)
        cost_t_emb = 1 - mem_matmul(self.emb_t, self.emb_t.T) / (div_prec + mem_matmul(energy_t, energy_t.T))
        return cost_s_emb, cost_t_emb

    def inter_cost_mat(self):
        energy_s = torch.norm(self.emb_s, dim=1)
        energy_s = energy_s.unsqueeze(1)
        energy_t = torch.norm(self.emb_t, dim=1)
        energy_t = energy_t.unsqueeze(1)
        inter_cost = 1 - mem_matmul(self.emb_s, self.emb_t.T) / (div_prec + mem_matmul(energy_s, energy_t.T))
        return inter_cost

    def forward(self, trans, mu_s, mu_t, cost_s, cost_t, alpha=1.0, return_cost_mat=False):
        ## calculating the GW discrepancy
        ## f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        ## cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        ## cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
        cost_s_emb, cost_t_emb = self.intra_cost_mat()
        cost1 = alpha * cost_s + (1 - alpha) * cost_s_emb
        cost2 = alpha * cost_t + (1 - alpha) * cost_t_emb
        f1_st = torch.matmul(cost1 ** 2, mu_s).repeat(1, self.nt)
        f2_st = torch.matmul(torch.t(mu_t), torch.t(cost2 ** 2)).repeat(self.ns, 1)
        gw_cost = f1_st + f2_st - 2 * mem_matmul(mem_matmul(cost1, trans), torch.t(cost2))
        d_gw = torch.sum(gw_cost * trans)

        ## calculating the Wasserstein distance
        inter_cost = self.inter_cost_mat()
        d_w = torch.sum(inter_cost * trans)

        reg = self.orthogonal()
        reg = reg + self.similarity(cost_s_emb=cost_s_emb, cost_t_emb=cost_t_emb, cost_s=cost_s, cost_t=cost_t)
        if return_cost_mat:
            return d_gw, d_w, reg, gw_cost, inter_cost
        else:
            return d_gw, d_w, reg
