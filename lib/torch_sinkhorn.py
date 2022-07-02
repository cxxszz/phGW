import torch
from lib.matmul import mem_matmul, two_matmul
from lib.lp import cw_min
from lib.util import row_norm

prec = 1e-20
div_prec = 1e-16
log_prec = 1e-16


def fused_grad(As: torch.Tensor, At: torch.Tensor, trans: torch.Tensor,
               As1: torch.Tensor, As2T: torch.Tensor, At1: torch.Tensor, At2T: torch.Tensor,
               emb_s: torch.Tensor, emb_t: torch.Tensor, alpha: float):
    emb_s_norm, emb_t_norm = row_norm(emb_s), row_norm(emb_t)
    cor_s_emb = torch.matmul(emb_s_norm, emb_s_norm.T)
    cor_t_emb = torch.matmul(emb_t_norm, emb_t_norm.T)
    cor_s = alpha * cor_s_emb + (1 - alpha) * As
    cor_t = alpha * cor_t_emb + (1 - alpha) * At

    mu_s = torch.sum(trans, dim=1).unsqueeze_(1)
    mu_t = torch.sum(trans, dim=0).unsqueeze_(1)
    one_s = 0 * mu_s + 1
    one_t = 0 * mu_t + 1
    deg_terms = torch.matmul(torch.matmul(cor_s ** 2, mu_s), torch.t(one_t))
    deg_terms = deg_terms + torch.matmul(one_s, torch.matmul(torch.t(mu_t), torch.t(cor_t ** 2)))

    num1 = two_matmul(trans=trans, X1=As1, X2T=As2T, Y1=At1, Y2T=At2T)
    num2 = two_matmul(trans=trans, X1=As1, X2T=As2T, Y1=emb_t_norm, Y2T=emb_t_norm.T)
    num3 = two_matmul(trans=trans, X1=emb_s_norm, X2T=emb_s_norm.T, Y1=At1, Y2T=At2T)
    num4 = two_matmul(trans=trans, X1=emb_s_norm, X2T=emb_s_norm.T, Y1=emb_t_norm, Y2T=emb_t_norm.T)
    num = (1 - alpha) ** 2 * num1 + (1 - alpha) * alpha * (num2 + num3) + alpha ** 2 * num4
    return deg_terms - 2 * num


def peyre_expon(Bs: torch.Tensor, Bt: torch.Tensor, trans: torch.Tensor):
    mu_s = torch.sum(trans, dim=1).unsqueeze_(1)
    mu_t = torch.sum(trans, dim=0).unsqueeze_(1)
    one_s = 0 * mu_s + 1
    one_t = 0 * mu_t + 1
    deg_terms = torch.matmul(torch.matmul(Bs ** 2, mu_s), torch.t(one_t))
    deg_terms += torch.matmul(one_s, torch.matmul(torch.t(mu_t), torch.t(Bt ** 2)))

    tmp = mem_matmul(Bs, trans)
    num = mem_matmul(tmp, Bt)
    del tmp
    return deg_terms - 2 * num


def peyre_expon_low_rank(Bs: torch.Tensor, Bt: torch.Tensor, trans: torch.Tensor,
                         Bs1: torch.Tensor, Bs2T: torch.Tensor, Bt1: torch.Tensor, Bt2T: torch.Tensor):
    mu_s = torch.sum(trans, dim=1).unsqueeze_(1)
    mu_t = torch.sum(trans, dim=0).unsqueeze_(1)
    one_s = 0 * mu_s + 1
    one_t = 0 * mu_t + 1
    deg_terms = torch.matmul(torch.matmul(Bs ** 2, mu_s), torch.t(one_t))
    deg_terms += torch.matmul(one_s, torch.matmul(torch.t(mu_t), torch.t(Bt ** 2)))

    # tmp = torch.matmul(Bs2T, trans)
    # mid = torch.matmul(tmp, Bt1)
    # # del tmp
    # num = Bs1 @ mid @ Bt2T
    num = two_matmul(trans=trans, X1=Bs1, X2T=Bs2T, Y1=Bt1, Y2T=Bt2T)
    return deg_terms - 2 * num


def normalize(sim: torch.Tensor, mu_s: torch.Tensor, mu_t: torch.Tensor, n_iter=10):
    """

    :param sim: ns * nt
    :param mu_s: ns * 1
    :param mu_t: nt * 1
    :param n_iter:
    :return:
    """
    assert bool((sim >= 0).all())
    T = sim
    for _ in range(n_iter):
        # T = T / torch.sum(T, dim=1).unsqueeze_(1) * mu_s
        # T = T / torch.sum(T, dim=0) * mu_t.squeeze(1)
        T /= torch.sum(T, dim=1).unsqueeze_(1)
        T *= mu_s
        T /= torch.sum(T, dim=0)
        T *= mu_t.squeeze(1)
    return T


def peri_proj(trans, mu_s, mu_t, total_mass=0.9, n_iter=100):
    dtype = trans.dtype
    device = trans.device
    p = mu_s.squeeze(1)
    q = mu_t.squeeze(1)
    one_s = torch.ones(p.size(), dtype=dtype, device=device)
    one_t = torch.ones(q.size(), dtype=dtype, device=device)
    for _ in range(n_iter):
        # torch.diagflat() builds a diagonal matrix
        P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), one_s)
        # P_p = torch.diagflat(P_p_d)
        # trans = torch.matmul(P_p, trans)
        trans *= P_p_d.unsqueeze_(1)

        P_q_d = cw_min(q / (div_prec + torch.sum(trans, dim=0)), one_t)
        # P_q = torch.diagflat(P_q_d)
        # trans = torch.matmul(trans, P_q)
        trans *= P_q_d

        # trans = trans / torch.sum(trans) * total_mass
        # print("trans={}".format(trans))
        trans /= torch.sum(trans)
        trans *= total_mass
    # --------- ending with this projection may be more appropriate ---------
    P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), one_s)
    trans *= P_p_d.unsqueeze_(1)
    return trans


def sinkhorn(cost, mu_s, mu_t, beta=0.1, n_iter=400):
    """
    ||p||_1=||q||_1
    :param cost:
    :param p:
    :param q:
    :return:
    """
    assert torch.all(mu_s >= 0)
    assert torch.all(mu_t >= 0)
    assert torch.sum(mu_s) == torch.sum(mu_t)
    ns = mu_s.size(0)
    nt = mu_t.size(0)
    trans = torch.matmul(mu_s, torch.t(mu_t))
    a = mu_s.sum().repeat(ns, 1)
    a /= a.sum()
    b = 0
    p = mu_s.unsqueeze(1)
    q = mu_t.unsqueeze(1)
    kernel = torch.exp(-cost / beta) * trans
    for _ in range(n_iter):
        b = p / (torch.matmul(torch.t(kernel), a) + prec)
        a = q / (torch.matmul(kernel, b) + prec)
    tmp = a * kernel
    trans = torch.t(b * torch.t(tmp))
    del tmp
    # trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
    return trans


def t_sinkhorn():
    cost = torch.FloatTensor([[10, 10, 1],
                              [3, -2, 0],
                              [-4, 7, 8]])
    mu_s = torch.ones(3)
    mu_t = torch.ones(3)
    print(sinkhorn(cost, mu_s, mu_t))
