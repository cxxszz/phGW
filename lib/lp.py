import torch
from copy import deepcopy as dc


def cw_min(a, b):
    """
    component-wise minimum
    :param a: 1d FloatTensor
    :param b: 1d FloatTensor
    :return:
    """
    return torch.min(a, b)


div_prec = 1e-16
log_prec = 1e-16


def lp_relaxed_ot(C: torch.FloatTensor, T0: torch.FloatTensor, lr=1e-2, n_iter=10000, eps=1e-16, sk_iter=10, reg=0.01):
    """
    Projeted gradient descent for solving linear programming has awful performance
    min_T <C,T>,
        s.t. T>=0,
        and row_sum(T)<=1,
        and col_sum(T)<=1.
    This is a convex problem, so we can find an approximation of the solution via projected gradient descent,
    which can be more efficient than CPU-based simplex methods.
    :param C:
    :param T:
    :param lr: step size of the projected gradient descent
    :param n_iter: the maximal number of steps of the projected gradient descent
    :param eps: terminating condition
    :return:
    """

    T = dc(T0)
    for _ in range(n_iter):
        # T_new = T - lr * C
        T_new = (1 - reg) * T - lr * C
        T_new = torch.clamp(T_new, min=0)
        for _ in range(sk_iter):
            T_new = T_new / (div_prec + torch.sum(T_new, dim=0))
            # print("after normalizing each column, T_new={}".format(T_new))
            T_new = T_new / (div_prec + torch.sum(T_new, dim=1).unsqueeze(1))
            # print("after normalizing each row, T_new={}".format(T_new))
        rela_error = torch.sum((T_new - T) ** 2) / (div_prec + torch.sum(T ** 2))
        print("rela_error={}, loss={}".format(rela_error, torch.sum(C * T_new)))
        if rela_error <= eps:
            break
        else:
            T = T_new
    return T_new


def lp_dykstra(C: torch.FloatTensor, p: torch.FloatTensor, q: torch.FloatTensor, n_iter=10, reg=0.025):
    cost = C / (div_prec + torch.max(torch.abs(C)))
    trans = torch.exp(-cost / reg)

    for _ in range(n_iter):
        # torch.diagflat() builds a diagonal matrix
        P_p_d = cw_min(p / (div_prec + torch.sum(trans, dim=1)), p)
        P_p = torch.diagflat(P_p_d)
        trans = torch.matmul(P_p, trans)

        P_q_d = cw_min(q / (div_prec + torch.sum(trans, dim=0)), q)
        P_q = torch.diagflat(P_q_d)
        trans = torch.matmul(trans, P_q)
    return trans


if __name__ == '__main__':
    C = torch.FloatTensor([[-1, -0.1, 0],
                           [0, 0, 1]])
    # T = torch.ones(C.size()) / C.size(0)
    beta = 0.025
    device = torch.device("cpu")
    p = torch.ones(C.size(0)).to(device)
    q = torch.ones(C.size(1)).to(device)
    print(lp_dykstra(C=C, p=p, q=q))
