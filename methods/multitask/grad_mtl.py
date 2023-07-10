import random
import numpy as np
import copy
from typing import List, Tuple
from scipy.optimize import minimize
import torch


def PCGrad(grads: List[Tuple[torch.Tensor]], reduction: str = "sum") -> torch.Tensor:
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = sum(
                [
                    torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                    for grad_i, grad_j in zip(g_i, g_j)
                ]
            )
            if g_i_g_j < 0:
                g_j_norm_square = (
                    torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                )
                for grad_i, grad_j in zip(g_i, g_j):
                    grad_i -= g_i_g_j * grad_j / g_j_norm_square

    merged_grad = [sum(g) for g in zip(*pc_grad)]
    if reduction == "mean":
        merged_grad = [g / len(grads) for g in merged_grad]

    return merged_grad


def CAGrad(grads, alpha=0.5, rescale=1):
    n_tasks = len(grads)
    grads = grads.t()

    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(n_tasks) / n_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
            + c
            * np.sqrt(x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def IMTL(grads_list):
    grads = {}
    norm_grads = {}

    for i, grad in enumerate(grads_list):

        norm_term = torch.norm(grad)

        grads[i] = grad
        norm_grads[i] = grad / norm_term

    G = torch.stack(tuple(v for v in grads.values()))
    D = (
        G[
            0,
        ]
        - G[
            1:,
        ]
    )

    U = torch.stack(tuple(v for v in norm_grads.values()))
    U = (
        U[
            0,
        ]
        - U[
            1:,
        ]
    )
    first_element = torch.matmul(
        G[
            0,
        ],
        U.t(),
    )
    try:
        second_element = torch.inverse(torch.matmul(D, U.t()))
    except:
        # workaround for cases where matrix is singular
        second_element = torch.inverse(
            torch.eye(len(grads_list) - 1, device=norm_term.device) * 1e-8
            + torch.matmul(D, U.t())
        )

    alpha_ = torch.matmul(first_element, second_element)
    alpha = torch.cat(
        (torch.tensor(1 - alpha_.sum(), device=norm_term.device).unsqueeze(-1), alpha_)
    )
    return sum([alpha[i] * grads[i] for i in range(len(grads_list))])


def AUGD(grads_list, unused_shit):

    scale_norm_grads1 = {}
    scale_norm_grads = {}
    grads = {}
    norm_grads = {}
    for i, grad in enumerate(grads_list):

        norm_term = torch.norm(grad)

        grads[i] = grad
        norm_grads[i] = grad / norm_term
    for i, g in enumerate(grads_list):
        if i > 0:
            scale_norm_grads1[i] = norm_grads[0]*torch.norm(grads[i])
            scale_norm_grads[i] = norm_grads[i]*torch.norm(grads[0])
    G = torch.stack(tuple(v for v in grads.values()))
    U = torch.stack(tuple(v for v in norm_grads.values()))
    U1 = torch.stack(tuple(v for v in scale_norm_grads1.values()))
    U2 = torch.stack(tuple(v for v in scale_norm_grads.values()))

    D = G[0, ] - G[1:, ]

    U = (U1-U2)
    first_element = torch.matmul(
        G[0, ], U.t(),
    )
    try:
        second_element = torch.inverse(torch.matmul(D, U.t()))
    except:
        # workaround for cases where matrix is singular
        second_element = torch.inverse(
            torch.eye(len(grads_list) - 1, device=norm_term.device) * 1e-8
            + torch.matmul(D, U.t())
        )

    alpha_ = torch.matmul(first_element, second_element)
    alpha = torch.cat(
        (torch.tensor(1 - alpha_.sum(), device=norm_term.device).unsqueeze(-1), alpha_)
    )
    new_grad =  sum([alpha[i] * grads[i] for i in range(len(grads_list))])
    new_grad = new_grad*(torch.norm(grads[0])/torch.dot(new_grad,norm_grads[0]))
    return new_grad


METHODS = dict(
    pcgrad=PCGrad,
    cagrad=CAGrad,
    augd=AUGD
)
