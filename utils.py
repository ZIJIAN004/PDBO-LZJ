import random

import numpy as np


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def grb_solve(data, k=None):
    import gurobipy as gp
    from gurobipy import GRB

    n = data["num_vars"]
    model = gp.Model()

    if k is None:
        x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
        model.update()
        obj = gp.LinExpr()

        for idx, indice in enumerate(data["Q_indices"].T):
            i, j = indice[0], indice[1]
            obj += data["Q_values"][idx] * x[i] * x[j]
        for i in range(n):
            obj += data["c"][i] * x[i]

        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.Params.MIPFocus = 1
        model.optimize()
    else:
        q_matrix = data["Q_sparse"].todense()
        indices = data["Q_indices"].T
        x = model.addVars(n, k, vtype=GRB.BINARY, name="x")

        obj = gp.quicksum(
            0.5 * q_matrix[i, j] * (1 - gp.quicksum(x[i, c] * x[j, c] for c in range(k)))
            for (i, j) in indices
        )
        model.setObjective(obj, GRB.MAXIMIZE)

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, c] for c in range(k)) == 1)

        model.optimize()

    return model
