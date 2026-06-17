import os
import subprocess
from collections import defaultdict

import pulp
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, lpSum

from src.misc.utils import mat2str


def _use_pulp(CPLEXPATH):
    return CPLEXPATH is None or str(CPLEXPATH).lower() in {"none", "null", ""}


def solveRebFlow(env, res_path, desiredAcc, CPLEXPATH, shadow_price=None, gamma=0.99, beta=None):
    """Solve the minimum-cost rebalancing flow.

    When shadow_price is provided, the objective becomes
    beta * travel_time + lambda_origin - gamma^travel_time * lambda_destination.
    The shadow-aware objective is solved with PuLP so no OPL model changes are
    required. If no shadow price is given and CPLEX is configured, the original
    CPLEX path is preserved.
    """
    if shadow_price is not None or _use_pulp(CPLEXPATH):
        return solveRebFlow_pulp(
            env,
            desiredAcc,
            shadow_price=shadow_price,
            gamma=gamma,
            beta=beta,
        )

    t = env.time
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

    modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
    OPTPath = os.getcwd().replace("\\", "/") + "/" + "saved_files/cplex_logs/rebalancing/" + res_path + "/"
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f"data_{t}.dat"
    resfile = OPTPath + f"res_{t}.dat"
    with open(datafile, "w") as file:
        file.write('path="' + resfile + '";\r\n')
        file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
        file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
        file.write("accRLTuple=" + mat2str(accRLTuple) + ";\r\n")
    modfile = modPath + "minRebDistRebOnly.mod"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = OPTPath + f"out_{t}.dat"
    with open(out_file, "w") as output_f:
        subprocess.check_call([CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env)

    flow = defaultdict(float)
    with open(resfile, "r", encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(";").split("=")
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, f = v.split(",")
                    flow[int(i), int(j)] = float(f)

    return [flow[i, j] for i, j in env.edges]


def solveRebFlow_pulp(env, desiredAcc, shadow_price=None, gamma=0.99, beta=None):
    t = env.time

    acc_init = {n: int(env.acc[n][t + 1]) for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}
    edges = [(i, j) for i, j in env.G.edges]
    region = [n for n in acc_init]
    time = {(i, j): env.G.edges[i, j]["time"] for i, j in edges}

    if beta is None:
        beta = env.beta

    model = LpProblem("RebalancingFlowMinimization", LpMinimize)
    rebFlow = {
        (i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat="Integer")
        for (i, j) in edges
    }

    if shadow_price is None:
        model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"
    else:
        lam = defaultdict(float, shadow_price)
        model += lpSum(
            rebFlow[(i, j)]
            * (
                beta * time[(i, j)]
                + lam[i]
                - (gamma ** max(int(time[(i, j)]), 1)) * lam[j]
            )
            for (i, j) in edges
        ), "ShadowPriceAwareRebalanceCost"

    for k in region:
        model += (
            lpSum(
                rebFlow[(j, i)] - rebFlow[(i, j)]
                for (i, j) in edges
                if j != i and i == k
            )
        ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

    for k in region:
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i == k) <= acc_init[k]
        ), f"RebalanceSupply_{k}"

    status = model.solve(
        pulp.PULP_CBC_CMD(
            msg=False,
            options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"],
        )
    )

    if LpStatus[status] == "Optimal":
        flow = defaultdict(float)
        for (i, j) in edges:
            flow[(i, j)] = rebFlow[(i, j)].varValue
        return [flow[i, j] for i, j in env.edges]

    print(f"Optimization failed with status: {LpStatus[status]}")
    return None
