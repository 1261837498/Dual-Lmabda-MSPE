from collections import defaultdict

import numpy as np
import pulp
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value

from src.algos.reb_flow_solver import solveRebFlow


def is_none_path(path):
    """Return True when a Hydra/CLI CPLEX path value means "use PuLP"."""
    return path is None or str(path).lower() in {"none", "null", ""}


def _nested_get(mapping, key, t, default=0.0):
    try:
        inner = mapping[key]
        if hasattr(inner, "get"):
            return inner.get(t, default)
        return inner[t]
    except Exception:
        return default


def _trip_time(env, i, j, t):
    return max(float(_nested_get(env.demandTime, (i, j), t, 1.0)), 1.0)


def _reb_time(env, i, j, t):
    try:
        return max(float(env.rebTime[i, j][t]), 0.0)
    except Exception:
        return max(float(env.G.edges[i, j].get("time", 1.0)), 0.0)


def current_rebalance_supply(env):
    """Vehicles available for rebalancing after the current passenger step."""
    t = env.time
    return {n: float(env.acc[n].get(t + 1, 0.0)) for n in env.region}


def future_origin_scores(env, horizon=6, start_offset=1, price_weight=False):
    """Aggregate forecasted origin demand into one score per region.

    The macro simulator exposes a static forecast in scenario.demand_input.
    The heuristic baselines use these scores to build a target vehicle
    distribution, then call the common minimum-cost rebalancing solver.
    """
    t0 = env.time + start_offset
    horizon = max(int(horizon), 1)
    demand_source = getattr(env.scenario, "demand_input", env.demand)
    scores = {i: 0.0 for i in env.region}

    for (o, d) in demand_source:
        if o not in scores:
            continue
        for tt in range(t0, t0 + horizon):
            demand = float(_nested_get(demand_source, (o, d), tt, 0.0))
            if demand <= 0:
                continue
            if price_weight:
                price = float(_nested_get(getattr(env.scenario, "p", env.price), (o, d), tt, 0.0))
                profit = max(price - env.beta * _trip_time(env, o, d, tt), 0.0)
                scores[o] += demand * profit
            else:
                scores[o] += demand
    return scores


def proportional_desired_acc(env, scores, total_vehicles, fallback_acc=None):
    """Convert region scores into an integer desired-vehicle dictionary."""
    total_vehicles = int(round(max(float(total_vehicles), 0.0)))
    fallback_acc = fallback_acc or current_rebalance_supply(env)
    if total_vehicles <= 0:
        return {i: 0 for i in env.region}

    score_vec = np.array([max(float(scores.get(i, 0.0)), 0.0) for i in env.region], dtype=np.float64)
    if not np.isfinite(score_vec).all() or score_vec.sum() <= 1e-9:
        desired = {i: int(round(float(fallback_acc.get(i, 0.0)))) for i in env.region}
        diff = total_vehicles - sum(desired.values())
        if diff == 0:
            return desired
        weights = np.ones(len(env.region), dtype=np.float64) / max(len(env.region), 1)
    else:
        weights = score_vec / score_vec.sum()
        desired_float = weights * total_vehicles
        floors = np.floor(desired_float).astype(int)
        remainder = total_vehicles - int(floors.sum())
        order = np.argsort(-(desired_float - floors))
        for idx in order[:remainder]:
            floors[idx] += 1
        return {env.region[i]: int(floors[i]) for i in range(len(env.region))}

    desired_float = weights * total_vehicles
    floors = np.floor(desired_float).astype(int)
    remainder = total_vehicles - int(floors.sum())
    for idx in range(remainder):
        floors[idx % len(floors)] += 1
    return {env.region[i]: int(floors[i]) for i in range(len(env.region))}


def solve_rebalance_to_scores(env, directory, cplexpath, scores, total_vehicles=None):
    """Build desiredAcc from scores and solve the common rebalancing problem."""
    supply = current_rebalance_supply(env)
    if total_vehicles is None:
        total_vehicles = sum(supply.values())
    desired = proportional_desired_acc(env, scores, total_vehicles, fallback_acc=supply)
    reb_action = solveRebFlow(env, directory, desired, cplexpath)
    if reb_action is None:
        return [0.0 for _ in env.edges]
    return reb_action


def solve_current_matching_pulp(env, supply=None):
    """Solve current passenger matching with PuLP for the macro MPC fallback.

    Returns:
        pax_action: list aligned with env.edges.
        supply_after_pax: vehicles remaining at each origin before rebalancing.
    """
    t = env.time
    supply = supply or {n: float(env.acc[n].get(t, 0.0)) for n in env.region}
    demand_edges = [
        (i, j)
        for (i, j) in env.demand
        if float(_nested_get(env.demand, (i, j), t, 0.0)) > 1e-9
    ]
    if not demand_edges:
        return [0.0 for _ in env.edges], dict(supply)

    model = LpProblem("MacroMPCPassengerMatching", LpMaximize)
    flow = {(i, j): LpVariable(f"pax_{i}_{j}", lowBound=0, cat="Continuous") for (i, j) in demand_edges}
    model += lpSum(
        flow[(i, j)]
        * (
            float(_nested_get(env.price, (i, j), t, 0.0))
            - env.beta * _trip_time(env, i, j, t)
        )
        for (i, j) in demand_edges
    )

    for i in env.region:
        model += lpSum(flow[(o, d)] for (o, d) in demand_edges if o == i) <= supply.get(i, 0.0)
    for (i, j) in demand_edges:
        model += flow[(i, j)] <= float(_nested_get(env.demand, (i, j), t, 0.0))

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if LpStatus[status] != "Optimal":
        print(f"[WARN] Macro MPC matching failed with status: {LpStatus[status]}.")
        return [0.0 for _ in env.edges], dict(supply)

    pax = defaultdict(float)
    supply_after = dict(supply)
    for (i, j) in demand_edges:
        val = float(value(flow[(i, j)]) or 0.0)
        pax[i, j] = val
        supply_after[i] = max(supply_after.get(i, 0.0) - val, 0.0)
    return [pax[i, j] for i, j in env.edges], supply_after


def solve_rebalance_from_supply_pulp(env, supply, desired_acc):
    """Minimum-cost rebalancing from an explicit supply snapshot.

    This mirrors solveRebFlow_pulp but is usable before env.pax_step mutates
    env.acc[t + 1], which is exactly what the MPC fallback needs.
    """
    t = env.time
    graph_edges = [(i, j) for (i, j) in env.G.edges]
    if not graph_edges:
        return [0.0 for _ in env.edges]

    model = LpProblem("MacroMPCRebalancing", LpMinimize)
    flow = {(i, j): LpVariable(f"reb_{i}_{j}", lowBound=0, cat="Integer") for (i, j) in graph_edges}
    model += lpSum(flow[(i, j)] * _reb_time(env, i, j, t) for (i, j) in graph_edges)

    for k in env.region:
        out_k = lpSum(flow[(i, j)] for (i, j) in graph_edges if i == k)
        in_k = lpSum(flow[(i, j)] for (i, j) in graph_edges if j == k)
        model += out_k <= max(float(supply.get(k, 0.0)), 0.0)
        model += in_k - out_k >= int(round(desired_acc.get(k, 0))) - max(float(supply.get(k, 0.0)), 0.0)

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if LpStatus[status] != "Optimal":
        print(f"[WARN] Macro MPC rebalancing failed with status: {LpStatus[status]}.")
        return [0.0 for _ in env.edges]

    reb = defaultdict(float)
    for (i, j) in graph_edges:
        reb[i, j] = float(value(flow[(i, j)]) or 0.0)
    return [reb[i, j] for i, j in env.edges]
