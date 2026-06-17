from collections import defaultdict
import subprocess
import os
import re
from tqdm import trange
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, lpSum, value
from src.algos.baseline_utils import (
    future_origin_scores,
    is_none_path,
    proportional_desired_acc,
    solve_current_matching_pulp,
    solve_rebalance_from_supply_pulp,
)
from src.misc.utils import mat2str
import numpy as np


class MPC:
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
        self.policy_name = kwargs.get('policy_name')
        self.T = kwargs.get("T")
        self.platform = None
        self.oracle = kwargs.get("oracle", True)

    def MPC_pulp_macro(self, env):
        """Exact macro MPC LP fallback when CPLEX/OPL is not configured.

        This mirrors src/cplex_mod/MPC.mod: it jointly optimizes passenger
        service and rebalancing flows over the look-ahead horizon, then applies
        only the first-step passenger and rebalancing actions.
        """
        t0 = int(env.time)
        horizon = max(1, min(int(self.T), int(env.tf - t0)))
        tf = t0 + horizon
        times = list(range(t0, tf))
        regions = list(env.region)
        edges = list(env.edges)

        def _get_nested(mapping, key, t, default=0.0):
            try:
                inner = mapping[key]
                if hasattr(inner, "get"):
                    return inner.get(t, default)
                return inner[t]
            except Exception:
                return default

        def _reb_time(i, j):
            return max(float(_get_nested(env.rebTime, (i, j), t0, 0.0)), 0.0)

        def _demand_time(i, j, tt):
            return max(float(_get_nested(env.demandTime, (i, j), tt, 0.0)), 0.0)

        demand_source = env.demand if self.oracle else env.scenario.demand_input
        demand = {}
        price = {}
        demand_time = {}
        for (i, j) in edges:
            for tt in times:
                raw_demand = float(_get_nested(demand_source, (i, j), tt, 0.0))
                demand[i, j, tt] = raw_demand if self.oracle else round(raw_demand)
                price[i, j, tt] = float(_get_nested(env.price, (i, j), tt, 0.0))
                demand_time[i, j, tt] = _demand_time(i, j, tt)

        model = LpProblem("MacroMPC", LpMaximize)
        acc = {
            (i, tt): LpVariable(f"acc_{i}_{tt}", lowBound=0, cat="Continuous")
            for i in regions
            for tt in range(t0, tf + 1)
        }
        demand_flow = {
            (i, j, tt): LpVariable(f"pax_{i}_{j}_{tt}", lowBound=0, cat="Continuous")
            for (i, j) in edges
            for tt in times
        }
        reb_flow = {
            (i, j, tt): LpVariable(f"reb_{i}_{j}_{tt}", lowBound=0, cat="Continuous")
            for (i, j) in edges
            for tt in times
        }

        model += (
            lpSum(
                demand_flow[i, j, tt] * price[i, j, tt]
                for (i, j) in edges
                for tt in times
            )
            - env.beta
            * lpSum(reb_flow[i, j, tt] * _reb_time(i, j) for (i, j) in edges for tt in times)
            - env.beta
            * lpSum(
                demand_flow[i, j, tt] * demand_time[i, j, tt]
                for (i, j) in edges
                for tt in times
            )
        )

        for i in regions:
            model += acc[i, t0] == float(env.acc[i].get(t0, 0.0))

        for tt in times:
            for i in regions:
                out_flow = lpSum(
                    demand_flow[i, j, tt] + reb_flow[i, j, tt]
                    for j in regions
                    if (i, j) in edges
                )
                pax_arrivals = lpSum(
                    demand_flow[o, d, dep_t]
                    for (o, d) in edges
                    for dep_t in times
                    if d == i and int(dep_t + demand_time[o, d, dep_t]) == tt
                )
                reb_arrivals = lpSum(
                    reb_flow[o, d, dep_t]
                    for (o, d) in edges
                    for dep_t in times
                    if d == i and int(dep_t + _reb_time(o, d)) == tt
                )
                model += (
                    acc[i, tt + 1]
                    == acc[i, tt]
                    - out_flow
                    + pax_arrivals
                    + reb_arrivals
                    + float(env.dacc[i].get(tt, 0.0))
                )
                model += out_flow <= acc[i, tt]

            for (i, j) in edges:
                model += demand_flow[i, j, tt] <= max(demand[i, j, tt], 0.0)
                if i == j:
                    # Rebalancing self-loops have no physical effect in env.reb_step.
                    model += reb_flow[i, j, tt] == 0.0

        status = model.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"],
            )
        )

        if LpStatus[status] == "Optimal":
            pax_action = [
                float(value(demand_flow[i, j, t0]) or 0.0)
                for (i, j) in env.edges
            ]
            reb_action = [
                float(value(reb_flow[i, j, t0]) or 0.0)
                for (i, j) in env.edges
            ]
            return pax_action, reb_action

        print(f"[WARN] Exact PuLP macro MPC failed with status: {LpStatus[status]}; using heuristic fallback.")
        return self.MPC_heuristic_macro(env)

    def MPC_heuristic_macro(self, env):
        """Small fallback used only if the exact PuLP MPC is infeasible."""
        supply = {n: float(env.acc[n].get(env.time, 0.0)) for n in env.region}
        pax_action, supply_after_pax = solve_current_matching_pulp(env, supply=supply)
        scores = future_origin_scores(env, horizon=self.T, start_offset=1, price_weight=True)
        desired_acc = proportional_desired_acc(
            env,
            scores,
            total_vehicles=sum(supply_after_pax.values()),
            fallback_acc=supply_after_pax,
        )
        reb_action = solve_rebalance_from_supply_pulp(env, supply_after_pax, desired_acc)
        return pax_action, reb_action
    
    def MPC_exact(self, env):
        if is_none_path(self.cplexpath):
            return self.MPC_pulp_macro(env)

        t = env.time

        if self.oracle:
            demandAttr = [
                (
                    i,
                    j,
                    tt,
                    env.demand[i, j][tt],
                    env.demandTime[i, j][tt],
                    env.price[i, j][tt],
                )
                for i, j in env.demand
                for tt in range(t, t + self.T)
                if env.demand[i, j][tt] > 1e-3
            ]
        else:
            demandAttr = [
                (
                    i,
                    j,
                    tt,
                    round(env.scenario.demand_input[i, j][tt]),
                    env.demandTime[i, j][tt],
                    env.price[i, j][tt],
                )
                for i, j in env.scenario.demand_input
                for tt in range(t, t + self.T)
                if env.scenario.demand_input[i, j][tt] > 1e-3
            ]
        accTuple = [(n, env.acc[n][t]) for n in env.acc]
        daccTuple = [
            (n, tt, env.dacc[n][tt])
            for n in env.acc
            for tt in range(t, t + self.T)
        ]
        edgeAttr = [(i, j, env.rebTime[i, j][t]) for i, j in env.edges]
        modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
        MPCPath = os.getcwd().replace("\\", "/") + "/saved_files/cplex_logs/" + self.directory + "/"
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + "data_{}.dat".format(t)
        resfile = MPCPath + "res_{}.dat".format(t)
        with open(datafile, "w") as file:
            file.write('path="' + resfile + '";\r\n')
            file.write("t0=" + str(t) + ";\r\n")
            file.write("T=" + str(self.T) + ";\r\n")
            file.write("beta=" + str(env.beta) + ";\r\n")
            file.write("demandAttr=" + mat2str(demandAttr) + ";\r\n")
            file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
            file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
            file.write("daccAttr=" + mat2str(daccTuple) + ";\r\n")

        modfile = modPath + "MPC.mod"
        my_env = os.environ.copy()
        if self.platform == None:
            my_env["LD_LIBRARY_PATH"] = self.cplexpath
        else:
            my_env["DYLD_LIBRARY_PATH"] = self.cplexpath
        out_file = MPCPath + "out_{}.dat".format(t)
        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [self.cplexpath + "oplrun", modfile, datafile],
                stdout=output_f,
                stderr=output_f,
                env=my_env,
            )
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f1, f2 = v.split(",")
                        f1 = float(re.sub("[^0-9e.-]", "", f1))
                        f2 = float(re.sub("[^0-9e.-]", "", f2))
                        paxFlow[int(i), int(j)] = float(f1)
                        rebFlow[int(i), int(j)] = float(f2)
        paxAction = [
            paxFlow[i, j] if (i, j) in paxFlow else 0 for i, j in env.edges
        ]
        rebAction = [
            rebFlow[i, j] if (i, j) in rebFlow else 0 for i, j in env.edges
        ]

        return paxAction, rebAction

    def test(self, num_episodes, env):
        """
        for testing MPC
        - num_episodes: An integer representing the number of episodes to run the test.
        - env: The AMoD environment object that contains various attributes and methods.
        """
        epochs = trange(num_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        seeds = list(range(env.cfg.seed, env.cfg.seed + num_episodes+1))
        inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            # Set seed for reproducibility across different policies
            np.random.seed(seeds[i_episode])
            inflow = np.zeros(env.nregion)
            done = False
            _ = env.reset_old()
            
            while not done:
                pax_action, reb_action = self.MPC_exact(env)

                _, paxreward, _, info = env.pax_step(paxAction=pax_action, CPLEXPATH=self.cplexpath)

                _, rebreward, done, info = env.reb_step(reb_action)

                rew = paxreward + rebreward
    
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]
                
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
        
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            inflows.append(inflow)
            epochs.set_description(f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}")
        return episode_reward, episode_served_demand, episode_rebalancing_cost, inflows
        
