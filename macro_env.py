from collections import defaultdict
import numpy as np
import subprocess
import os
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import json
import torch
from torch_geometric.data import Data
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp


class AMoD:
    # initialization
    def __init__(self, scenario, cfg, beta=0.2):  # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = deepcopy(
            scenario)  # I changed it to deep copy so that the scenario input is not modified by env
        self.G = scenario.G  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)  # set of regions
        self.cfg = cfg
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)

        self.price = defaultdict(dict)  # price
        for i, j, t, d, p in scenario.tripAttr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.demandTime[i, j][t]] += d
        self.acc = defaultdict(dict)  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(dict)  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(
            dict)  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        self.tstep = scenario.tstep  # time step
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)

        self.N = len(self.region)  # total number of cells

        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.shadow_price = None
        self.gamma_sp = 0.99
        self.use_response_aware = False
        self.response_alpha = 0.0

    def matching(self, CPLEXPATH=None, PATH='', platform='linux'):
        # CPLEXPATH = 'None'
        if CPLEXPATH is None or str(CPLEXPATH).lower() in {"none", "null", ""} or self.shadow_price is not None:
            return self.matching_pulp()
        else:
            t = self.time
            demandAttr = [(i, j, self.demand[i, j][t], self.price[i, j][t]) for i, j in self.demand \
                          if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3]
            accTuple = [(n, self.acc[n][t + 1]) for n in self.acc]

            modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
            matchingPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/matching/' + PATH + '/'
            if not os.path.exists(matchingPath):
                os.makedirs(matchingPath)
            datafile = matchingPath + 'data_{}.dat'.format(t)
            resfile = matchingPath + 'res_{}.dat'.format(t)
            with open(datafile, 'w') as file:
                file.write('path="' + resfile + '";\r\n')
                file.write('demandAttr=' + mat2str(demandAttr) + ';\r\n')
                file.write('accInitTuple=' + mat2str(accTuple) + ';\r\n')
            modfile = modPath + 'matching.mod'

            my_env = os.environ.copy()
            if platform == 'mac':
                my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
            else:
                my_env["LD_LIBRARY_PATH"] = CPLEXPATH
            out_file = matchingPath + 'out_{}.dat'.format(t)
            with open(out_file, 'w') as output_f:
                subprocess.check_call([CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env)
            output_f.close()
            flow = defaultdict(float)
            with open(resfile, 'r', encoding="utf8") as file:
                for row in file:
                    item = row.replace('e)', ')').strip().strip(';').split('=')
                    if item[0] == 'flow':
                        values = item[1].strip(')]').strip('[(').split(')(')
                        for v in values:
                            if len(v) == 0:
                                continue
                            i, j, f = v.split(',')
                            flow[int(i), int(j)] = float(f)

            paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
            return paxAction

    def matching_pulp(self):
        # region, acc_init, demand, price, demand_edges
        t = self.time

        acc_init = {n: self.acc[n][t + 1] for n in self.acc}

        demand = {(i, j): self.demand[i, j][t] for i, j in self.demand if
                  t in self.demand[i, j] and self.demand[i, j][t] > 1e-3}

        price = {(i, j): self.price[i, j][t] for i, j in self.price if
                 t in self.demand[i, j] and self.demand[i, j][t] > 1e-3}

        demand_edges = [(i, j) for i, j in self.demand if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3]

        region = [n for n in acc_init]

        # Create a new PuLP model
        model = LpProblem("DemandFlowOptimization", LpMaximize)

        # Decision variables: flow on each edge
        flow = {
            (i, j): LpVariable(f"flow_{i}_{j}", lowBound=0, cat='Continuous')
            for (i, j) in demand_edges
        }

        lam = defaultdict(float, self.shadow_price) if self.shadow_price is not None else None
        if lam is None:
            model += lpSum(flow[(i, j)] * price[(i, j)] for (i, j) in demand_edges), "TotalRevenue"
        else:
            adjusted_reward = {}
            for (i, j) in demand_edges:
                tau = max(int(self.demandTime[i, j][t]), 1)
                adjusted_reward[(i, j)] = (
                        price[(i, j)]
                        - self.demandTime[i, j][t] * self.beta
                        - lam[i]
                        + (self.gamma_sp ** tau) * lam[j]
                )
            model += lpSum(
                flow[(i, j)] * adjusted_reward[(i, j)] for (i, j) in demand_edges
            ), "ShadowPriceAwareRevenue"

        # Supply constraints: total flow out of each region <= initial availability
        for k in region:
            model += lpSum(flow[(i, j)] for (i, j) in demand_edges if i == k) <= acc_init[k], f"Supply_Region_{k}"

        # Demand constraints: flow on each edge <= demand on that edge
        for (i, j) in demand_edges:
            model += flow[(i, j)] <= demand[(i, j)], f"Demand_Edge_{i}_{j}"

        # Solve the problem
        status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-10", "dualTol=1e-10", "mipGap=1e-10"]))
        # Output the results
        if LpStatus[status] == "Optimal":
            flow = {(i, j): value(flow[(i, j)]) for (i, j) in demand_edges}

            paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]

            return paxAction
        else:
            print(f"Optimization failed with status: {LpStatus[status]}")
            return None

    # pax step
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH='', platform='linux'):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.acc[i][t + 1] = self.acc[i][t]
        self.info['served_demand'] = 0  # initialize served demand
        self.info['revenue'] = 0
        self.info['profit'] = 0
        if paxAction is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            paxAction = self.matching(CPLEXPATH=CPLEXPATH, PATH=PATH, platform=platform)
        self.paxAction = paxAction
        # serving passengers
        test_rew = 0
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.demand or t not in self.demand[i, j] or self.paxAction[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            # assert paxAction[k] < self.acc[i][t+1] + 1e-3
            self.paxAction[k] = min(self.acc[i][t + 1], paxAction[k])
            self.servedDemand[i, j][t] = self.paxAction[k]
            self.paxFlow[i, j][t + self.demandTime[i, j][t]] = self.paxAction[k]
            self.info["operating_cost"] += self.demandTime[i, j][t] * self.beta * self.paxAction[k]
            self.acc[i][t + 1] -= self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i, j][t]
            self.dacc[j][t + self.demandTime[i, j][t]] += self.paxFlow[i, j][t + self.demandTime[i, j][t]]
            self.reward += self.paxAction[k] * (self.price[i, j][t] - self.demandTime[i, j][t] * self.beta)
            test_rew += self.paxAction[k] * (self.price[i, j][t])

            self.info['revenue'] += self.paxAction[k] * (self.price[i, j][t])
            self.info['profit'] += self.paxAction[k] * (self.price[i, j][t] - self.demandTime[i, j][t] * self.beta)

        self.obs = (self.acc, self.time, self.dacc,
                    self.demand)  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self.obs, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, rebAction):
        self.info['rebalancing_cost'] = 0
        self.info["operating_cost"] = 0
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.rebAction = rebAction
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t + 1], rebAction[k])
            self.rebFlow[i, j][t + self.rebTime[i, j][t]] = self.rebAction[k]
            self.acc[i][t + 1] -= self.rebAction[k]
            self.dacc[j][t + self.rebTime[i, j][t]] += self.rebFlow[i, j][t + self.rebTime[i, j][t]]
            self.info['rebalancing_cost'] += self.rebTime[i, j][t] * self.beta * self.rebAction[k]
            self.info["operating_cost"] += self.rebTime[i, j][t] * self.beta * self.rebAction[k]
            self.reward -= self.rebTime[i, j][t] * self.beta * self.rebAction[k]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t + 1] += self.paxFlow[i, j][
                    t]  # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion

        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand)  # use self.time to index the next time step
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
        done = (self.tf == self.time + 1)  # if the episode is completed

        return self.obs, self.reward, done, self.info

    def step(self, reb_action):
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        # Take action in environment
        if reb_action is None:
            reb_action = [0.0 for _ in self.edges]
        rew = 0
        info = {}
        info['rebalancing_cost'] = 0
        info['profit'] = 0
        info['served_demand'] = 0
        info['response_penalty'] = 0.0
        obs, rebreward, _, _ = self.reb_step(reb_action)
        info['rebalancing_cost'] = -rebreward
        rew += rebreward

        obs, paxreward, _, pax_info = self.pax_step(CPLEXPATH=self.cfg.cplexpath, PATH=self.cfg.directory)
        info['profit'] = paxreward
        info['served_demand'] = pax_info.get('served_demand', 0)
        rew += paxreward
        if self.use_response_aware and self.response_alpha > 0:
            unserved_t = 0.0
            for (i, j) in self.demand:
                unserved_t += max(
                    float(self.demand[i, j].get(self.time, 0.0))
                    - float(self.servedDemand[i, j].get(self.time, 0.0)),
                    0.0,
                )
            response_penalty = self.response_alpha * unserved_t
            rew -= response_penalty
            info['response_penalty'] = response_penalty
        done = (self.tf == self.time + 1)  # if the episode is completed
        return obs, rew, done, info

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        for i, j, t, d, p in tripAttr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)

        obs, paxreward, done, info = self.pax_step(CPLEXPATH=self.cfg.cplexpath, PATH=self.cfg.directory)

        self.reward = 0
        return obs, paxreward

    def reset_old(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        for i, j, t, d, p in tripAttr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)

        return self.obs


class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.2, json_file=None, json_hr=9,
                 json_tstep=2, varying_time=False, json_regions=None, prune=False):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1 * N2)
            self.G = self.G.to_directed()
            self.demandTime = dict()
            self.rebTime = dict()
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            for i, j in self.edges:
                self.demandTime[i, j] = defaultdict(
                    lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1)) * grid_travel_time)
                self.rebTime[i, j] = defaultdict(
                    lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1)) * grid_travel_time)

            for n in self.G.nodes:
                self.G.nodes[n]['accInit'] = int(ninit)
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            if demand_ratio == None or type(demand_ratio) == list:
                for i, j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i, j] = list(
                            np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio) - 1)),
                                      demand_ratio)) + [demand_ratio[-1]] * tf
                    else:
                        self.demand_ratio[i, j] = [1] * (tf + tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = list(
                            np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio[i, j]) - 1)),
                                      demand_ratio[i, j])) + [1] * tf
                    else:
                        self.demand_ratio[i, j] = list(
                            np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio['default']) - 1)),
                                      demand_ratio['default'])) + [1] * tf
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand() * 2 + 1) * (self.demandTime[i, j][0] + 1)
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand


        else:
            self.varying_time = varying_time
            self.is_json = True
            with open(json_file, "r") as file:
                data = json.load(file)
            self.tstep = json_tstep
            if prune:
                self.N1 = 2
                self.N2 = 2
            else:
                self.N1 = data["nlat"]
                self.N2 = data["nlon"]
            self.demand_input = defaultdict(dict)
            self.json_regions = json_regions

            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif 'region' in data:
                self.G = nx.complete_graph(data['region'])
            else:
                self.G = nx.complete_graph(self.N1 * self.N2)
            self.G = self.G.to_directed()
            self.p = defaultdict(dict)
            self.alpha = 0
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.json_start = json_hr * 60
            self.tf = tf
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(int)
                self.rebTime[i, j] = 1

            for item in data["demand"]:
                t, o, d, v, tt, p = item["time_stamp"], item["origin"], item["destination"], item["demand"], item[
                    "travel_time"], item["price"]
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(float), defaultdict(
                        float), defaultdict(float)

                self.demand_input[o, d][(t - self.json_start) // json_tstep] += v * demand_ratio
                self.p[o, d][(t - self.json_start) // json_tstep] += p * v * demand_ratio
                self.demandTime[o, d][(t - self.json_start) // json_tstep] += tt * v * demand_ratio / json_tstep

            for o, d in self.edges:
                for t in range(0, tf * 2):
                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(int(round(self.demandTime[o, d][t])), 1)
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0

            for item in data["rebTime"]:
                hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if varying_time:
                    t0 = int((hr * 60 - self.json_start) // json_tstep)
                    t1 = int((hr * 60 + 60 - self.json_start) // json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf + 1):
                            self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)

            if prune:
                for n in self.G.nodes:
                    self.G.nodes[n]['accInit'] = 10
            else:
                for item in data["totalAcc"]:
                    hr, acc = item["hour"], item["acc"]
                    if hr == json_hr + int(round(json_tstep / 2 * tf / 60)):
                        for n in self.G.nodes:
                            self.G.nodes[n]['accInit'] = int(acc / len(self.G))
            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = (np.random.rand(len(self.G)) * self.alpha * 2 + 1 - self.alpha)
            if type(self.demand_input) in [float, int, list, np.array]:

                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:
                    self.region_demand = region_rand * np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array([np.math.exp(-self.rebTime[i, j][0] * self.trip_length_preference) for j in J])
                    prob = prob / sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i, J[idx]] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = self.demand_input[i, j] if (i, j) in self.demand_input else \
                    self.demand_input['default']

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception("demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(self.static_demand[i, j] * self.demand_ratio[i, j][t])
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(2) + 1) * self.demandTime[i, j][t]
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr


class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01, cfg=None):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file

        # ---- Service-pressure encoder switches ----
        self.use_service_pressure_encoder = False
        self.use_multi_scale_pressure_encoder = False

        self.pressure_horizon = 6
        self.pressure_horizons = [2, 6, 12]
        self.pressure_scale = scale_factor
        self.pressure_clip = 20.0

        if cfg is not None:
            try:
                model_cfg = cfg.model
            except Exception:
                model_cfg = cfg

            self.use_service_pressure_encoder = bool(
                getattr(model_cfg, "use_service_pressure_encoder", False)
            )

            self.use_multi_scale_pressure_encoder = bool(
                getattr(model_cfg, "use_multi_scale_pressure_encoder", False)
            )

            self.pressure_horizon = int(
                getattr(model_cfg, "pressure_horizon", 6)
            )

            self.pressure_scale = float(
                getattr(model_cfg, "pressure_scale", scale_factor)
            )

            self.pressure_clip = float(
                getattr(model_cfg, "pressure_clip", 20.0)
            )

            raw_horizons = getattr(model_cfg, "pressure_horizons", [2, 6, 12])
            self.pressure_horizons = self._parse_pressure_horizons(raw_horizons)

        if self.use_multi_scale_pressure_encoder:
            pressure_dim = 6
        elif self.use_service_pressure_encoder:
            pressure_dim = 3
        else:
            pressure_dim = 0

        # feature dimension for automatic input-size handling
        self.feature_dim = 1 + 2 * self.T + pressure_dim

        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def _parse_pressure_horizons(self, value):
        """
        Supports:
        1) Hydra ListConfig: [2,6,12]
        2) Python list/tuple: [2,6,12]
        3) comma-separated string: "2,6,12"
        4) single int/string: 6
        """
        if value is None:
            return [2, 6, 12]

        if not isinstance(value, str):
            try:
                horizons = [int(v) for v in list(value)]
                return sorted(list(set([h for h in horizons if h > 0])))
            except TypeError:
                pass

        value = str(value).strip().strip("'").strip('"')

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]

        if "," in value:
            horizons = [int(v.strip()) for v in value.split(",") if v.strip()]
        else:
            horizons = [int(value)]

        horizons = sorted(list(set([h for h in horizons if h > 0])))

        if len(horizons) == 0:
            return [2, 6, 12]

        return horizons

    def _safe_pressure_ratio(self, numerator, denominator):
        ratio = float(numerator) / (float(denominator) + 1e-6)
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=self.pressure_clip, neginf=0.0)
        ratio = np.clip(ratio, 0.0, self.pressure_clip)
        return ratio * self.pressure_scale

    def _future_outgoing_demand(self, region_i, start_t, horizon):
        total = 0.0
        demand_source = getattr(self.env.scenario, "demand_input", None)
        if demand_source is None:
            demand_source = self.env.demand
        for h in range(horizon):
            tt = start_t + h
            if tt > self.env.tf:
                break
            for (o, d) in demand_source:
                if o != region_i:
                    continue
                total += float(demand_source[o, d].get(tt, 0.0))
        return total

    def _future_incoming_supply(self, region_i, start_t, horizon):
        total = 0.0
        for tt in range(start_t + 1, min(start_t + horizon + 1, self.env.tf + 1)):
            total += float(self.env.dacc[region_i].get(tt, 0.0))
        return total

    def build_service_pressure_features(self):
        """
        If use_multi_scale_pressure_encoder=True:
            Return [N, 6] multi-scale pressure features:
            1) current pressure
            2) short-horizon future pressure
            3) mid-horizon future pressure
            4) long-horizon future pressure
            5) current unmet-demand gap ratio
            6) incoming supply ratio

        Else if use_service_pressure_encoder=True:
            Return the original [N, 3] pressure features:
            1) current pressure
            2) future pressure
            3) current unmet-demand gap
        """
        t = self.env.time
        feats = []

        for i in self.env.region:
            current_demand_i = 0.0
            current_served_i = 0.0
            current_supply_i = float(self.env.acc[i].get(t + 1, 0.0))

            # Current outgoing demand
            for (o, d) in self.env.demand:
                if o != i:
                    continue
                current_demand_i += float(self.env.demand[o, d].get(t, 0.0))

            # Current served demand
            for (o, d) in self.env.servedDemand:
                if o != i:
                    continue
                current_served_i += float(self.env.servedDemand[o, d].get(t, 0.0))

            current_gap_i = max(current_demand_i - current_served_i, 0.0)

            # ------------------------------
            # New multi-scale encoder
            # ------------------------------
            if self.use_multi_scale_pressure_encoder:
                horizons = self.pressure_horizons

                # Ensure at least three horizons
                if len(horizons) == 1:
                    h_short, h_mid, h_long = horizons[0], horizons[0], horizons[0]
                elif len(horizons) == 2:
                    h_short, h_mid, h_long = horizons[0], horizons[1], horizons[1]
                else:
                    h_short, h_mid, h_long = horizons[0], horizons[1], horizons[2]

                future_demand_short = self._future_outgoing_demand(i, t, h_short)
                future_demand_mid = self._future_outgoing_demand(i, t, h_mid)
                future_demand_long = self._future_outgoing_demand(i, t, h_long)

                incoming_short = self._future_incoming_supply(i, t, h_short)
                incoming_mid = self._future_incoming_supply(i, t, h_mid)
                incoming_long = self._future_incoming_supply(i, t, h_long)

                current_pressure = self._safe_pressure_ratio(
                    current_demand_i,
                    1.0 + current_supply_i
                )

                future_pressure_short = self._safe_pressure_ratio(
                    future_demand_short,
                    1.0 + current_supply_i + incoming_short
                )

                future_pressure_mid = self._safe_pressure_ratio(
                    future_demand_mid,
                    1.0 + current_supply_i + incoming_mid
                )

                future_pressure_long = self._safe_pressure_ratio(
                    future_demand_long,
                    1.0 + current_supply_i + incoming_long
                )

                unmet_gap_ratio = self._safe_pressure_ratio(
                    current_gap_i,
                    1.0 + current_demand_i
                )

                incoming_supply_ratio = self._safe_pressure_ratio(
                    incoming_mid,
                    1.0 + current_supply_i
                )

                feats.append([
                    current_pressure,
                    future_pressure_short,
                    future_pressure_mid,
                    future_pressure_long,
                    unmet_gap_ratio,
                    incoming_supply_ratio,
                ])

            # ------------------------------
            # Original single-scale encoder
            # ------------------------------
            else:
                H = self.pressure_horizon
                future_demand_i = self._future_outgoing_demand(i, t, H)
                future_incoming_i = self._future_incoming_supply(i, t, H)

                current_pressure = self._safe_pressure_ratio(
                    current_demand_i,
                    1.0 + current_supply_i
                )

                future_pressure = self._safe_pressure_ratio(
                    future_demand_i,
                    1.0 + current_supply_i + future_incoming_i
                )

                current_gap = current_gap_i * self.pressure_scale
                current_gap = np.nan_to_num(current_gap, nan=0.0, posinf=self.pressure_clip, neginf=0.0)
                current_gap = np.clip(current_gap, 0.0, self.pressure_clip)

                feats.append([
                    current_pressure,
                    future_pressure,
                    current_gap,
                ])

        return torch.tensor(feats, dtype=torch.float32)

    def parse_obs(self, obs):
        x_base = torch.cat((
            torch.tensor([obs[0][n][self.env.time + 1] * self.s for n in self.env.region]).view(1, 1,
                                                                                                self.env.nregion).float(),

            torch.tensor([[(obs[0][n][self.env.time + 1] + self.env.dacc[n][t]) * self.s for n in self.env.region] \
                          for t in range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T,
                                                                                               self.env.nregion).float(),

            torch.tensor([[sum([(self.env.scenario.demand_input[i, j][t]) * (self.env.price[i, j][t]) * self.s \
                                for j in self.env.region]) for i in self.env.region] for t in
                          range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T,
                                                                                      self.env.nregion).float()),

            dim=1).squeeze(0).view(1 + self.T + self.T, self.env.nregion).T
        if self.use_service_pressure_encoder or self.use_multi_scale_pressure_encoder:
            pressure_feats = self.build_service_pressure_features()
            x = torch.cat((x_base, pressure_feats), dim=1)
        else:
            x = x_base

        if self.json_file is not None:
            edge_index = torch.vstack((torch.tensor([edge['i'] for edge in self.data["topology_graph"]]).view(1, -1),
                                       torch.tensor([edge['j'] for edge in self.data["topology_graph"]]).view(1,
                                                                                                              -1))).long()
        else:
            edge_index = torch.cat((torch.arange(self.env.nregion).view(1, self.env.nregion),
                                    torch.arange(self.env.nregion).view(1, self.env.nregion)), dim=0).long()
        data = Data(x, edge_index)
        return data
