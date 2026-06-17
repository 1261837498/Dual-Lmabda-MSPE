import os
import pickle
import random
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from tqdm import trange

from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.nets.actor import GNNActor, GNNActorLSTM
from src.nets.critic import GNNCritic, GNNCriticLSTM


class PairData(Data):
    """Store two graph states in one transition object."""

    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        lambda_dispatch=None,
        lambda_reb=None,
        heur_lambda_dispatch=None,
        heur_lambda_reb=None,
        context_s=None,
        context_t=None,
        edge_index_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.lambda_dispatch = lambda_dispatch
        self.lambda_reb = lambda_reb
        self.heur_lambda_dispatch = heur_lambda_dispatch
        self.heur_lambda_reb = heur_lambda_reb
        self.context_s = context_s
        self.context_t = context_t
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """Replay buffer for SAC agents."""

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(
            self,
            data1,
            action,
            lambda_dispatch,
            lambda_reb,
            reward,
            data2,
            heur_lambda_dispatch=None,
            heur_lambda_reb=None,
            context_s=None,
            context_t=None,
    ):
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        zero_lambda = torch.zeros_like(action_tensor)
        lambda_dispatch_tensor = (
            zero_lambda if lambda_dispatch is None else torch.as_tensor(lambda_dispatch, dtype=torch.float32)
        )
        lambda_reb_tensor = zero_lambda if lambda_reb is None else torch.as_tensor(lambda_reb, dtype=torch.float32)
        heur_dispatch_tensor = (
            zero_lambda
            if heur_lambda_dispatch is None
            else torch.as_tensor(heur_lambda_dispatch, dtype=torch.float32)
        )
        heur_reb_tensor = (
            zero_lambda if heur_lambda_reb is None else torch.as_tensor(heur_lambda_reb, dtype=torch.float32)
        )
        context_s_tensor = (
            None if context_s is None else torch.as_tensor(context_s, dtype=torch.float32)
        )
        context_t_tensor = (
            None if context_t is None else torch.as_tensor(context_t, dtype=torch.float32)
        )
        self.data_list.append(
            PairData(
                data1.edge_index,
                data1.x,
                torch.as_tensor(reward, dtype=torch.float32),
                action_tensor,
                lambda_dispatch_tensor,
                lambda_reb_tensor,
                heur_dispatch_tensor,
                heur_reb_tensor,
                context_s_tensor,
                context_t_tensor,
                data2.edge_index,
                data2.x,
            )
        )
        self.rewards.append(reward)

    def create_dataset(self, edge_index, memory_path, rew_scale, size=60000):
        with open(f"data/{memory_path}.pkl", "rb") as file:
            data = pickle.load(file)

        state_batch = data["state"]
        action_batch = data["action"]
        reward_batch = rew_scale * data["reward"]
        next_state_batch = data["next_state"]

        max_size = min(len(state_batch), size)
        for i in range(max_size):
            action_tensor = torch.as_tensor(action_batch[i], dtype=torch.float32)
            zero_lambda = torch.zeros_like(action_tensor)
            self.data_list.append(
                PairData(
                    edge_index,
                    torch.as_tensor(state_batch[i], dtype=torch.float32),
                    torch.as_tensor(reward_batch[i], dtype=torch.float32),
                    action_tensor,
                    zero_lambda,
                    zero_lambda.clone(),
                    zero_lambda.clone(),
                    zero_lambda.clone(),
                    edge_index,
                    torch.as_tensor(next_state_batch[i], dtype=torch.float32),
                )
            )

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32):
        data = random.sample(self.data_list, batch_size)
        return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(self.device)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


class SAC(nn.Module):
    """Soft Actor-Critic for AMoD rebalancing with optional dual shadow prices."""

    def __init__(self, env, input_size, cfg, parser, device=torch.device("cpu")):
        super().__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item()
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.act_dim = env.nregion
        self.nodes = env.nregion
        self.parser = parser

        self.alpha = cfg.alpha
        self.polyak = 0.995
        self.BATCH_SIZE = cfg.batch_size
        self.p_lr = cfg.p_lr
        self.q_lr = cfg.q_lr
        self.gamma = 0.99
        self.use_automatic_entropy_tuning = cfg.auto_entropy
        self.clip = cfg.clip
        self.use_LSTM = cfg.use_LSTM

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.agent_name = cfg.agent_name
        self.step = 0
        self.wandb = None

        self.use_shadow_price = bool(getattr(cfg, "use_shadow_price", False))
        self.use_learned_shadow = bool(getattr(cfg, "use_learned_shadow", False))
        self.use_sp_dispatch = bool(getattr(cfg, "use_sp_dispatch", False))
        self.use_sp_reb = bool(getattr(cfg, "use_sp_reb", False))
        self.single_lambda_mode = bool(getattr(cfg, "single_lambda_mode", False))

        self.lambda_scale_dispatch = float(getattr(cfg, "lambda_scale_dispatch", 2.0))
        self.lambda_scale_reb = float(getattr(cfg, "lambda_scale_reb", 2.0))
        self.lambda_reg_dispatch = float(getattr(cfg, "lambda_reg_dispatch", 0.0))
        self.lambda_reg_reb = float(getattr(cfg, "lambda_reg_reb", 0.0))
        self.sp_gamma_dispatch = float(getattr(cfg, "sp_gamma_dispatch", self.gamma))
        self.sp_gamma_reb = float(getattr(cfg, "sp_gamma_reb", self.gamma))

        self.heuristic_horizon = int(getattr(cfg, "heuristic_horizon", 6))
        self.heuristic_eps = 1e-6
        self.use_teacher_distill = bool(getattr(cfg, "use_teacher_distill", False))
        self.distill_coef_dispatch = float(getattr(cfg, "distill_coef_dispatch", 0.05))
        self.distill_coef_reb = float(getattr(cfg, "distill_coef_reb", 0.05))

        self.use_response_aware = bool(getattr(cfg, "use_response_aware", False))
        self.response_alpha = float(getattr(cfg, "response_alpha", 0.0))
        self.env.use_response_aware = self.use_response_aware
        self.env.response_alpha = self.response_alpha

        self.use_regime_aware = bool(getattr(cfg, "use_regime_aware", False))
        self.context_dim = int(getattr(cfg, "context_dim", 6))
        self.context_input_dim = self.context_dim if self.use_regime_aware else 0

        self.replay_buffer = ReplayData(device=device)

        if self.use_LSTM:
            if self.use_shadow_price and self.use_learned_shadow:
                raise NotImplementedError("Dual-lambda SAC currently supports use_LSTM=false only.")
            self.actor = GNNActorLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic1 = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic2 = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic1_target = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic2_target = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
        else:
            self.actor = GNNActor(
                input_size,
                hidden_size=cfg.hidden_size,
                act_dim=self.env.nregion,
                lambda_scale_dispatch=self.lambda_scale_dispatch,
                lambda_scale_reb=self.lambda_scale_reb,
                context_dim=self.context_dim if self.use_regime_aware else 0,
                safe_actor=getattr(cfg, "safe_actor", False),
            )
            self.critic1 = GNNCritic(
                self.input_size,
                self.hidden_size,
                act_dim=self.act_dim,
                context_dim=self.context_input_dim,
            )
            self.critic2 = GNNCritic(
                self.input_size,
                self.hidden_size,
                act_dim=self.act_dim,
                context_dim=self.context_input_dim,
            )
            self.critic1_target = GNNCritic(
                self.input_size,
                self.hidden_size,
                act_dim=self.act_dim,
                context_dim=self.context_input_dim,
            )
            self.critic2_target = GNNCritic(
                self.input_size,
                self.hidden_size,
                act_dim=self.act_dim,
                context_dim=self.context_input_dim,
            )

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()
        self.saved_actions = []
        self.rewards = []

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -float(self.act_dim)
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=1e-3)

        if hasattr(cfg, "min_q_weight"):
            self.min_q_weight = cfg.min_q_weight
        if hasattr(cfg, "temp"):
            self.temp = cfg.temp
        if hasattr(cfg, "num_random"):
            self.num_random = cfg.num_random

        self.to(self.device)

    def _context_tensor(self, context, batch_size=None):
        if not self.use_regime_aware:
            return None

        if context is None:
            batch_size = 1 if batch_size is None else batch_size
            return torch.zeros(batch_size, self.context_dim, device=self.device)

        context = torch.as_tensor(context, dtype=torch.float32, device=self.device)

        if context.dim() == 1:
            context = context.unsqueeze(0)

        if context.size(-1) < self.context_dim:
            pad = torch.zeros(
                context.size(0),
                self.context_dim - context.size(-1),
                dtype=context.dtype,
                device=context.device,
            )
            context = torch.cat([context, pad], dim=-1)

        if context.size(-1) > self.context_dim:
            context = context[:, : self.context_dim]
        context = torch.nan_to_num(context, nan=0.0, posinf=10.0, neginf=-10.0)
        context = torch.clamp(context, min=-10.0, max=10.0)
        return context

    def _env_end_time(self, env):
        return int(getattr(env, "tf", getattr(env, "duration", 0)))

    def _get_trip_time(self, env, i, j, tt):
        if hasattr(env, "demandTime"):
            try:
                return float(env.demandTime[i, j][tt])
            except Exception:
                pass
        if hasattr(env, "demand_time"):
            try:
                return float(env.demand_time[i, j][tt])
            except Exception:
                pass
        try:
            return float(env.rebTime[i, j][tt])
        except Exception:
            pass
        try:
            return float(env.G.edges[i, j]["time"])
        except Exception:
            return 1.0

    def build_regime_context(self, env):
        t = env.time
        end_time = max(self._env_end_time(env), 1)
        H = min(self.heuristic_horizon, max(end_time - t, 1))
        total_supply = sum(float(env.acc[i].get(t + 1, 0.0)) for i in env.region)
        total_demand = 0.0
        for h in range(H):
            tt = t + h
            for (o, d) in env.demand:
                total_demand += float(env.demand[o, d].get(tt, 0.0))
        avg_reb_time = 0.0
        cnt = 0
        for (i, j) in env.G.edges:
            avg_reb_time += float(env.rebTime[i, j].get(t, env.G.edges[i, j].get("time", 1)))
            cnt += 1
        avg_reb_time /= max(cnt, 1)
        t_norm = float(t / end_time)
        n_region = max(len(env.region), 1)

        demand_per_region = total_demand / (n_region + 1e-6)
        supply_per_region = total_supply / (n_region + 1e-6)
        demand_supply_ratio = total_demand / (total_supply + 1e-6)

        # Stable normalization.
        # log1p compresses large demand/supply values.
        demand_feat = np.log1p(max(demand_per_region, 0.0)) / 5.0
        supply_feat = np.log1p(max(supply_per_region, 0.0)) / 5.0
        ratio_feat = np.clip(demand_supply_ratio, 0.0, 10.0) / 10.0
        reb_time_feat = np.clip(avg_reb_time, 0.0, 60.0) / 60.0

        dynamic_context = np.array(
            [
                np.sin(2 * np.pi * t_norm),
                np.cos(2 * np.pi * t_norm),
                demand_feat,
                supply_feat,
                ratio_feat,
                reb_time_feat,
            ],
            dtype=np.float32,
        )

        scenario_context = self._get_scenario_context(env)

        context = np.concatenate([dynamic_context, scenario_context], axis=0)

        context = np.nan_to_num(context, nan=0.0, posinf=10.0, neginf=-10.0)
        context = np.clip(context, -10.0, 10.0)

        if len(context) < self.context_dim:
            context = np.concatenate(
                [context, np.zeros(self.context_dim - len(context), dtype=np.float32)],
                axis=0,
            )

        if len(context) > self.context_dim:
            context = context[: self.context_dim]

        return context.astype(np.float32)

    def _get_scenario_context(self, env):
        ctx = getattr(env, "scenario_context", {}) or {}
        scenario_file = str(getattr(env, "scenario_file", "")).lower()
        scenario_type = str(ctx.get("scenario_type", ctx.get("type", "normal"))).lower()

        # Fallback: infer from file name
        if scenario_type == "normal":
            if "peak" in scenario_file:
                scenario_type = "peak"
            elif "rain" in scenario_file:
                scenario_type = "rain"
            elif "event" in scenario_file:
                scenario_type = "event"
            elif "incident" in scenario_file:
                scenario_type = "incident"

        demand_scale = float(ctx.get("demand_scale", 1.0))
        travel_time_scale = float(ctx.get("travel_time_scale", ctx.get("reb_time_scale", 1.0)))
        event_level = float(ctx.get("event_level", 0.0))
        weather_level = float(ctx.get("weather_level", 0.0))
        incident_level = float(ctx.get("incident_level", 0.0))

        if scenario_type == "peak":
            demand_scale = max(demand_scale, 1.5)
        elif scenario_type == "rain":
            demand_scale = max(demand_scale, 1.2)
            travel_time_scale = max(travel_time_scale, 1.3)
            weather_level = max(weather_level, 1.0)
        elif scenario_type == "event":
            event_level = max(event_level, 1.0)
        elif scenario_type == "incident":
            travel_time_scale = max(travel_time_scale, 1.5)
            incident_level = max(incident_level, 1.0)

        affected_regions = ctx.get("affected_regions", [])
        if affected_regions is None:
            affected_regions = []
        affected_region_ratio = len(affected_regions) / max(len(env.region), 1)

        return np.array(
            [
                demand_scale,
                travel_time_scale,
                event_level,
                weather_level,
                incident_level,
                affected_region_ratio,
            ],
            dtype=np.float32,
        )

    def normalize_region_scores(self, raw_scores, scale):
        vals = np.array([raw_scores[r] for r in raw_scores], dtype=np.float32)
        if len(vals) == 0:
            return {r: 0.0 for r in raw_scores}
        std = float(vals.std())
        if std < self.heuristic_eps:
            return {r: 0.0 for r in raw_scores}
        mean = float(vals.mean())
        return {
            r: float(scale * np.tanh((raw_scores[r] - mean) / (std + self.heuristic_eps)))
            for r in raw_scores
        }

    def build_heuristic_lambda(self, env, H=None):
        H = self.heuristic_horizon if H is None else H
        t = env.time
        end_time = self._env_end_time(env)
        raw_dispatch = {}
        raw_reb = {}
        for i in env.region:
            current_profit = 0.0
            future_profit = 0.0
            current_supply = float(env.acc[i].get(t + 1, 0.0))
            future_supply = current_supply
            for h in range(H):
                tt = t + h
                if tt > end_time:
                    break
                for (o, d) in env.demand:
                    if o != i:
                        continue
                    dem = float(env.demand[o, d].get(tt, 0.0))
                    price = float(env.price[o, d].get(tt, 0.0))
                    trip_time = self._get_trip_time(env, o, d, tt)
                    unit_profit = max(price - env.beta * trip_time, 0.0)
                    if h == 0:
                        current_profit += dem * unit_profit
                    future_profit += dem * unit_profit
            for tt in range(t + 1, min(t + H + 1, end_time + 1)):
                future_supply += float(env.dacc[i].get(tt, 0.0))
            raw_dispatch[i] = current_profit / (1.0 + current_supply)
            raw_reb[i] = future_profit / (1.0 + future_supply)
        return (
            self.normalize_region_scores(raw_dispatch, self.lambda_scale_dispatch),
            self.normalize_region_scores(raw_reb, self.lambda_scale_reb),
        )

    def select_action(self, data, context=None, deterministic=False, return_lambda=False):
        with torch.no_grad():
            ctx = self._context_tensor(context)
            if self.use_LSTM:
                action, _ = self.actor(data.x, data.edge_index, deterministic)
                action = action.squeeze(-1).detach().cpu().numpy()[0]
                if return_lambda:
                    zeros = [0.0] * self.nodes
                    return list(action), zeros, zeros
                return list(action)

            action, _, lambda_dispatch, lambda_reb = self.actor(
                data.x,
                data.edge_index,
                context=ctx,
                deterministic=deterministic,
                return_lambda=True,
            )
        action = action.detach().cpu().numpy()[0]
        lambda_dispatch = lambda_dispatch.detach().cpu().numpy()[0]
        lambda_reb = lambda_reb.detach().cpu().numpy()[0]
        if self.single_lambda_mode:
            lambda_reb = lambda_dispatch.copy()
        if return_lambda:
            return list(action), list(lambda_dispatch), list(lambda_reb)
        return list(action)

    def compute_loss_q(self, data, conservative=False):
        state_batch = data.x_s
        edge_index = data.edge_index_s
        next_state_batch = data.x_t
        edge_index2 = data.edge_index_t
        reward_batch = data.reward
        action_batch = data.action.reshape(-1, self.nodes)
        lambda_dispatch_batch = data.lambda_dispatch.reshape(-1, self.nodes)
        lambda_reb_batch = data.lambda_reb.reshape(-1, self.nodes)
        if self.single_lambda_mode:
            lambda_reb_batch = lambda_dispatch_batch.clone()

        context_s_batch = None
        context_t_batch = None

        if self.use_regime_aware and hasattr(data, "context_s") and data.context_s is not None:
            if data.context_s.numel() > 0:
                context_s_batch = data.context_s.reshape(-1, self.context_dim).to(self.device)

        if self.use_regime_aware and hasattr(data, "context_t") and data.context_t is not None:
            if data.context_t.numel() > 0:
                context_t_batch = data.context_t.reshape(-1, self.context_dim).to(self.device)

        q1 = self.critic1(
            state_batch,
            edge_index,
            action_batch,
            lambda_dispatch_batch,
            lambda_reb_batch,
            context=context_s_batch,
        )
        q2 = self.critic2(
            state_batch,
            edge_index,
            action_batch,
            lambda_dispatch_batch,
            lambda_reb_batch,
            context=context_s_batch,
        )

        if self.wandb is not None:
            self.wandb.log({"Q1": q1.mean().item()})

        with torch.no_grad():
            if self.use_LSTM:
                a2, logp_a2 = self.actor(next_state_batch, edge_index2)
                lam_dispatch_2 = torch.zeros_like(a2)
                lam_reb_2 = torch.zeros_like(a2)
            else:
                a2, logp_a2, lam_dispatch_2, lam_reb_2 = self.actor(
                    next_state_batch,
                    edge_index2,
                    context=context_t_batch,
                    return_lambda=True,
                )
            if self.single_lambda_mode:
                lam_reb_2 = lam_dispatch_2.clone()
            q1_pi_targ = self.critic1_target(
                next_state_batch,
                edge_index2,
                a2,
                lam_dispatch_2,
                lam_reb_2,
                context=context_t_batch,
            )
            q2_pi_targ = self.critic2_target(
                next_state_batch,
                edge_index2,
                a2,
                lam_dispatch_2,
                lam_reb_2,
                context=context_t_batch,
            )
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        return F.mse_loss(q1, backup), F.mse_loss(q2, backup)

    def compute_loss_pi(self, data):
        state_batch = data.x_s
        edge_index = data.edge_index_s
        context_s_batch = None

        if self.use_regime_aware and hasattr(data, "context_s") and data.context_s is not None:
            if data.context_s.numel() > 0:
                context_s_batch = data.context_s.reshape(-1, self.context_dim).to(self.device)

        if self.use_LSTM:
            actions, logp_a = self.actor(state_batch, edge_index)
            lambda_dispatch_pi = torch.zeros_like(actions)
            lambda_reb_pi = torch.zeros_like(actions)
        else:
            actions, logp_a, lambda_dispatch_pi, lambda_reb_pi = self.actor(
                state_batch,
                edge_index,
                context=context_s_batch,
                return_lambda=True,
            )
        if self.single_lambda_mode:
            lambda_reb_pi = lambda_dispatch_pi.clone()

        q1_a = self.critic1(
            state_batch,
            edge_index,
            actions,
            lambda_dispatch_pi,
            lambda_reb_pi,
            context=context_s_batch,
        )
        q2_a = self.critic2(
            state_batch,
            edge_index,
            actions,
            lambda_dispatch_pi,
            lambda_reb_pi,
            context=context_s_batch,
        )
        q_a = torch.min(q1_a, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (logp_a + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        reg_dispatch = self.lambda_reg_dispatch * (lambda_dispatch_pi**2).mean()
        reg_reb = 0.0 if self.single_lambda_mode else self.lambda_reg_reb * (lambda_reb_pi**2).mean()

        distill_dispatch = torch.tensor(0.0, device=state_batch.device)
        distill_reb = torch.tensor(0.0, device=state_batch.device)
        if self.use_teacher_distill and hasattr(data, "heur_lambda_dispatch"):
            heur_dispatch = data.heur_lambda_dispatch.reshape(-1, self.nodes)
            heur_reb = data.heur_lambda_reb.reshape(-1, self.nodes)
            distill_dispatch = self.distill_coef_dispatch * F.mse_loss(lambda_dispatch_pi, heur_dispatch)
            if not self.single_lambda_mode:
                distill_reb = self.distill_coef_reb * F.mse_loss(lambda_reb_pi, heur_reb)

        return (self.alpha * logp_a - q_a).mean() + reg_dispatch + reg_reb + distill_dispatch + distill_reb

    def update(self, data, conservative=False, only_q=False):
        # =========================
        # 1. Critic update
        # =========================
        loss_q1, loss_q2 = self.compute_loss_q(data, conservative)
        loss_q = loss_q1 + loss_q2

        # Skip non-finite critic loss
        if not torch.isfinite(loss_q).all():
            print(
                f"[WARN] Non-finite critic loss detected. "
                f"loss_q1={loss_q1.item() if torch.isfinite(loss_q1).all() else loss_q1}, "
                f"loss_q2={loss_q2.item() if torch.isfinite(loss_q2).all() else loss_q2}. "
                f"Skip this update."
            )
            self.optimizers["c1_optimizer"].zero_grad(set_to_none=True)
            self.optimizers["c2_optimizer"].zero_grad(set_to_none=True)
            return

        self.optimizers["c1_optimizer"].zero_grad(set_to_none=True)
        self.optimizers["c2_optimizer"].zero_grad(set_to_none=True)

        loss_q.backward()

        c1_grad_norm = nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        c2_grad_norm = nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)

        # Skip critic optimizer step if gradients are non-finite
        if (not torch.isfinite(c1_grad_norm)) or (not torch.isfinite(c2_grad_norm)):
            print(
                f"[WARN] Non-finite critic grad norm detected. "
                f"c1_grad_norm={c1_grad_norm}, c2_grad_norm={c2_grad_norm}. "
                f"Skip critic step."
            )
            self.optimizers["c1_optimizer"].zero_grad(set_to_none=True)
            self.optimizers["c2_optimizer"].zero_grad(set_to_none=True)
            return

        self.optimizers["c1_optimizer"].step()
        self.optimizers["c2_optimizer"].step()

        # Target critic soft update
        with torch.no_grad():
            for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if self.wandb is not None:
            self.wandb.log({
                "Q1 Loss": loss_q1.item(),
                "Q2 Loss": loss_q2.item(),
                "Critic Grad Norm 1": float(c1_grad_norm),
                "Critic Grad Norm 2": float(c2_grad_norm),
            })

        if only_q:
            return

        # =========================
        # 2. Actor update
        # =========================
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        loss_pi = None
        a_grad_norm = None

        try:
            self.optimizers["a_optimizer"].zero_grad(set_to_none=True)

            loss_pi = self.compute_loss_pi(data)

            # Skip non-finite actor loss
            if not torch.isfinite(loss_pi).all():
                print(f"[WARN] Non-finite actor loss detected: {loss_pi}. Skip actor update.")
                self.optimizers["a_optimizer"].zero_grad(set_to_none=True)
                return

            loss_pi.backward()

            a_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)

            # Skip actor optimizer step if gradients are non-finite
            if not torch.isfinite(a_grad_norm):
                print(f"[WARN] Non-finite actor grad norm detected: {a_grad_norm}. Skip actor step.")
                self.optimizers["a_optimizer"].zero_grad(set_to_none=True)
                return

            self.optimizers["a_optimizer"].step()

        finally:
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

        if self.wandb is not None and loss_pi is not None and torch.isfinite(loss_pi).all():
            log_dict = {"Policy Loss": loss_pi.item()}
            if a_grad_norm is not None and torch.isfinite(a_grad_norm):
                log_dict["Actor Grad Norm"] = float(a_grad_norm)
            self.wandb.log(log_dict)

    def _get_action_and_values(self, data, num_actions, batch_size, action_dim):
        raise NotImplementedError("Conservative/offline CQL path is not implemented for learned-lambda SAC.")

    def configure_optimizers(self):
        optimizers = dict()
        optimizers["a_optimizer"] = torch.optim.Adam(self.actor.parameters(), lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(self.critic1.parameters(), lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(self.critic2.parameters(), lr=self.q_lr)
        return optimizers

    def _policy_outputs(self, obs, env, deterministic=False, context=None):
        if context is None:
            context = self.build_regime_context(env) if self.use_regime_aware else None
        if self.use_shadow_price and self.use_learned_shadow:
            action_rl, lambda_dispatch, lambda_reb = self.select_action(
                obs,
                context=context,
                deterministic=deterministic,
                return_lambda=True,
            )
            lambda_dispatch_dict = {
                env.region[i]: float(lambda_dispatch[i]) for i in range(len(env.region))
            }
            lambda_reb_dict = {env.region[i]: float(lambda_reb[i]) for i in range(len(env.region))}
        else:
            action_rl = self.select_action(obs, context=context, deterministic=deterministic)
            lambda_dispatch = None
            lambda_reb = None
            lambda_dispatch_dict = None
            lambda_reb_dict = None

        if self.single_lambda_mode and lambda_dispatch_dict is not None:
            lambda_reb = list(lambda_dispatch)
            lambda_reb_dict = dict(lambda_dispatch_dict)

        return action_rl, lambda_dispatch, lambda_reb, lambda_dispatch_dict, lambda_reb_dict

    def _apply_shadow_dispatch(self, env, lambda_dispatch_dict):
        env.shadow_price = (
            lambda_dispatch_dict if (self.use_shadow_price and self.use_sp_dispatch) else None
        )
        env.gamma_sp = self.sp_gamma_dispatch

    def _solve_rebalance(self, env, desiredAcc, lambda_reb_dict):
        reb_action = solveRebFlow(
            env,
            env.cfg.directory,
            desiredAcc,
            self.cplexpath,
            shadow_price=lambda_reb_dict if (self.use_shadow_price and self.use_sp_reb) else None,
            gamma=self.sp_gamma_reb,
            beta=env.beta,
        )
        if reb_action is None:
            return [0.0 for _ in env.edges]
        return reb_action

    def _init_training_log(self, cfg):
        os.makedirs("Log", exist_ok=True)
        checkpoint_path = str(cfg.model.checkpoint_path)
        safe_checkpoint = re.sub(r'[<>:"/\\\\|?*\\s]+', "_", checkpoint_path).strip("_")
        now = datetime.now()
        log_path = os.path.join("Log", f"{now.strftime('%m_%d')}_{checkpoint_path}.log")
        lambda_config = (
            "Lambda config | "
            f"scale_dispatch={self.lambda_scale_dispatch} | "
            f"scale_reb={self.lambda_scale_reb} | "
            f"reg_dispatch={self.lambda_reg_dispatch} | "
            f"reg_reb={self.lambda_reg_reb}"
        )
        print(lambda_config)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"[{now.strftime('%m-%d %H:%M:%S')}] "
                f"Training started. CheckPoint_Path:{checkpoint_path}\n"
            )
            f.write(f"{lambda_config}\n")
        return log_path

    def _write_episode_log(
        self,
        log_path,
        episode,
        episode_reward,
        episode_served_demand,
        episode_rebalancing_cost,
    ):
        now = datetime.now()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{now.strftime('%H:%M:%S')}] Episode {episode} | "
                f"Reward: {episode_reward:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | "
                f"Reb. Cost: {episode_rebalancing_cost:.2f}\n"
            )

    def learn(self, cfg, Dataset=None):
        log_path = self._init_training_log(cfg)

        if Dataset is not None:
            train_episodes = cfg.model.max_episodes
            T = cfg.simulator.max_steps
            epochs = trange(train_episodes * T)
            self.train()
            for step in epochs:
                if step % 1000 == 0:
                    self.eval()
                    reward, served, cost, _ = self.test(1, self.env, verbose=False)
                    self.train()
                    epochs.set_description(
                        f"Offline Step {step} | Reward: {np.mean(reward):.2f} | "
                        f"ServedDemand: {np.mean(served):.2f} | Reb. Cost: {np.mean(cost):.2f}"
                    )
                self.save_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}.pth")
                batch = Dataset.sample_batch(self.BATCH_SIZE)
                self.update(data=batch, conservative=bool(getattr(cfg.model, "conservative", False)))
            return

        train_episodes = cfg.model.max_episodes
        epochs = trange(train_episodes)
        best_reward = -np.inf
        self.train()

        for i_episode in epochs:
            # Multi-scenario training: sample one scenario per episode.
            if hasattr(self, "env_pool") and self.env_pool:
                self.env, self.parser = random.choice(self.env_pool)

            obs, rew = self.env.reset()
            obs = self.parser.parse_obs(obs).to(self.device)
            episode_reward = rew
            #episode_served_demand = rew
            episode_served_demand = self.env.info.get("served_demand", 0)
            episode_rebalancing_cost = 0.0
            done = False

            while not done:
                context_s = self.build_regime_context(self.env) if self.use_regime_aware else None

                action_rl, lambda_dispatch, lambda_reb, lambda_dispatch_dict, lambda_reb_dict = self._policy_outputs(
                    obs,
                    self.env,
                    deterministic=False,
                    context=context_s,
                )

                desiredAcc = {
                    self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time + 1))
                    for i in range(len(self.env.region))
                }

                self._apply_shadow_dispatch(self.env, lambda_dispatch_dict)
                reb_action = self._solve_rebalance(self.env, desiredAcc, lambda_reb_dict)

                new_obs, rew, done, info = self.env.step(reb_action=reb_action)
                new_obs = self.parser.parse_obs(new_obs).to(self.device)
                context_t = self.build_regime_context(self.env) if self.use_regime_aware else None

                heur_dispatch = None
                heur_reb = None
                if self.use_teacher_distill:
                    h_dispatch_dict, h_reb_dict = self.build_heuristic_lambda(self.env)
                    heur_dispatch = [h_dispatch_dict[r] for r in self.env.region]
                    heur_reb = [h_reb_dict[r] for r in self.env.region]

                self.replay_buffer.store(
                    obs,
                    action_rl,
                    lambda_dispatch,
                    lambda_reb,
                    cfg.model.rew_scale * rew,
                    new_obs,
                    heur_lambda_dispatch=heur_dispatch,
                    heur_lambda_reb=heur_reb,
                    context_s=context_s,
                    context_t=context_t,
                )

                obs = new_obs
                episode_reward += rew
                #episode_served_demand += info["profit"]
                episode_served_demand += info.get("served_demand", 0)
                episode_rebalancing_cost += info["rebalancing_cost"]

                if i_episode > 10 and self.replay_buffer.size() >= cfg.model.batch_size:
                    batch = self.replay_buffer.sample_batch(cfg.model.batch_size)
                    self.update(data=batch, only_q=i_episode < cfg.model.only_q_steps)

            epochs.set_description(
                f"Episode {i_episode + 1} | Reward: {episode_reward:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            self._write_episode_log(
                log_path,
                i_episode + 1,
                episode_reward,
                episode_served_demand,
                episode_rebalancing_cost,
            )
            if self.wandb is not None:
                self.wandb.log(
                    {
                        "Reward": episode_reward,
                        "Served Demand": episode_served_demand,
                        "Rebalancing Cost": episode_rebalancing_cost,
                        "Step": i_episode,
                    }
                )

            self.save_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}.pth")
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}_best.pth")

    def test(self, test_episodes, env, verbose=True):
        epochs = trange(test_episodes) if verbose else range(test_episodes)
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_inflows = []
        seeds = list(range(env.cfg.seed, env.cfg.seed + test_episodes + 1))

        for i_episode in epochs:
            np.random.seed(seeds[i_episode])
            obs, rew = env.reset()
            obs = self.parser.parse_obs(obs).to(self.device)
            eps_reward = rew
            #eps_served_demand = rew
            eps_served_demand = env.info.get("served_demand", 0)
            eps_rebalancing_cost = 0.0
            inflow = np.zeros(len(env.region))
            done = False

            while not done:
                action_rl, _, _, lambda_dispatch_dict, lambda_reb_dict = self._policy_outputs(
                    obs,
                    env,
                    deterministic=True,
                )
                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                self._apply_shadow_dispatch(env, lambda_dispatch_dict)
                reb_action = self._solve_rebalance(env, desiredAcc, lambda_reb_dict)
                new_obs, rew, done, info = env.step(reb_action=reb_action)

                for k in range(len(env.edges)):
                    _, j = env.edges[k]
                    inflow[j] += reb_action[k]

                if not done:
                    obs = self.parser.parse_obs(new_obs).to(self.device)

                eps_reward += rew
                #eps_served_demand += info["profit"]
                eps_served_demand += info.get("served_demand", 0)
                eps_rebalancing_cost += info["rebalancing_cost"]

            if verbose:
                epochs.set_description(
                    f"Test Episode {i_episode + 1} | Reward: {eps_reward:.2f} | "
                    f"ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}"
                )
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_inflows.append(inflow)

        return episode_reward, episode_served_demand, episode_rebalancing_cost, episode_inflows

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = {"model": self.state_dict()}
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint["model"]
        current_state = self.state_dict()

        remapped = {}
        for key, value in model_state.items():
            new_key = key
            if "conv1.weight" in new_key:
                new_key = new_key.replace("conv1.weight", "conv1.lin.weight")
            if ".lin3." in new_key and new_key.startswith("actor."):
                new_key = new_key.replace(".lin3.", ".lin_action.")
            if new_key in current_state and current_state[new_key].shape == value.shape:
                remapped[new_key] = value

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            print(
                f"Checkpoint loaded partially from {path}: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
            )

        for key, value in self.optimizers.items():
            if key in checkpoint:
                try:
                    value.load_state_dict(checkpoint[key])
                except Exception:
                    pass

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
