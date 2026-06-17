import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from tqdm import trange

from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.nets.actor import GNNActor
from src.nets.critic import GNNCritic


class TD3Transition(Data):
    """One graph transition for TD3 replay."""

    def __init__(self, edge_index_s=None, x_s=None, action=None, reward=None, done=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.action = action
        self.reward = reward
        self.done = done
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class TD3ReplayBuffer:
    """Replay buffer storing graph states and simplex actions."""

    def __init__(self, device, max_size=100000):
        self.device = device
        self.max_size = int(max_size)
        self.data = []
        self.ptr = 0

    def store(self, state, action, reward, done, next_state):
        transition = TD3Transition(
            edge_index_s=state.edge_index.detach().cpu(),
            x_s=state.x.detach().cpu(),
            action=torch.as_tensor(action, dtype=torch.float32).cpu(),
            reward=torch.as_tensor(float(reward), dtype=torch.float32),
            done=torch.as_tensor(float(done), dtype=torch.float32),
            edge_index_t=next_state.edge_index.detach().cpu(),
            x_t=next_state.x.detach().cpu(),
        )
        if len(self.data) < self.max_size:
            self.data.append(transition)
        else:
            self.data[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size

    def size(self):
        return len(self.data)

    def sample_batch(self, batch_size):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return Batch.from_data_list(batch, follow_batch=["x_s", "x_t"]).to(self.device)


class TD3(nn.Module):
    """TD3-GNN baseline with deterministic simplex actions and twin graph critics."""

    def __init__(self, env, input_size, cfg, parser, device=torch.device("cpu")):
        super().__init__()
        self.env = env
        self.parser = parser
        self.device = device
        self.input_size = input_size
        self.hidden_size = int(cfg.hidden_size)
        self.act_dim = env.nregion

        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.polyak = float(getattr(cfg, "polyak", 0.995))
        self.batch_size = int(cfg.batch_size)
        self.policy_delay = int(getattr(cfg, "policy_delay", 2))
        self.target_noise = float(getattr(cfg, "target_noise", 0.2))
        self.noise_clip = float(getattr(cfg, "noise_clip", 0.5))
        self.expl_noise = float(getattr(cfg, "expl_noise", 0.1))
        self.rew_scale = float(getattr(cfg, "rew_scale", 0.01))
        self.update_after = int(getattr(cfg, "update_after", 10))
        self.clip = float(getattr(cfg, "clip", 10.0))

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.agent_name = getattr(cfg, "agent_name", "TD3_GNN")
        self.wandb = None
        self.update_step = 0

        self.actor = GNNActor(
            input_size,
            hidden_size=self.hidden_size,
            act_dim=self.act_dim,
            safe_actor=getattr(cfg, "safe_actor", True),
        )
        self.actor_target = GNNActor(
            input_size,
            hidden_size=self.hidden_size,
            act_dim=self.act_dim,
            safe_actor=getattr(cfg, "safe_actor", True),
        )
        self.critic1 = GNNCritic(input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2 = GNNCritic(input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic1_target = GNNCritic(input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2_target = GNNCritic(input_size, self.hidden_size, act_dim=self.act_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for module in [self.actor_target, self.critic1_target, self.critic2_target]:
            for param in module.parameters():
                param.requires_grad = False

        self.replay_buffer = TD3ReplayBuffer(
            device=device,
            max_size=int(getattr(cfg, "replay_size", 100000)),
        )
        self.optimizers = self.configure_optimizers(cfg)
        self.to(self.device)

    def configure_optimizers(self, cfg):
        return {
            "a_optimizer": torch.optim.Adam(self.actor.parameters(), lr=float(cfg.p_lr)),
            "c1_optimizer": torch.optim.Adam(self.critic1.parameters(), lr=float(cfg.q_lr)),
            "c2_optimizer": torch.optim.Adam(self.critic2.parameters(), lr=float(cfg.q_lr)),
        }

    def _normalize_action_np(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0.0, None)
        total = action.sum()
        if total <= 1e-12:
            return np.ones(self.act_dim, dtype=np.float64) / self.act_dim
        return action / total

    def _normalize_action_tensor(self, action):
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = torch.clamp(action, min=0.0)
        total = action.sum(dim=-1, keepdim=True)
        uniform = torch.ones_like(action) / action.shape[-1]
        return torch.where(total > 1e-12, action / (total + 1e-12), uniform)

    def select_action(self, data, noise=0.0):
        with torch.no_grad():
            action, _ = self.actor(data.x, data.edge_index, deterministic=True)
        action = action.detach().cpu().numpy()[0]
        if noise > 0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return self._normalize_action_np(action)

    def _rebalance_from_action(self, env, action_rl):
        action_rl = self._normalize_action_np(action_rl)
        desired_acc = {
            env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
            for i in range(len(env.region))
        }
        reb_action = solveRebFlow(env, env.cfg.directory, desired_acc, self.cplexpath)
        if reb_action is None:
            return [0.0 for _ in env.edges]
        return reb_action

    def update(self):
        if self.replay_buffer.size() < max(self.batch_size, self.update_after):
            return

        self.update_step += 1
        data = self.replay_buffer.sample_batch(self.batch_size)
        state = data.x_s
        edge_index = data.edge_index_s
        next_state = data.x_t
        next_edge_index = data.edge_index_t
        action = data.action.reshape(-1, self.act_dim)
        reward = data.reward
        done = data.done

        with torch.no_grad():
            next_action, _ = self.actor_target(next_state, next_edge_index, deterministic=True)
            noise = torch.randn_like(next_action) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = self._normalize_action_tensor(next_action + noise)
            q1_target = self.critic1_target(next_state, next_edge_index, next_action)
            q2_target = self.critic2_target(next_state, next_edge_index, next_action)
            backup = reward + self.gamma * (1.0 - done) * torch.min(q1_target, q2_target)

        q1 = self.critic1(state, edge_index, action)
        q2 = self.critic2(state, edge_index, action)
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        self.optimizers["c1_optimizer"].zero_grad(set_to_none=True)
        loss_q1.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad(set_to_none=True)
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        if self.update_step % self.policy_delay == 0:
            for param in self.critic1.parameters():
                param.requires_grad = False
            actor_action, _ = self.actor(state, edge_index, deterministic=True)
            actor_action = self._normalize_action_tensor(actor_action)
            actor_loss = -self.critic1(state, edge_index, actor_action).mean()
            self.optimizers["a_optimizer"].zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
            self.optimizers["a_optimizer"].step()
            for param in self.critic1.parameters():
                param.requires_grad = True

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            if self.wandb is not None:
                self.wandb.log({"TD3 Actor Loss": actor_loss.item()})

        if self.wandb is not None:
            self.wandb.log({"TD3 Q1 Loss": loss_q1.item(), "TD3 Q2 Loss": loss_q2.item()})

    def _soft_update(self, source, target):
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1.0 - self.polyak) * param.data)

    def learn(self, cfg):
        if cfg.simulator.name != "macro":
            raise NotImplementedError("TD3-GNN baseline is currently wired for the macro simulator.")

        train_episodes = int(cfg.model.max_episodes)
        epochs = trange(train_episodes)
        best_reward = -np.inf
        total_steps = 0
        self.train()

        for i_episode in epochs:
            if hasattr(self, "env_pool") and self.env_pool:
                self.env, self.parser = random.choice(self.env_pool)

            obs, rew = self.env.reset()
            data = self.parser.parse_obs(obs).to(self.device)
            episode_reward = rew
            episode_served_demand = rew
            episode_rebalancing_cost = 0.0
            done = False

            while not done:
                if total_steps < self.update_after:
                    action_rl = np.random.dirichlet(np.ones(self.act_dim))
                else:
                    action_rl = self.select_action(data, noise=self.expl_noise)

                reb_action = self._rebalance_from_action(self.env, action_rl)
                new_obs, rew, done, info = self.env.step(reb_action=reb_action)
                new_data = self.parser.parse_obs(new_obs).to(self.device)
                self.replay_buffer.store(data, action_rl, self.rew_scale * rew, done, new_data)
                data = new_data
                total_steps += 1
                self.update()

                episode_reward += rew
                episode_served_demand += info["profit"]
                episode_rebalancing_cost += info["rebalancing_cost"]

            epochs.set_description(
                f"Episode {i_episode + 1} | Reward: {episode_reward:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            if self.wandb is not None:
                self.wandb.log({
                    "Reward": episode_reward,
                    "Served Demand": episode_served_demand,
                    "Rebalancing Cost": episode_rebalancing_cost,
                    "Episode": i_episode,
                })

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

        self.eval()
        for i_episode in epochs:
            np.random.seed(seeds[i_episode])
            obs, rew = env.reset()
            data = self.parser.parse_obs(obs).to(self.device)
            eps_reward = rew
            eps_served_demand = rew
            eps_rebalancing_cost = 0.0
            inflow = np.zeros(len(env.region))
            done = False

            while not done:
                action_rl = self.select_action(data, noise=0.0)
                reb_action = self._rebalance_from_action(env, action_rl)
                new_obs, rew, done, info = env.step(reb_action=reb_action)

                for k in range(len(env.edges)):
                    _, j = env.edges[k]
                    inflow[j] += reb_action[k]

                if not done:
                    data = self.parser.parse_obs(new_obs).to(self.device)
                eps_reward += rew
                eps_served_demand += info["profit"]
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {"model": self.state_dict()}
        for key, optimizer in self.optimizers.items():
            checkpoint[key] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model"], strict=False)
        for key, optimizer in self.optimizers.items():
            if key in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint[key])
                except Exception:
                    pass
