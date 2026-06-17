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
from src.nets.critic import GNNVF


def _clone_graph(data):
    return Data(x=data.x.detach().cpu(), edge_index=data.edge_index.detach().cpu())


class PPORolloutBuffer:
    """On-policy graph rollout storage with GAE support."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(_clone_graph(state))
        self.actions.append(torch.as_tensor(action, dtype=torch.float32).cpu())
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def size(self):
        return len(self.rewards)

    def compute_gae(self, gamma, gae_lambda, device):
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(self.size())):
            nonterminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * nonterminal - self.values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            next_value = self.values[step]

        actions = torch.stack(self.actions).to(device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return self.states, actions, old_log_probs, returns, advantages


class PPO(nn.Module):
    """PPO-GNN baseline using the same graph parser and Dirichlet action space as SAC."""

    def __init__(self, env, input_size, cfg, parser, device=torch.device("cpu")):
        super().__init__()
        self.env = env
        self.parser = parser
        self.device = device
        self.input_size = input_size
        self.hidden_size = int(cfg.hidden_size)
        self.act_dim = env.nregion

        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.gae_lambda = float(getattr(cfg, "gae_lambda", 0.95))
        self.clip_ratio = float(getattr(cfg, "clip_ratio", 0.2))
        self.ppo_epochs = int(getattr(cfg, "ppo_epochs", 4))
        self.minibatch_size = int(getattr(cfg, "minibatch_size", getattr(cfg, "batch_size", 64)))
        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.01))
        self.value_coef = float(getattr(cfg, "value_coef", 0.5))
        self.max_grad_norm = float(getattr(cfg, "clip", 10.0))
        self.rew_scale = float(getattr(cfg, "rew_scale", 0.01))

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.agent_name = getattr(cfg, "agent_name", "PPO_GNN")
        self.wandb = None

        self.actor = GNNActor(
            input_size,
            hidden_size=self.hidden_size,
            act_dim=self.act_dim,
            safe_actor=getattr(cfg, "safe_actor", True),
        )
        self.critic = GNNVF(input_size, hidden_size=self.hidden_size, act_dim=self.act_dim)
        self.optimizers = self.configure_optimizers(cfg)
        self.to(self.device)

    def configure_optimizers(self, cfg):
        return {
            "a_optimizer": torch.optim.Adam(self.actor.parameters(), lr=float(cfg.p_lr)),
            "v_optimizer": torch.optim.Adam(self.critic.parameters(), lr=float(cfg.q_lr)),
        }

    def _normalize_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0.0, None)
        total = action.sum()
        if total <= 1e-12:
            action = np.ones(self.act_dim, dtype=np.float64) / self.act_dim
        else:
            action = action / total
        return action

    def _rebalance_from_action(self, env, action_rl):
        action_rl = self._normalize_action(action_rl)
        desired_acc = {
            env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
            for i in range(len(env.region))
        }
        reb_action = solveRebFlow(env, env.cfg.directory, desired_acc, self.cplexpath)
        if reb_action is None:
            return [0.0 for _ in env.edges]
        return reb_action

    def act(self, data, deterministic=False):
        if deterministic:
            with torch.no_grad():
                action, _ = self.actor(data.x, data.edge_index, deterministic=True)
            return self._normalize_action(action.detach().cpu().numpy()[0])

        dist = self.actor(data.x, data.edge_index, return_dist=True)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(data.x, data.edge_index)
        return (
            self._normalize_action(action.detach().cpu().numpy()[0]),
            float(log_prob.detach().cpu().item()),
            float(value.detach().cpu().item()),
        )

    def update(self, rollout):
        if rollout.size() == 0:
            return

        states, actions, old_log_probs, returns, advantages = rollout.compute_gae(
            self.gamma, self.gae_lambda, self.device
        )
        num_steps = rollout.size()
        indices = np.arange(num_steps)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_steps, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                batch = Batch.from_data_list([states[i] for i in mb_idx]).to(self.device)
                action_batch = actions[mb_idx]
                old_logp_batch = old_log_probs[mb_idx]
                return_batch = returns[mb_idx]
                adv_batch = advantages[mb_idx]

                dist = self.actor(batch.x, batch.edge_index, return_dist=True)
                log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()
                values = self.critic(batch.x, batch.edge_index)

                ratio = torch.exp(log_probs - old_logp_batch)
                unclipped = ratio * adv_batch
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, return_batch)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizers["a_optimizer"].zero_grad(set_to_none=True)
                self.optimizers["v_optimizer"].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizers["a_optimizer"].step()
                self.optimizers["v_optimizer"].step()

                if self.wandb is not None:
                    self.wandb.log({
                        "PPO Policy Loss": policy_loss.item(),
                        "PPO Value Loss": value_loss.item(),
                        "PPO Entropy": entropy.item(),
                    })

    def learn(self, cfg):
        if cfg.simulator.name != "macro":
            raise NotImplementedError("PPO-GNN baseline is currently wired for the macro simulator.")

        train_episodes = int(cfg.model.max_episodes)
        epochs = trange(train_episodes)
        best_reward = -np.inf
        self.train()

        for i_episode in epochs:
            if hasattr(self, "env_pool") and self.env_pool:
                self.env, self.parser = random.choice(self.env_pool)

            rollout = PPORolloutBuffer()
            obs, rew = self.env.reset()
            data = self.parser.parse_obs(obs).to(self.device)
            episode_reward = rew
            episode_served_demand = rew
            episode_rebalancing_cost = 0.0
            done = False

            while not done:
                action_rl, log_prob, value = self.act(data, deterministic=False)
                reb_action = self._rebalance_from_action(self.env, action_rl)
                new_obs, rew, done, info = self.env.step(reb_action=reb_action)
                rollout.store(data, action_rl, log_prob, self.rew_scale * rew, done, value)

                if not done:
                    data = self.parser.parse_obs(new_obs).to(self.device)

                episode_reward += rew
                episode_served_demand += info["profit"]
                episode_rebalancing_cost += info["rebalancing_cost"]

            self.update(rollout)
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
                action_rl = self.act(data, deterministic=True)
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
