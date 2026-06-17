from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv
import torch


class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(
            self,
            in_channels,
            hidden_size=32,
            act_dim=6,
            lambda_scale_dispatch=5.0,
            lambda_scale_reb=5.0,
            context_dim=0,
            safe_actor=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.lambda_scale_dispatch = lambda_scale_dispatch
        self.lambda_scale_reb = lambda_scale_reb
        self.context_dim = context_dim
        self.use_context = context_dim > 0
        self.safe_actor = safe_actor

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        if self.use_context:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.lin_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.lin_action = nn.Linear(hidden_size, 1)
        self.lin_lambda_dispatch = nn.Linear(hidden_size, 1)
        self.lin_lambda_reb = nn.Linear(hidden_size, 1)

    def forward(
            self,
            state,
            edge_index,
            context=None,
            deterministic=False,
            return_dist=False,
            return_lambda=False,
    ):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        h = F.leaky_relu(self.lin1(x))
        h = F.leaky_relu(self.lin2(h))

        if self.use_context:
            if context is None:
                context = torch.zeros(h.shape[0], self.context_dim, device=h.device, dtype=h.dtype)
            context = context.to(device=h.device, dtype=h.dtype)
            ctx = self.context_mlp(context).unsqueeze(1).expand(-1, self.act_dim, -1)
            h = F.leaky_relu(self.lin_fuse(torch.cat([h, ctx], dim=-1)))

        # ---------- Numerical safety before action/lambda heads ----------
        if getattr(self, "safe_actor", False):
            h = torch.nan_to_num(h, nan=0.0, posinf=10.0, neginf=-10.0)
            h = torch.clamp(h, min=-20.0, max=20.0)

            action_logits = self.lin_action(h)
            action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            action_logits = torch.clamp(action_logits, min=-20.0, max=20.0)

            concentration = F.softplus(action_logits).squeeze(-1)
            concentration = torch.nan_to_num(concentration, nan=1.0, posinf=1e3, neginf=1e-4)
            concentration = torch.clamp(concentration, min=1e-4, max=1e3)

            lambda_dispatch_raw = self.lin_lambda_dispatch(h).squeeze(-1)
            lambda_reb_raw = self.lin_lambda_reb(h).squeeze(-1)

            lambda_dispatch_raw = torch.nan_to_num(lambda_dispatch_raw, nan=0.0, posinf=20.0, neginf=-20.0)
            lambda_reb_raw = torch.nan_to_num(lambda_reb_raw, nan=0.0, posinf=20.0, neginf=-20.0)

            lambda_dispatch = self.lambda_scale_dispatch * torch.tanh(lambda_dispatch_raw)
            lambda_reb = self.lambda_scale_reb * torch.tanh(lambda_reb_raw)

        else:
            concentration = F.softplus(self.lin_action(h)).squeeze(-1)

            lambda_dispatch = self.lambda_scale_dispatch * torch.tanh(
                self.lin_lambda_dispatch(h).squeeze(-1)
            )

            lambda_reb = self.lambda_scale_reb * torch.tanh(
                self.lin_lambda_reb(h).squeeze(-1)
            )

        # Lambda heads also need finite protection.
        lambda_dispatch_raw = self.lin_lambda_dispatch(h).squeeze(-1)
        lambda_reb_raw = self.lin_lambda_reb(h).squeeze(-1)

        lambda_dispatch_raw = torch.nan_to_num(lambda_dispatch_raw, nan=0.0, posinf=20.0, neginf=-20.0)
        lambda_reb_raw = torch.nan_to_num(lambda_reb_raw, nan=0.0, posinf=20.0, neginf=-20.0)

        lambda_dispatch = self.lambda_scale_dispatch * torch.tanh(lambda_dispatch_raw)
        lambda_reb = self.lambda_scale_reb * torch.tanh(lambda_reb_raw)

        if getattr(self, "safe_actor", False):
            concentration_for_dist = concentration
        else:
            concentration_for_dist = concentration + 1e-20

        if return_dist:
            return Dirichlet(concentration_for_dist)

        if deterministic:
            action = concentration / (concentration.sum(dim=-1, keepdim=True) + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration_for_dist)
            action = m.rsample()
            log_prob = m.log_prob(action)
        if return_lambda:
            return action, log_prob, lambda_dispatch, lambda_reb
        return action, log_prob


class GNNActorLSTM(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=13):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lstm = nn.LSTM(in_channels, hidden_size, dropout=0.3)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x, _ = self.lstm(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.softplus(self.lin2(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration)
            action = m.rsample()
            log_prob = m.log_prob(action)
        return action, log_prob

