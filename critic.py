from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch 


class GNNCritic(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout.

    The SAC shadow-price variant passes two node-level lambda vectors in
    addition to the action. For backward compatibility, callers that only pass
    (state, edge_index, action) get zero lambdas and recover the original model.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, context_dim=0):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.context_dim = context_dim
        self.use_context = context_dim > 0
        self.lin1 = nn.Linear(in_channels + 3, hidden_size)
        if self.use_context:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.lin_ctx = nn.Linear(hidden_size * 2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action, lambda_dispatch=None, lambda_reb=None, context=None):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        if lambda_dispatch is None:
            lambda_dispatch = torch.zeros_like(action)
        if lambda_reb is None:
            lambda_reb = torch.zeros_like(action)
        concat = torch.cat(
            [
                x,
                action.unsqueeze(-1),
                lambda_dispatch.unsqueeze(-1),
                lambda_reb.unsqueeze(-1),
            ],
            dim=-1,
        )
        x = F.relu(self.lin1(concat))
        if self.use_context:
            if context is None:
                context = torch.zeros(x.shape[0], self.context_dim, device=x.device, dtype=x.dtype)
            context = context.to(device=x.device, dtype=x.dtype)
            ctx = self.context_mlp(context).unsqueeze(1).expand(-1, self.act_dim, -1)
            x = F.relu(self.lin_ctx(torch.cat([x, ctx], dim=-1)))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCriticLSTM(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lstm = nn.LSTM(in_channels + 1, hidden_size, dropout=0.3)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x, _ = self.lstm(concat)
        x = F.relu(self.lin1(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin2(x).squeeze(-1)  # (B)
        return x
    

class GNNValue(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t). For one-step data (on-policy).
    """
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x 
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
class GNNVF(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t). For batched data (off-policy).
    """

    def __init__(self, in_channels, hidden_size=64, act_dim=16):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = torch.sum(x, dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x).squeeze(-1)
        return x
