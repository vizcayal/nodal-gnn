import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn


class GCNEdgePredictor(torch.nn.Module):
    """Deep GCN (5 layers) with BatchNorm and Residual Connections for edge-level
    congestion prediction on the IEEE 57-bus grid.

    Architecture:
        1. Linear projection: in_channels → hidden_channels
        2. 5 × (GCNConv → BatchNorm1d → ReLU → Dropout) + skip connection
        3. Edge MLP: concat(h_u, h_v) → hidden → 1
    """

    def __init__(self, in_channels, hidden_channels, branch_u, branch_v,
                 dropout_rate=0.2, num_layers=5, num_edge_feats=0):
        super().__init__()
        self.branch_u = branch_u
        self.branch_v = branch_v
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_edge_feats = num_edge_feats

        # Project raw node features into the hidden dimension (enables residual add)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Deep GCN blocks
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )

        # Edge-level MLP head
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_edge_feats, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        # Step 1 – project to hidden dim
        x = self.input_proj(x)

        # Step 2 – deep message-passing with residual connections
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + residual  # skip / residual connection

        # Step 3 – edge predictions aligned with the 80-branch label vector
        num_graphs = batch.num_graphs
        num_nodes_per_graph = 57  # Fixed for IEEE-57

        offsets = torch.arange(
            0, num_graphs * num_nodes_per_graph, num_nodes_per_graph, device=x.device
        )

        u_idx = (self.branch_u.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)
        v_idx = (self.branch_v.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)

        edge_features = torch.cat([x[u_idx], x[v_idx]], dim=-1)
        if self.num_edge_feats > 0:
            smax = batch.smax.view(-1, 1)
            status = batch.status.view(-1, 1)
            b = batch.b.view(-1, 1)
            edge_features = torch.cat([edge_features, smax, status, b], dim=-1)
        return self.mlp(edge_features)
