import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv


class ECGNNEdgePredictor(torch.nn.Module):
    """Deep Edge-Conditioned Graph Neural Network (ECGNN) model with BatchNorm 
    and Residual Connections for edge-level congestion prediction on the IEEE 30-bus grid.

    Architecture:
        1. Linear projection: in_channels -> hidden_channels
        2. N x (NNConv -> BatchNorm1d -> ReLU -> Dropout) + skip connection
        3. Edge MLP: concat(h_u, h_v) -> hidden -> 1
    """

    def __init__(self, in_channels, hidden_channels, branch_u, branch_v,
                 dropout_rate=0.2, num_layers=5):
        super().__init__()
        self.branch_u = branch_u
        self.branch_v = branch_v
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # NNConv maps edge_attr (2*hidden_channels) to a weight matrix of shape [hidden_channels, hidden_channels]
            nn_edge = nn.Sequential(
                nn.Linear(2 * hidden_channels, 32),
                nn.ReLU(),
                nn.Linear(32, hidden_channels * hidden_channels)
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, nn=nn_edge, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        # Step 1 - project to hidden dim
        x = self.input_proj(x)

        # Step 2 - deep message-passing with residual connections
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            
            # Dynamically compute edge features based on current node embeddings
            edge_attr = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + residual

        # Step 3 - edge predictions aligned with the branch label vector
        num_graphs = batch.num_graphs
        num_nodes_per_graph = 30  # Fixed for IEEE-30

        offsets = torch.arange(
            0, num_graphs * num_nodes_per_graph, num_nodes_per_graph, device=x.device
        )

        u_idx = (self.branch_u.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)
        v_idx = (self.branch_v.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)

        edge_features = torch.cat([x[u_idx], x[v_idx]], dim=-1)
        return self.mlp(edge_features)
