import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPNNConv(MessagePassing):
    """Custom Message Passing Neural Network (MPNN) Layer.
    Computes messages using an MLP applied to concatenated source and target node features, 
    aggregates them, and updates node states using another MLP.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') # Mean aggregation for numerical stability
        
        # Message network: computes message from node j to node i
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update network: updates node i's state based on its previous state and aggregated messages
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i is the target node feature, x_j is the source node feature
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.msg_mlp(tmp)

    def update(self, aggr_out, x):
        # aggr_out is the aggregated messages for each node
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.upd_mlp(tmp)


class MPNNEdgePredictor(torch.nn.Module):
    """Deep MPNN model with BatchNorm and Residual Connections 
    for edge-level congestion prediction on the IEEE 57-bus grid.

    Architecture:
        1. Linear projection: in_channels -> hidden_channels
        2. N x (MPNNConv -> BatchNorm1d -> ReLU -> Dropout) + skip connection
        3. Edge MLP: concat(h_u, h_v) -> hidden -> 1
    """

    def __init__(self, in_channels, hidden_channels, branch_u, branch_v,
                 dropout_rate=0.2, num_layers=5, num_edge_feats=0):
        super().__init__()
        self.branch_u = branch_u
        self.branch_v = branch_v
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_edge_feats = num_edge_feats

        # Project raw node features into the hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Deep MPNN blocks
        self.convs = nn.ModuleList(
            [MPNNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
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

        # Step 1 - project to hidden dim
        x = self.input_proj(x)

        # Step 2 - deep message-passing with residual connections
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + residual  # skip / residual connection

        # Step 3 - edge predictions aligned with the 80-branch label vector
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
