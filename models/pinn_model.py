import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

class PINNEdgePredictor(torch.nn.Module):
    """Physics-Informed Graph Neural Network (PI-GNN) for edge-level congestion prediction.
    
    Architecture:
        1. Deep GCN message passing to learn nodal embeddings from power grid states.
        2. Physics Branch: Predicts a surrogate 'voltage angle' (theta) at each bus.
        3. Feature Branch: Standard node embedding concatenation for edges.
        4. Edge Predictor: Uses conventional hidden features PLUS the physical surrogate
           flow (proportional to theta_u - theta_v) to predict line congestion.
           
    Note: To make this fully physics-informed, a custom loss function should be applied 
    during training to enforce Kirchhoff's laws (e.g., net power injection = sum of flows).
    """

    def __init__(self, in_channels, hidden_channels, branch_u, branch_v,
                 dropout_rate=0.2, num_layers=5):
        super().__init__()
        self.branch_u = branch_u
        self.branch_v = branch_v
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Project raw node features
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Message passing layers
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )

        # Physics branch: Predict simulated voltage phase angle (theta) per node
        # In DC power flow, line flow is proportional to angle difference.
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Predicts scalar theta
        )

        # Edge-level MLP head
        # Input size: 2 * hidden_channels (node embeddings) + 1 (delta theta)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        # Step 1: Project to hidden dim
        x = self.input_proj(x)

        # Step 2: GCN Message passing
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x + residual # Skip connection

        # Step 3: Compute nodal physics states (Surrogate Phase Angle)
        theta = self.physics_head(x) # Shape: [num_nodes, 1]

        # Step 4: Extract edge features
        num_graphs = batch.num_graphs
        num_nodes_per_graph = 57  # Fixed for IEEE-57

        offsets = torch.arange(
            0, num_graphs * num_nodes_per_graph, num_nodes_per_graph, device=x.device
        )

        u_idx = (self.branch_u.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)
        v_idx = (self.branch_v.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)

        # Nodal embeddings for edge
        edge_features = torch.cat([x[u_idx], x[v_idx]], dim=-1) # Shape: [num_edges, 2*hidden]

        # Physics-informed feature: Difference in surrogate voltage angles (Delta Theta)
        # Represents the simulated power flow along the line
        delta_theta = theta[u_idx] - theta[v_idx] # Shape: [num_edges, 1]

        # Step 5: Final edge prediction combining latent embeddings and physics features
        combined_edge_features = torch.cat([edge_features, delta_theta], dim=-1)
        
        return self.mlp(combined_edge_features)

