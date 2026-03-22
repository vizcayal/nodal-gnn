import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import pandapower.networks as nw

class IEEECongestionCSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        # Topology and Node Mappings from pandapower
        net = nw.case30()
        
        # Map indices (0-indexed)
        self.load_buses = torch.tensor(net.load.bus.values, dtype=torch.long)
        # Generators (Slack bus + standard generators)
        self.gen_buses = torch.tensor(np.concatenate([net.ext_grid.bus.values, net.gen.bus.values]), dtype=torch.long)
        
        self.branches = [] # List of tuples (from_bus, to_bus)
        for idx, row in net.line.iterrows():
            self.branches.append((int(row.from_bus), int(row.to_bus)))
            
        # Target evaluation pairs
        self.branch_u = torch.tensor([b[0] for b in self.branches], dtype=torch.long)
        self.branch_v = torch.tensor([b[1] for b in self.branches], dtype=torch.long)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values
        
        # Parse inputs according to the v3 CSV schema
        # 0 to 20: PD (21 values)
        # 21 to 41: QD (21 values)
        # 42 to 47: GEN_STATUS (6 values)
        # 48 to 53: RMAX (6 values)
        # 54 to 59: RMIN (6 values)
        # 60 to 100: Branch Status (41 ones or zeros)
        # 101 to 141: Congestion Target (41 values)
        
        pd_data = torch.tensor(row[0:21], dtype=torch.float32)
        qd_data = torch.tensor(row[21:42], dtype=torch.float32)
        
        gen_status_data = torch.tensor(row[42:48], dtype=torch.float32)
        rmax_data = torch.tensor(row[48:54], dtype=torch.float32)
        rmin_data = torch.tensor(row[54:60], dtype=torch.float32)
        
        status_data = torch.tensor(row[60:101], dtype=torch.float32)
        targets = torch.tensor(row[101:], dtype=torch.float32).unsqueeze(-1) # (41, 1)
        
        # Node features shape (30, 5) -> PD, QD, GEN_STATUS, RMAX, RMIN
        x = torch.zeros((30, 5), dtype=torch.float32)
        
        # Map loads safely (pandapower case30 sometimes has 20 loads vs 21 in MATPOWER datasets)
        num_loads = min(len(self.load_buses), pd_data.shape[0])
        x[self.load_buses[:num_loads], 0] = pd_data[:num_loads]
        x[self.load_buses[:num_loads], 1] = qd_data[:num_loads]
        
        # Map generators safely (slack included)
        num_gens = min(len(self.gen_buses), gen_status_data.shape[0])
        x[self.gen_buses[:num_gens], 2] = gen_status_data[:num_gens]
        x[self.gen_buses[:num_gens], 3] = rmax_data[:num_gens]
        x[self.gen_buses[:num_gens], 4] = rmin_data[:num_gens]
        
        # ---- Enmascaramiento Dinámico (N-1 Contingency Masking) ----
        active_edges = []
        for branch_idx, is_active in enumerate(status_data):
            if is_active > 0.5:
                u, v = self.branches[branch_idx]
                active_edges.append([u, v])
                active_edges.append([v, u])
                
        if len(active_edges) > 0:
            edge_index = torch.tensor(active_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        data = Data(
            x=x,
            edge_index=edge_index,
            y=targets,
            status=status_data.unsqueeze(-1), 
            num_nodes=30
        )
        return data

class DeepEdgeCongestionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, branch_u, branch_v):
        super().__init__()
        self.num_layers = num_layers
        
        # Encoding inicial de features (5 features -> hidden)
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Listas de módulos para iterar la profundidad
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.branch_u = branch_u
        self.branch_v = branch_v
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout added for regularization
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # 1. Proyección lineal inicial
        x = self.node_encoder(x)
        
        # 2. Paso de mensajes iterativo con Múltiples Saltos (Over-smoothing Prevention)
        for i in range(self.num_layers):
            x_res = x  # Guardar conexión residual (Skip Connection)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = x + x_res # Sumar conexión residual (Evita difuminado del over-smoothing)
        
        # 3. Agrupación y Clasificación de Ramas
        num_graphs = batch.num_graphs
        num_nodes_per_graph = 30
        offsets = torch.arange(0, num_graphs * num_nodes_per_graph, num_nodes_per_graph, device=x.device)
        
        u_indices = (self.branch_u.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)
        v_indices = (self.branch_v.unsqueeze(0).to(x.device) + offsets.unsqueeze(1)).view(-1)
        
        nodes_u = x[u_indices]
        nodes_v = x[v_indices]
        
        edge_h = torch.cat([nodes_u, nodes_v], dim=-1)
        out = self.edge_mlp(edge_h)
        return out

if __name__ == "__main__":
    csv_path = "congestion_dataset_v3.csv"
    
    print("Loading physically mapped dataset...")
    dataset = IEEECongestionCSVDataset(csv_path)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instanciar el modelo GNN Profundo
    model = DeepEdgeCongestionGNN(
        in_channels=5, 
        hidden_channels=64, # Aumentamos la base de representación interna
        num_layers=5,       # 5 capas permiten a los nodos "ver" a 5 líneas de distancia
        branch_u=dataset.branch_u,
        branch_v=dataset.branch_v
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    epochs = 10
    print(f"Empezando entrenamiento...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            y = batch.y.view(-1, 1)
            status_mask = batch.status.view(-1, 1)
            
            loss_components = criterion(out, y)
            loss_masked = (loss_components * status_mask).sum() / (status_mask.sum() + 1e-8)
            
            loss_masked.backward()
            optimizer.step()
            total_loss += loss_masked.item() * batch.num_graphs
            
        train_loss = total_loss / len(train_dataset)
        
        model.eval()
        correct = 0
        total_active_branches = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                y = batch.y.view(-1, 1)
                status_mask = batch.status.view(-1, 1)
                
                preds = (out > 0).float()
                correct_preds = (preds == y).float() * status_mask
                
                correct += correct_preds.sum().item()
                total_active_branches += status_mask.sum().item()
                
        test_acc = correct / total_active_branches
        print(f"Epoch {epoch+1:02d} | Train Loss (Masked): {train_loss:.4f} | Test Acc (Masked): {test_acc:.4f}")
