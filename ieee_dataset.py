import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pandapower.networks as nw

class IEEECongestionCSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        net = nw.case30()
        
        self.load_buses = torch.tensor(net.load.bus.values, dtype=torch.long)
        self.gen_buses = torch.tensor(np.concatenate([net.ext_grid.bus.values, net.gen.bus.values]), dtype=torch.long)
        
        self.branches = []
        for idx, row in net.line.iterrows():
            self.branches.append((int(row.from_bus), int(row.to_bus)))
            
        self.branch_u = torch.tensor([b[0] for b in self.branches], dtype=torch.long)
        self.branch_v = torch.tensor([b[1] for b in self.branches], dtype=torch.long)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values
        
        pd_data = torch.tensor(row[0:21], dtype=torch.float32)
        qd_data = torch.tensor(row[21:42], dtype=torch.float32)
        gen_status_data = torch.tensor(row[42:48], dtype=torch.float32)
        rmax_data = torch.tensor(row[48:54], dtype=torch.float32)
        rmin_data = torch.tensor(row[54:60], dtype=torch.float32)
        status_data = torch.tensor(row[60:101], dtype=torch.float32)
        targets = torch.tensor(row[101:], dtype=torch.float32).unsqueeze(-1)
        
        x = torch.zeros((30, 5), dtype=torch.float32)
        
        num_loads = min(len(self.load_buses), pd_data.shape[0])
        x[self.load_buses[:num_loads], 0] = pd_data[:num_loads]
        x[self.load_buses[:num_loads], 1] = qd_data[:num_loads]
        
        num_gens = min(len(self.gen_buses), gen_status_data.shape[0])
        x[self.gen_buses[:num_gens], 2] = gen_status_data[:num_gens]
        x[self.gen_buses[:num_gens], 3] = rmax_data[:num_gens]
        x[self.gen_buses[:num_gens], 4] = rmin_data[:num_gens]
        
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
