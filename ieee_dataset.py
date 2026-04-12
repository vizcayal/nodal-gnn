import os
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pandapower.networks as nw

class IEEECongestionCSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        net = nw.case57()
        base_mva = 100.0
        
        self.load_buses = torch.tensor(net.load.bus.values, dtype=torch.long)
        self.gen_buses = torch.tensor(np.concatenate([net.ext_grid.bus.values, net.gen.bus.values]), dtype=torch.long)
        
        # Generator capacity bounds (pgmax, pgmin) per bus in p.u.
        pgmax_per_bus = torch.zeros(57, dtype=torch.float32)
        pgmin_per_bus = torch.zeros(57, dtype=torch.float32)
        # ext_grid
        for _, row in net.ext_grid.iterrows():
            pgmax_per_bus[int(row.bus)] = row.max_p_mw / base_mva
            pgmin_per_bus[int(row.bus)] = row.min_p_mw / base_mva
        # generators
        for _, row in net.gen.iterrows():
            pgmax_per_bus[int(row.bus)] = row.max_p_mw / base_mva
            pgmin_per_bus[int(row.bus)] = row.min_p_mw / base_mva
        self.pgmax_per_bus = pgmax_per_bus
        self.pgmin_per_bus = pgmin_per_bus
        
        self.branches = []
        for idx, row in net.line.iterrows():
            self.branches.append((int(row.from_bus), int(row.to_bus)))
        for idx, row in net.trafo.iterrows():
            self.branches.append((int(row.hv_bus), int(row.lv_bus)))
            
        self.branch_u = torch.tensor([b[0] for b in self.branches], dtype=torch.long)
        self.branch_v = torch.tensor([b[1] for b in self.branches], dtype=torch.long)
        
        # Branch thermal limits (smax) from case.json in p.u.
        case_path = os.path.join(os.path.dirname(csv_file), "PGLearn-Small-57_ieee-nminus1", "case.json")
        with open(case_path) as f:
            case_data = json.load(f)['data']
        self.smax = torch.tensor(case_data['smax'], dtype=torch.float32)
        self.b = torch.tensor(case_data['b'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values
        
        # 57-bus CSV layout: pd(42) + qd(42) + gen_status(7) + branch_status(80) + congestion(80)
        pd_data = torch.tensor(row[0:42], dtype=torch.float32)
        qd_data = torch.tensor(row[42:84], dtype=torch.float32)
        gen_status_data = torch.tensor(row[84:91], dtype=torch.float32)
        status_data = torch.tensor(row[91:171], dtype=torch.float32)
        targets = torch.tensor(row[171:], dtype=torch.float32).unsqueeze(-1)
        
        x = torch.zeros((57, 5), dtype=torch.float32)
        
        num_loads = min(len(self.load_buses), pd_data.shape[0])
        x[self.load_buses[:num_loads], 0] = pd_data[:num_loads]
        x[self.load_buses[:num_loads], 1] = qd_data[:num_loads]
        
        num_gens = min(len(self.gen_buses), gen_status_data.shape[0])
        x[self.gen_buses[:num_gens], 2] = gen_status_data[:num_gens]
        
        # Static generator capacity features (same for all samples)
        x[:, 3] = self.pgmax_per_bus
        x[:, 4] = self.pgmin_per_bus
        
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
            smax=self.smax,
            b=self.b,
            num_nodes=57
        )
        return data
