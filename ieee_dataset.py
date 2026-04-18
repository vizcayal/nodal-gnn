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

        case_path = os.path.join(os.path.dirname(csv_file), "PGLearn-Small-57_ieee-nminus1", "case.json")
        with open(case_path) as f:
            case_data = json.load(f)["data"]
        
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

        # Keep branch indexing in case/H5 order so status, smax, b, and y are aligned.
        bus_fr = np.array(case_data["bus_fr"], dtype=np.int64) - 1
        bus_to = np.array(case_data["bus_to"], dtype=np.int64) - 1
        self.smax = torch.tensor(case_data["smax"], dtype=torch.float32)
        self.b = torch.tensor(case_data["b"], dtype=torch.float32)

        if len(bus_fr) != len(self.smax):
            raise ValueError("Branch endpoint count and smax count do not match in case.json.")

        self.branches = list(zip(bus_fr.tolist(), bus_to.tolist()))
        self.branch_u = torch.tensor(bus_fr, dtype=torch.long)
        self.branch_v = torch.tensor(bus_to, dtype=torch.long)

        # Sanity check topology consistency against pandapower (ignoring direction/order).
        pp_branches = []
        for _, row in net.line.iterrows():
            pp_branches.append((int(row.from_bus), int(row.to_bus)))
        for _, row in net.trafo.iterrows():
            pp_branches.append((int(row.hv_bus), int(row.lv_bus)))

        normalize = lambda pairs: {tuple(sorted(pair)) for pair in pairs}
        if normalize(self.branches) != normalize(pp_branches):
            raise ValueError("case.json branch endpoints do not match pandapower case57 topology.")
        
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
        
        x = torch.zeros((57, 6), dtype=torch.float32)
        
        num_loads = min(len(self.load_buses), pd_data.shape[0])
        x[self.load_buses[:num_loads], 0] = pd_data[:num_loads]
        x[self.load_buses[:num_loads], 1] = qd_data[:num_loads]
        
        num_gens = min(len(self.gen_buses), gen_status_data.shape[0])
        x[self.gen_buses[:num_gens], 2] = gen_status_data[:num_gens]
        
        # Static generator capacity features (same for all samples)
        x[:, 3] = self.pgmax_per_bus
        x[:, 4] = self.pgmin_per_bus
        
        active_edges = []
        degree = torch.zeros(57, dtype=torch.float32)
        for branch_idx, is_active in enumerate(status_data):
            if is_active > 0.5:
                u, v = self.branches[branch_idx]
                active_edges.append([u, v])
                active_edges.append([v, u])
                degree[u] += 1
                degree[v] += 1
        # Degree centrality: normalize by max possible degree (N-1 = 56)
        x[:, 5] = degree / 56.0
                
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
