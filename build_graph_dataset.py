import os
import tempfile
import gzip
import shutil
import h5py
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

def extract_gzipped_h5(file_path):
    print(f"Extracting {file_path} to temporary file...")
    temp_fd, temp_path = tempfile.mkstemp(suffix='.h5')
    os.close(temp_fd)
    with gzip.open(file_path, 'rb') as f_in:
        with open(temp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return temp_path

class IEEE30Dataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, max_samples=None):
        self.root_dir = root_dir
        self.max_samples = max_samples
        
        # Load topology from pandapower if available
        self.edge_index = self._get_topology()
        
        # Paths to gzipped h5 files
        self.input_file = os.path.join(root_dir, "input.h5.gz")
        self.primal_file = os.path.join(root_dir, "DCOPF", "primal.h5.gz")
        self.dual_file = os.path.join(root_dir, "DCOPF", "dual.h5.gz")
        
        self.temp_files = []
        self._load_data()
        
        super().__init__(root_dir, transform, pre_transform)

    def _get_topology(self):
        try:
            import pandapower.networks as nw
            net = nw.case30()
            # Construct edge index with bidirectional edges
            edges = []
            for idx, row in net.line.iterrows():
                u, v = int(row.from_bus), int(row.to_bus)
                edges.append([u, v])
                edges.append([v, u]) # undirected for typical GNN processing
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return edge_index
        except Exception as e:
            print(f"Failed to load topology using pandapower: {e}")
            print("Using a placeholder linear chain topology (update this!).")
            # Linear chain fallback for 30 buses
            edges = []
            for i in range(29):
                edges.append([i, i+1])
                edges.append([i+1, i])
            return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _load_data(self):
        # Extract files
        input_h5_path = extract_gzipped_h5(self.input_file)
        primal_h5_path = extract_gzipped_h5(self.primal_file)
        dual_h5_path = extract_gzipped_h5(self.dual_file)
        self.temp_files.extend([input_h5_path, primal_h5_path, dual_h5_path])
        
        # Open HDF5 objects
        print("Loading data into memory...")
        with h5py.File(input_h5_path, 'r') as f_in, \
             h5py.File(primal_h5_path, 'r') as f_primal, \
             h5py.File(dual_h5_path, 'r') as f_dual:
             
             # Node features (Inputs)
             self.pd = torch.tensor(f_in['pd'][:])
             self.qd = torch.tensor(f_in['qd'][:])
             
             # Branch statuses
             self.branch_status = torch.tensor(f_in['branch_status'][:])
             
             # Node targets
             self.va = torch.tensor(f_primal['va'][:])
             
             # Dual targets (lmp / kcl_p)
             self.lmp = torch.tensor(f_dual['kcl_p'][:])
             
             # Edge targets (Power flows)
             self.pf = torch.tensor(f_primal['pf'][:])
             
        self.num_samples = min(self.pd.shape[0], self.max_samples) if self.max_samples else self.pd.shape[0]
        print(f"Loaded {self.num_samples} samples.")
             
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For nodes, pd and qd have 21 elements. In a strict implementation, 
        # these need to be mapped to the correct 21 load buses.
        # Here we pad them to 30 to match the node count for a simple baseline.
        # Warning: This is a placeholder mapping. Real applications need exact bus mapping.
        
        pd_sample = self.pd[idx]
        qd_sample = self.qd[idx]
        # Pad 21 to 30
        x_node = torch.zeros((30, 2), dtype=torch.float32)
        x_node[:21, 0] = pd_sample
        x_node[:21, 1] = qd_sample
        
        # Edge features (branch status). 
        # The true 41 branch statuses should correspond to the respective edges.
        # Since we duplicated edges to make them bidirectional, we'd need shape (82, 1) or maintain directed.
        # We supply the raw 41-dim vector as a graph-level feature or dummy edge attr.
        graph_attr = self.branch_status[idx].unsqueeze(0)
        
        # Targets
        y_node = self.va[idx].unsqueeze(-1)  # Voltage angles
        y_dual = self.lmp[idx].unsqueeze(-1) # Nodal marginal prices
        
        data = Data(
            x=x_node,
            edge_index=self.edge_index,
            y=y_node,
            lmp=y_dual,
            branch_status=graph_attr
        )
        return data

    def cleanup(self):
        """Removes temporary extracted h5 files."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    dataset_dir = r"C:\Users\luisv\ML-AI\case studies of ml\nodal-gnn\PGLearn-Small-30_ieee\train"
    print("Instantiating dataset (limiting to 1000 samples for test)...")
    dataset = IEEE30Dataset(root_dir=dataset_dir, max_samples=1000)
    
    print(f"Dataset length: {len(dataset)}")
    sample_graph = dataset[0]
    print(f"Sample graph details:\n{sample_graph}")
    
    # Always remember to cleanup temporary files manually or use an atexit hook
    dataset.cleanup()
    print("Done!")
