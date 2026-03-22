import os
import h5py
import numpy as np
import pandas as pd

# Paths
train_dir = r"C:\Users\luisv\ML-AI\case studies of ml\nodal-gnn\PGLearn-Small-30_ieee\train"
input_file = os.path.join(train_dir, "input.h5")
dual_file = os.path.join(train_dir, "DCOPF", "dual.h5")

def evaluate_congestion(ohm_pf, threshold=1e-4):
    """
    In DCOPF, a non-zero dual multiplier on the flow constraint (ohm_pf)
    indicates that the constraint is binding, i.e., the line is congested.
    We return 1 if congested, 0 otherwise.
    """
    return (np.abs(ohm_pf) > threshold).astype(int)

def generate_csv(num_samples=10000, out_csv="congestion_dataset.csv"):
    print(f"Loading {num_samples} samples from HDF5 files to generate CSV...")
    
    with h5py.File(input_file, 'r') as f_in, h5py.File(dual_file, 'r') as f_out:
        # 1. Loads (21 buses)
        pd_data = f_in['pd'][:num_samples]
        qd_data = f_in['qd'][:num_samples]
        
        # 2. Generators (6 nodes: 5 gens + 1 slack)
        gen_status_data = f_in['gen_status'][:num_samples]
        rmax_data = f_in['rmax'][:num_samples]
        rmin_data = f_in['rmin'][:num_samples]
        
        # 3. Grid Topology Status (41 branches)
        branch_status_data = f_in['branch_status'][:num_samples]
        
        # 4. Target Duals (41 branches)
        ohm_pf_data = f_out['ohm_pf'][:num_samples]
        
        # Calculate binary congestion
        congestion_data = evaluate_congestion(ohm_pf_data)
        
    print(f"Extracted shape PD: {pd_data.shape}, RMAX: {rmax_data.shape}, Status: {branch_status_data.shape}, Congestion: {congestion_data.shape}")
    
    # Create column names
    pd_cols = [f'pd_bus_{i}' for i in range(pd_data.shape[1])]
    qd_cols = [f'qd_bus_{i}' for i in range(qd_data.shape[1])]
    
    gen_stat_cols = [f'gen_status_{i}' for i in range(gen_status_data.shape[1])]
    rmax_cols = [f'rmax_{i}' for i in range(rmax_data.shape[1])]
    rmin_cols = [f'rmin_{i}' for i in range(rmin_data.shape[1])]
    
    stat_cols = [f'branch_status_{i}' for i in range(branch_status_data.shape[1])]
    cong_cols = [f'cong_branch_{i}' for i in range(congestion_data.shape[1])]
    
    # Concatenate features and targets
    dataset_array = np.hstack([
        pd_data, qd_data, 
        gen_status_data, rmax_data, rmin_data, 
        branch_status_data, 
        congestion_data
    ])
    all_cols = pd_cols + qd_cols + gen_stat_cols + rmax_cols + rmin_cols + stat_cols + cong_cols
    
    # Create DataFrame and save
    df = pd.DataFrame(dataset_array, columns=all_cols)
    df.to_csv(out_csv, index=False)
    print(f"Successfully saved {out_csv} with {num_samples} rows and {len(all_cols)} columns.")

if __name__ == "__main__":
    generate_csv(num_samples=50000, out_csv="congestion_dataset_v3.csv")

