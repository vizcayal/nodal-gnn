import os
import h5py
import json
import numpy as np
import pandas as pd
import gzip
import shutil

# Paths
train_dir = r"PGLearn-Small-57_ieee-nminus1\train"
input_file = os.path.join(train_dir, "input.h5")
primal_file = os.path.join(train_dir, "DCOPF", "primal.h5")
case_file = os.path.join(os.path.dirname(train_dir), "case.json")

# Automatically extract .h5 files if only .gz versions exist
for fpath in [input_file, primal_file]:
    if not os.path.exists(fpath) and os.path.exists(fpath + ".gz"):
        print(f"File {fpath} not found. Extracting from {fpath}.gz...")
        with gzip.open(fpath + ".gz", 'rb') as f_in:
            with open(fpath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully extracted {fpath}")


# Congestion threshold: a branch is congested when |pf| / smax >= this value.
# Both pf and smax are in per-unit (base_mva = 100).
# The DCOPF solver enforces |pf| <= smax, so binding branches have utilization = 1.0.
# A small tolerance (0.999) accounts for floating-point precision.
CONGESTION_THRESHOLD = 0.999


def generate_csv(num_samples=10000, out_csv="congestion_dataset.csv"):
    print(f"Loading {num_samples} samples from HDF5 files to generate CSV...")

    # Load smax (branch thermal capacity) from case.json — same p.u. as pf
    with open(case_file) as f:
        case = json.load(f)
    smax = np.array(case['data']['smax'])  # shape (80,)
    print(f"Loaded smax from case.json: {len(smax)} branches, range [{smax.min():.3f}, {smax.max():.3f}] p.u.")

    with h5py.File(input_file, 'r') as f_in, h5py.File(primal_file, 'r') as f_pr:
        # 1. Loads (42 buses)
        pd_data = f_in['pd'][:num_samples]
        qd_data = f_in['qd'][:num_samples]

        # 2. Generators (7 nodes: 6 gens + 1 slack)
        gen_status_data = f_in['gen_status'][:num_samples]

        # 3. Grid Topology Status (80 branches = 63 lines + 17 trafos)
        branch_status_data = f_in['branch_status'][:num_samples]

        # 4. Congestion: |pf| / smax >= threshold
        # pf (power flow) and smax are both in per-unit (base_mva = 100 MVA).
        pf_data = f_pr['pf'][:num_samples]  # shape (N, 80)

    print(f"Calculating congestion: |pf| / smax >= {CONGESTION_THRESHOLD} ...")
    utilization = np.abs(pf_data) / smax[None, :]
    congestion_data = (utilization >= CONGESTION_THRESHOLD).astype(int)

    n_congested = congestion_data.sum()
    n_total = congestion_data.size
    n_samples_with = (congestion_data.sum(axis=1) > 0).sum()
    print(f"Congestion rate: {n_congested}/{n_total} ({100*n_congested/n_total:.2f}%)")
    print(f"Samples with any congestion: {n_samples_with}/{num_samples} ({100*n_samples_with/num_samples:.2f}%)")
    print(f"Congested branches per sample: mean={congestion_data.sum(axis=1).mean():.2f}, "
          f"max={congestion_data.sum(axis=1).max():.0f}")

    print(f"Extracted shapes — PD: {pd_data.shape}, GenStatus: {gen_status_data.shape}, "
          f"BranchStatus: {branch_status_data.shape}, Congestion: {congestion_data.shape}")

    # Create column names
    pd_cols = [f'pd_bus_{i}' for i in range(pd_data.shape[1])]
    qd_cols = [f'qd_bus_{i}' for i in range(qd_data.shape[1])]
    gen_stat_cols = [f'gen_status_{i}' for i in range(gen_status_data.shape[1])]
    stat_cols = [f'branch_status_{i}' for i in range(branch_status_data.shape[1])]
    cong_cols = [f'cong_branch_{i}' for i in range(congestion_data.shape[1])]

    # Concatenate features and targets
    dataset_array = np.hstack([
        pd_data, qd_data,
        gen_status_data,
        branch_status_data,
        congestion_data
    ])
    all_cols = pd_cols + qd_cols + gen_stat_cols + stat_cols + cong_cols

    # Create DataFrame and save
    df = pd.DataFrame(dataset_array, columns=all_cols)
    df.to_csv(out_csv, index=False)
    print(f"Successfully saved {out_csv} with {num_samples} rows and {len(all_cols)} columns.")


if __name__ == "__main__":
    generate_csv(num_samples=50000, out_csv="congestion_dataset_v5.csv")
