import h5py
import numpy as np
import os

def find_congestion(file_path):
    print(f"Searching for congestion in: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with h5py.File(file_path, 'r') as f:
        print(f"Datasets available in dual.h5: {list(f.keys())}")
        
        # Check active power flow duals (often linked to line limits)
        if 'pf' in f:
            pf_duals = f['pf'][:]
            # Find cases where any line limit dual is non-zero
            # We use a small epsilon because of float32 precision
            congested_lines_mask = np.abs(pf_duals) > 1e-5
            cases_with_congestion = np.any(congested_lines_mask, axis=1)
            congestion_indices = np.where(cases_with_congestion)[0]
            
            print(f"Found {len(congestion_indices)} cases with binding line limits (pf dual > 0) out of {len(pf_duals)}.")
            
            if len(congestion_indices) > 0:
                print("\nSample cases where line limits are reached:")
                for idx in congestion_indices[:3]:
                    line_idx = np.where(congested_lines_mask[idx])[0]
                    print(f"Index {idx}: Line(s) {line_idx} are binding. Dual values: {pf_duals[idx, line_idx]}")
        
        # Check generation duals
        if 'pg' in f:
            pg_duals = f['pg'][:]
            at_limit_gen = np.abs(pg_duals) > 1e-5
            gen_limit_cases = np.any(at_limit_gen, axis=1)
            print(f"Found {np.sum(gen_limit_cases)} cases where a generator is at a limit.")

        # Check LMP spread again
        if 'kcl_p' in f:
            lmp = f['kcl_p'][:]
            diff = np.max(lmp, axis=1) - np.min(lmp, axis=1)
            spread_cases = np.where(diff > 1e-2)[0]
            print(f"Found {len(spread_cases)} cases with LMP spread (price difference between nodes).")

if __name__ == "__main__":
    path = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\PGLearn-Small-30_ieee\train\DCOPF\dual.h5"
    find_congestion(path)
