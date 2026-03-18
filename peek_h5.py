import h5py
import gzip
import io
import os
import sys
import numpy as np

def read_h5(file_path):
    print(f"\n{'='*20} Reading HDF5: {os.path.basename(file_path)} {'='*20}")
    try:
        # Check if it's gzipped or raw
        if file_path.endswith('.gz'):
            print("Reading gzipped content into memory...")
            with gzip.open(file_path, 'rb') as gzf:
                content = gzf.read()
                f = h5py.File(io.BytesIO(content), 'r')
        else:
            f = h5py.File(file_path, 'r')
        
        with f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\n--- Dataset: {name} ---")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Type:  {obj.dtype}")
                    
                    # Print first 5 rows/elements
                    if obj.ndim == 0:
                        print(f"  Value: {obj[()]}")
                    elif obj.ndim == 1:
                        sample = obj[:5]
                        print(f"  Sample (first 5): {sample}")
                    else:
                        sample = obj[:5, :]
                        print(f"  Sample (first 5 rows):\n{sample}")
                    
                elif isinstance(obj, h5py.Group):
                    print(f"\nGroup: {name}")

            f.visititems(print_structure)
            print("\n" + "="*60)
    except EOFError:
        print(f"Error: {file_path} appears corrupted or incomplete (EOFError).")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    base_path = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\PGLearn-Small-14_ieee\train"
    
    # Target files to inspect
    targets = [
        os.path.join(base_path, "DCOPF", "primal.h5"),
        os.path.join(base_path, "DCOPF", "dual.h5"),
    ]
    
    found_any = False
    for target in targets:
        if os.path.exists(target):
            read_h5(target)
            found_any = True
    
    if not found_any:
        print(f"No .h5 files found at {targets}")
