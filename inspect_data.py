import h5py
import gzip
import os

data_dir = r"PGLearn-Small-57_ieee-nminus1\train"

def inspect_h5_gz(file_path, out_file):
    out_file.write(f"\n--- Inspecting {file_path} ---\n")
    try:
        import tempfile
        import shutil
        with gzip.open(file_path, 'rb') as f_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f_out:
                shutil.copyfileobj(f_in, f_out)
                temp_path = f_out.name
        
        with h5py.File(temp_path, 'r') as f:
            def print_attrs(name, obj):
                out_file.write(f"{name}\n")
                for key, val in obj.attrs.items():
                    out_file.write(f"    {key}: {val}\n")
                if isinstance(obj, h5py.Dataset):
                    out_file.write(f"    Shape: {obj.shape}, Type: {obj.dtype}\n")
            f.visititems(print_attrs)
        os.remove(temp_path)
    except Exception as e:
        out_file.write(f"Error: {e}\n")

with open("schema.txt", "w") as f:
    inspect_h5_gz(os.path.join(data_dir, "input.h5.gz"), f)
    inspect_h5_gz(os.path.join(data_dir, "DCOPF", "primal.h5.gz"), f)
    inspect_h5_gz(os.path.join(data_dir, "DCOPF", "dual.h5.gz"), f)
    inspect_h5_gz(os.path.join(data_dir, "DCOPF", "meta.h5.gz"), f)
