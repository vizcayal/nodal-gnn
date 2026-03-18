
import os
import pandas as pd
import glob

def convert_parquet_to_csv(source_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process train files
    train_files = sorted(glob.glob(os.path.join(source_dir, "train-*.parquet")))
    if train_files:
        print(f"Found {len(train_files)} training files.")
        train_csv_path = os.path.join(output_dir, "train.csv")
        
        # Initialize with header
        first = True
        for file in train_files:
            print(f"Processing {file}...")
            df = pd.read_parquet(file)
            mode = 'w' if first else 'a'
            header = first
            df.to_csv(train_csv_path, mode=mode, header=header, index=False)
            first = False
        print(f"Training data saved to {train_csv_path}")
    else:
        print("No training files found.")

    # Process test files
    test_files = sorted(glob.glob(os.path.join(source_dir, "test-*.parquet")))
    if test_files:
        print(f"Found {len(test_files)} test files.")
        test_csv_path = os.path.join(output_dir, "test.csv")
        
        # Initialize with header
        first = True
        for file in test_files:
            print(f"Processing {file}...")
            df = pd.read_parquet(file)
            mode = 'w' if first else 'a'
            header = first
            df.to_csv(test_csv_path, mode=mode, header=header, index=False)
            first = False
        print(f"Test data saved to {test_csv_path}")
    else:
        print("No test files found.")

if __name__ == "__main__":
    SOURCE_DIR = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\14_ieee"
    OUTPUT_DIR = r"C:\Users\i34005\OneDrive - Wood Mackenzie Limited\CSML\csv_output"
    
    convert_parquet_to_csv(SOURCE_DIR, OUTPUT_DIR)
