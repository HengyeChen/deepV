import os
import re
import sys
import pandas as pd

def main():
    base_dir = sys.argv[1]
    merge_dir = os.path.join(base_dir, "conv2", "conv1_merge")

    # Check if merge_dir exists
    if not os.path.exists(merge_dir):
        print(f"Warning: Merge directory {merge_dir} does not exist, no data to merge")
        sys.exit(0)

    csv_files = [f for f in os.listdir(merge_dir) if f.endswith(".merge.csv")]

    # Check if there are any files to merge
    if not csv_files:
        print(f"Warning: No CSV files found in {merge_dir}")
        sys.exit(0)

    def extract_sort_key(filename):
        match = re.search(r'conv1.(\d+)-', filename)
        return int(match.group(1)) if match else float('inf')

    sorted_files = sorted(csv_files, key=extract_sort_key)
    sorted_file_paths = [os.path.join(merge_dir, f) for f in sorted_files]

    all_dfs = []
    for idx, file_path in enumerate(sorted_file_paths):
        df = pd.read_csv(file_path, sep='\t')
        if idx > 0:
            df = df.iloc[1:]
        all_dfs.append(df)

    if not all_dfs:
        print("Warning: No valid data found after reading files")
        sys.exit(0)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    chr_part = base_dir.split('_')[0]
    min_key = extract_sort_key(sorted_files[0]) if sorted_files else 0

    def extract_max_end(filename):
        match = re.search(r'-(\d+)kb', filename)
        return int(match.group(1)) if match else 0

    max_end = max(extract_max_end(f) for f in sorted_files) if sorted_files else 0

    output_dir = os.path.join(base_dir, "conv2")
    output_filename = f"{chr_part}.conv1.{min_key}-{max_end}kb.merge.csv"
    output_path = os.path.join(output_dir, output_filename)

    merged_df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    main()
