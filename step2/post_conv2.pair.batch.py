import sys
import json
import pandas as pd
import glob
import os
import re
import numpy as np

def process_group(df: pd.DataFrame, image: int, y_value: int, kernel: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    min_rows = df[df["type"] == "min"]
    min_values = pd.concat([min_rows["x_right"], min_rows["x_left"]]).dropna().unique()
    if min_values.size == 0:
        return pd.DataFrame()
    min_values = min_values.astype(int, copy=False)

    max_rows = df[df["type"] == "max"].copy()
    if max_rows.empty:
        return pd.DataFrame()

    max_x = max_rows["x"].to_numpy(dtype=int)
    within_range = (np.abs(max_x[:, None] - min_values[None, :]) <= 5).any(axis=1)
    filtered_max_rows = max_rows.loc[within_range].copy()
    if filtered_max_rows.empty:
        return pd.DataFrame()

    filtered_max_rows["x_value"] = filtered_max_rows["x"].astype(int)
    filtered_max_rows["min_left"] = filtered_max_rows["x_left"]
    filtered_max_rows["min_right"] = filtered_max_rows["x_right"]
    filtered_max_rows["y_value"] = y_value
    filtered_max_rows["point"] = list(zip(filtered_max_rows["x_value"], [y_value] * len(filtered_max_rows)))
    filtered_max_rows["image"] = image
    filtered_max_rows["kernel"] = kernel
    filtered_max_rows["conv2_value"] = filtered_max_rows["y"]

    return filtered_max_rows[[
        "image",
        "x_value",
        "y_value",
        "kernel",
        "point",
        "type",
        "min_left",
        "min_right",
        "conv2_value",
    ]]

def merge_and_process_files(root_dir, output_path_prefix):
    batch_pattern = f"{root_dir}/post_conv2/max_min/{output_path_prefix}/batch_*.max_min.csv"
    file_paths = sorted(glob.glob(batch_pattern))

    use_batch = True
    if not file_paths:
        use_batch = False
        file_pattern = f"{root_dir}/post_conv2/max_min/{output_path_prefix}/image_*.y_*.kernel_*.max_min.csv"
        file_paths = glob.glob(file_pattern)

    if not file_paths:
        print("未找到匹配的文件")
        return pd.DataFrame()

    data_frames = []

    if use_batch:
        for file_path in file_paths:
            df = pd.read_csv(
                file_path,
                sep='\t',
                dtype={
                    "image": int,
                    "y_value": int,
                    "kernel": str,
                    "type": str,
                    "x": float,
                    "y": float,
                    "x_left": float,
                    "x_right": float,
                },
            )
            if df.empty:
                continue
            for (image, y_value, kernel), group in df.groupby(["image", "y_value", "kernel"], sort=False):
                filtered_df = process_group(group, int(image), int(y_value), str(kernel))
                if not filtered_df.empty:
                    data_frames.append(filtered_df)
    else:
        file_pattern = re.compile(r"image_(\d+)\.y_(\d+)\.kernel_([a-zA-Z0-9_]+)\.max_min\.csv")
        for file_path in file_paths:
            match = file_pattern.search(os.path.basename(file_path))
            if not match:
                print(f"无法从文件名中提取信息: {file_path}")
                continue
            image = int(match.group(1))
            y_value = int(match.group(2))
            kernel = match.group(3)

            df = pd.read_csv(
                file_path,
                sep='\t',
                usecols=["x", "x_left", "x_right", "type", "y"],
                dtype={"x": float, "x_left": float, "x_right": float, "type": str, "y": float},
            )
            filtered_df = process_group(df, image, y_value, kernel)
            if not filtered_df.empty:
                data_frames.append(filtered_df)

    if not data_frames:
        return pd.DataFrame()

    merged_df = pd.concat(data_frames, ignore_index=True)
    merged_df = merged_df.sort_values(by=["image", "y_value", "x_value", "kernel"], ascending=True)
    return merged_df

def main():
    if len(sys.argv) != 2:
        print("python script.py <config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)

    region_base_dir_Vinner = config["region_base_dir_Vinner"]
    output_path_prefix = config["output_path_prefix"]

    root_dir = region_base_dir_Vinner.split(os.sep)[0]
    filtered_df = merge_and_process_files(root_dir, output_path_prefix)
    output_file_path = f"{root_dir}/post_conv2/pair/{output_path_prefix}.pair.csv"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    filtered_df.to_csv(output_file_path, sep='\t', index=False)

if __name__ == "__main__":
    main()
