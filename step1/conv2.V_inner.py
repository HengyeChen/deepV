import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json
import sys

def create_image_datacounts(data_points, size=(1000, 100)):
    image = np.zeros(size)
    for x, y in data_points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            image[x, y] += 1
    return image

def get_data_points_datacounts(file_path, start, end, return_min_x=False):
    data = pd.read_csv(file_path, sep='\t', header=None)
    filtered_data = data[(data[1] >= start) & (data[1] <= end)]
    data_points = []
    for _, row in filtered_data.iterrows():
        x, y, count = row[1], row[2], row[3]
        data_points.extend([(x, y)] * count)
    min_x = min(filtered_data[1])
    adjusted_data_points = [(x - min_x, y) for x, y in data_points]
    if return_min_x:
        return adjusted_data_points, min_x
    return adjusted_data_points

def create_image(data_points, size=(1000, 100)):
    unique_data_points = set(data_points)
    image = np.zeros(size)
    for x, y in unique_data_points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            image[x, y] += 1
    return image

def get_data_points(file_path, start, end, return_min_x=False):
    data = pd.read_csv(file_path, sep='\t', header=None)
    filtered_data = data[(data[1] >= start) & (data[1] <= end)]
    data_points = list(zip(filtered_data[1], filtered_data[2]))
    min_x = min(filtered_data[1])
    adjusted_data_points = [(x - min_x, y) for x, y in data_points]
    if return_min_x:
        return adjusted_data_points, min_x
    return adjusted_data_points

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_csv_merge_region(file_path):
    df = pd.read_csv(file_path, sep='\t')
    columns_to_keep = ["start_y", "start_x", "end_y", "end_x"]
    df = df[columns_to_keep]
    df = df.drop_duplicates()
    df = df.sort_values(by="start_y")
    df['group'] = (df['start_y'].diff() > 1).cumsum()
    result = df.groupby('group').agg({
        'start_y': 'min',
        'end_y': 'max',
    }).reset_index(drop=True)

    merged_result = []
    for i in range(len(result)):
        if i == 0:
            merged_result.append(result.iloc[i])
        else:
            prev = merged_result[-1]
            curr = result.iloc[i]
            if prev['end_y'] >= curr['start_y']:
                merged_result[-1]['end_y'] = max(prev['end_y'], curr['end_y'])
            else:
                merged_result.append(curr)
    result = pd.DataFrame(merged_result)

    result['range'] = result['end_y'] - result['start_y']
    result['start_x'] = 0
    result['end_x'] = 100
    return result

def process_csv(file_path):
    # Check if file exists, return None if not found
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, sep='\t')
    # Check if dataframe is empty or has no valid data
    if df.empty:
        return None
    columns_to_keep = ["start_y", "start_x", "end_y", "end_x"]
    df = df[columns_to_keep]
    df = df.drop_duplicates()
    df = df.sort_values(by="start_y")
    df['group'] = (df['start_y'].diff() > 1).cumsum()
    result = df.groupby('group').agg({
        'start_y': 'min',
        'end_y': 'max',
    }).reset_index(drop=True)
    result['range'] = result['end_y'] - result['start_y']
    result['start_x'] = 0
    result['end_x'] = 100
    return result

def filter_csv_by_sigmoid(csv_path, sigmoid_output_dir, threshold=0.95):
    df = pd.read_csv(csv_path, sep='\t')
    values = df['conv2_channel_value'].values
    values = values / df['conv2_channel_value'].mean()
    log_values = np.log1p(values)
    sigmoid_values = sigmoid(log_values)

    df['sigmoid_value'] = sigmoid_values
    significant_rows = df[df['sigmoid_value'] > threshold]

    sigmoid_base_name = os.path.basename(csv_path).replace(".csv", f".sigmoid.csv")
    sigmoid_csv_path = os.path.join(sigmoid_output_dir, sigmoid_base_name)
    df.to_csv(sigmoid_csv_path, sep='\t', index=False)
    return sigmoid_csv_path

def filter_sigmoid_files(input_dir, output_file, threshold=0.8):
    filtered_rows = pd.DataFrame()
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".sigmoid.csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path, sep='\t')
            filtered = df[df['sigmoid_value'] >= threshold]
            filtered_rows = pd.concat([filtered_rows, filtered], ignore_index=True)
    filtered_rows = filtered_rows.sort_values(by=['start_y', 'start_x'], ascending=[True, True])
    output_file = os.path.join(input_dir, output_file)
    filtered_rows.to_csv(output_file, sep='\t', index=False)

def generate_images(file_path, result):
    max_range = result['range'].max()
    images = []
    min_x_values = []

    for _, row in result.iterrows():
        start = row['start_y']
        end = row['end_y']
        if pd.isna(start) or pd.isna(end):
            print(f"Invalid start or end value in row: {row}")
            continue
        try:
            data_points, min_x = get_data_points(file_path, start, end, return_min_x=True)
            if not data_points:
                print(f"No data points found for range {start}-{end}. Skipping...")
                continue

            image = create_image(data_points, size=(row['range'], 100))

            pad_height = max_range - image.shape[0]
#            pad_top = pad_height // 2
#            pad_bottom = pad_height - pad_top
            pad_top = 0
            pad_bottom = pad_height
            padded_image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

            padded_image = np.expand_dims(padded_image, axis=0)
            padded_image = np.expand_dims(padded_image, axis=-1)

            images.append(padded_image)
            min_x_values.append(min_x)
        except ValueError as e:
            print(f"Error processing range {start}-{end}: {e}")
            continue
    if images:
        return np.vstack(images), min_x_values, max_range
    else:
        print("No images were generated.")
        return np.array([]), [], max_range

def save_conv2_output_to_csv(base_dir, conv_output, min_x_values, kernel_height=10, kernel_width=20, stride_height=1, stride_width=1):
    image_dir = os.path.join(base_dir, "conv2_image")
    sigmoid_dir = os.path.join(base_dir, "conv2_image_sigmoid")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sigmoid_dir, exist_ok=True)
    
    for i, output in enumerate(conv_output):
        conv2_channel = output[:, :, 0]
        conv2_positions = np.argwhere(conv2_channel)

        input_positions = []
        for pos in conv2_positions:
            output_y, output_x = pos
            start_y = output_y * stride_height + min_x_values[i]
            start_x = output_x * stride_width
            end_y = start_y + kernel_height
            end_x = start_x + kernel_width
            conv2_channel_value = conv2_channel[output_y, output_x]
            input_positions.append((start_y, start_x, end_y, end_x, conv2_channel_value))

        region_start = min_x_values[i]
        region_end = min_x_values[i] + (1000 * stride_height)

        image_csv_path = os.path.join(image_dir, f"image_{i}.csv")
        with open(image_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(["start_y", "start_x", "end_y", "end_x", "conv2_channel_value", "image", "region_start", "region_end"])
            for region in input_positions:
                writer.writerow([*region, i, region_start, region_end])
        filter_csv_by_sigmoid(image_csv_path, threshold=0.95)

def generate_fixed_kernel_before(kernel_image,
                          y_values=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
                          reference_y=25,
                          final_shape=(20, 100)):
    results = {}
    for y in y_values:
        if y < reference_y:
            # y 小于参考值，起始值为 (reference_y - y)，终止值为 100
            start_y = reference_y - y
            end_y = 100
            sub_matrix = kernel_image[15:35, start_y:end_y]
            padding = np.zeros((20, start_y))
            final_matrix = np.zeros(final_shape)
            final_matrix[:, :end_y - start_y] = sub_matrix
            final_matrix[:, end_y - start_y:] = padding
        elif y == reference_y:
            # y 等于参考值，不需要 padding
            sub_matrix = kernel_image[15:35, :]
            final_matrix = np.zeros(final_shape)
            final_matrix[:, :] = sub_matrix
        elif y > reference_y:
            # y 大于参考值，起始值为 0，终止值为 (100 - (y - reference_y))
            start_y = 0
            end_y = 100 - (y - reference_y)
            sub_matrix = kernel_image[15:35, start_y:end_y]
            padding = np.zeros((20, y - reference_y))
            final_matrix = np.zeros(final_shape)
            final_matrix[:, y - reference_y:] = sub_matrix
            final_matrix[:, :y - reference_y] = padding
        results[f"final_matrix_y{y}"] = final_matrix

    num_kernels = len(y_values)
    fixed_kernel = np.zeros((final_shape[0], final_shape[1], 1, num_kernels))
    for i, y in enumerate(y_values):
        fixed_kernel[:, :, 0, i] = results[f"final_matrix_y{y}"]
    return fixed_kernel, num_kernels

def generate_fixed_kernel(kernel_dir, name, y_values=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], final_shape=(20, 100)):
    results = {}
    for y in y_values:
        file_path = os.path.join(kernel_dir, f"kernel_{name}_y{y}.scale.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping...")
            continue

        kernel_matrix = pd.read_csv(file_path, sep='\t', header=None).values
        # 提取 [15:35] 部分
        sub_matrix = kernel_matrix[15:35, :]
        results[f"final_matrix_y{y}"] = sub_matrix

    num_kernels = len(results)
    fixed_kernel = np.zeros((final_shape[0], final_shape[1], 1, num_kernels))
    for i, y in enumerate(y_values):
        fixed_kernel[:, :, 0, i] = results[f"final_matrix_y{y}"]
    return fixed_kernel, num_kernels

def process_sigmoid_files(input_base_dir, output_file, threshold=0.7):
    all_filtered_rows = pd.DataFrame()
    for channel_name in ["y5", "y10", "y15", "y20", "y25", "y30", "y35", "y40", "y45", "y50", "y55", "y60", "y65", "y70", "y75", "y80", "y85", "y90", "y95"]:
        input_dir = f"{input_base_dir}_{channel_name}"
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist. Skipping...")
            continue
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_file_path = os.path.join(input_dir, file_name)
                df = pd.read_csv(input_file_path, sep='\t')
                filtered_df = df[df['sigmoid_value'] >= threshold]
                all_filtered_rows = pd.concat([all_filtered_rows, filtered_df], ignore_index=True)
    # 按照 image 列从小到大排序，在每个 image 分组中，先按 start_y 再按 kernel_y_value 排序
    all_filtered_rows = all_filtered_rows.sort_values(by=['image', 'start_y', 'kernel_y_value'], ascending=[True, True, True])
    all_filtered_rows.to_csv(output_file, sep='\t', index=False)

def generate_conv_output(model, name, images, kernel_dir='kernel_generate/kernel'):
    # Check if images array is empty
    if len(images) == 0:
        print(f"No images provided for {name}, skipping convolution")
        return None

    conv_layer = model.get_layer('conv2')
    fixed_kernel, num_kernels = generate_fixed_kernel(kernel_dir=kernel_dir, name=name)
    fixed_bias = np.zeros(num_kernels)
    conv_layer.set_weights([fixed_kernel, fixed_bias])
    conv_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer.output)
    conv_output = conv_model.predict(images)
    return conv_output

def process_conv_output(conv_output, base_dir, min_x_values, kernel_height=20, kernel_width=100, stride_height=1, stride_width=1):
    for i, output in enumerate(conv_output):
        for channel_idx, channel_name in enumerate(["y5", "y10", "y15", "y20", "y25", "y30", "y35", "y40", "y45", "y50", "y55", "y60", "y65", "y70", "y75", "y80", "y85", "y90", "y95"]):
            conv_channel = output[:, :, channel_idx]
            conv_positions = np.argwhere(conv_channel)

            input_positions = []
            for pos in conv_positions:
                output_y, output_x = pos
                start_y = output_y * stride_height + min_x_values[i]
                start_x = output_x * stride_width
                end_y = start_y + kernel_height
                end_x = start_x + kernel_width
                conv_channel_value = conv_channel[output_y, output_x]

                input_positions.append((start_y, start_x, end_y, end_x, conv_channel_value))

            region_start = min_x_values[i]
            region_end = min_x_values[i] + (1000 * stride_height)

            image_csv_dir = os.path.join(base_dir, f"conv2_image_{channel_name}")
            sigmoid_output_dir = os.path.join(base_dir, f"conv2_image_sigmoid_{channel_name}")
            os.makedirs(image_csv_dir, exist_ok=True)
            os.makedirs(sigmoid_output_dir, exist_ok=True)

            image_csv_path = os.path.join(image_csv_dir, f"image_{i}.csv")
            ensure_directories_exist(image_csv_path)
            with open(image_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(["start_y", "start_x", "end_y", "end_x", "conv2_channel_value", "image", "region_start", "region_end", "kernel_y_value"])
                for region in input_positions:
                    kernel_y_value = int(channel_name[1:])
                    writer.writerow([*region, i, region_start, region_end, kernel_y_value])

            filter_csv_by_sigmoid(image_csv_path, sigmoid_output_dir)
            
def ensure_directories_exist(*file_paths):
    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


if len(sys.argv) != 2:
    print("Usage: python script.py <config_file>")
    sys.exit(1)

config_file_path = sys.argv[1]
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

file_path = config["file_path"]
conv1_file_path = config["conv1_file_path"]
conv1_merge_path = config["conv1_merge_path"]
region_base_dir_Vinner = config["region_base_dir_Vinner"]

ensure_directories_exist(conv1_merge_path)

# 把conv1的结果按照区间合并
result = process_csv(conv1_file_path)

# Skip processing if no data available
if result is None:
    print(f"Warning: No data found in {conv1_file_path}, skipping")
    sys.exit(0)

result.to_csv(conv1_merge_path, sep='\t', index=False)

'''
# 产生images
images, min_x_values, max_range = generate_images(file_path, result)

model = Sequential([
    Conv2D(19, (20, 100), strides=(1, 1), activation='relu', input_shape=(max_range, 100, 1), name='conv2'),
])

# 设置卷积核
names = ['standard', 'left', 'right', 'middle']
for name in names:
    globals()[f"conv_output_{name}"] = generate_conv_output(
        model=model,
        name=name,
        kernel_dir='/home/nwh/software/temp/loMNase_K562/pre_test_4/kernel_generate/kernel',
        images=images
    )

# 反推
for name in names:
    conv_output = globals()[f"conv_output_{name}"]
#    base_dir = f"conv2/V_inner/V_{name}"
    base_dir = f"{region_base_dir_Vinner}/V_{name}"
    process_conv_output(
        conv_output=conv_output,
        base_dir=base_dir,
        min_x_values=min_x_values,
        kernel_height=20,
        kernel_width=100,
        stride_height=1,
        stride_width=1
    )
'''
