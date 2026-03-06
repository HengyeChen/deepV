import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json
import sys
import os
import tempfile

def create_image(data_points, size=(1000, 100)):
#    unique_data_points = set(data_points)
    image = np.zeros(size)
    for x, y in data_points:
#    for x, y in unique_data_points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            image[x, y] += 1
    return image

def get_data_points(file_path, start, end, return_min_x=False):
    data = pd.read_csv(file_path, sep='\t', header=None)
    filtered_data = data[(data[1] >= start) & (data[1] <= end)]
#    data_points = list(zip(filtered_data[1], filtered_data[2]))
    data_points = []
    for _, row in filtered_data.iterrows():
        x, y, count = row[1], row[2], row[3]
        data_points.extend([(x, y)] * count)
    min_x = min(filtered_data[1])
    adjusted_data_points = [(x - min_x, y) for x, y in data_points]
    if return_min_x:
        return adjusted_data_points, min_x
    return adjusted_data_points

def create_label_image(size=(1000, 100), slope=-2):
    label_image = np.zeros(size)
    for x in range(size[0]):
        y = int(slope * x)
        if 0 <= y < size[1]:
            label_image[x, y] = 1
    return label_image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_images(file_path, start, end, step, overlap):
    images = []
    min_x_values = []
    skipped_ranges = []

    while start + step <= end:
        current_end = start + step
        try:
            data_points, min_x = get_data_points(file_path, start, current_end, return_min_x=True)
            if not data_points:
                print(f"No data points found for range {start}-{current_end}. Skipping...")
                skipped_ranges.append((start, current_end))
                start = current_end - overlap
                continue

            image = create_image(data_points)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
            images.append(image)
            min_x_values.append(min_x)
        except ValueError as e:
            print(f"range {start}-{current_end}: {e}")
            skipped_ranges.append((start, current_end))
        start = current_end - overlap
    if images:
        return np.vstack(images), min_x_values
    else:
        print("No images were generated.")
        return np.array([]), min_x_values

def save_conv1_output_to_csv(output_csv_path, conv_output, min_x_values):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["start_y", "start_x", "end_y", "end_x", "conv1_channel_value", "image", "region_start", "region_end"])

        kernel_height, kernel_width = 50, 100
        stride_height, stride_width = 1, 1

        for i, output in enumerate(conv_output):
            conv1_channel = output[:, :, 0]
            conv1_positions = np.argwhere(conv1_channel)

            input_positions = []
            for pos in conv1_positions:
                output_y, output_x = pos
                start_y = output_y * stride_height + min_x_values[i]
                start_x = output_x * stride_width
                end_y = start_y + kernel_height
                end_x = start_x + kernel_width
                conv1_channel_value = conv1_channel[output_y, output_x]
                input_positions.append((start_y, start_x, end_y, end_x, conv1_channel_value))

            region_start = min_x_values[i]
            region_end = min_x_values[i] + (1000 * stride_height)

            for region in input_positions:
                writer.writerow([*region, i, region_start, region_end])

def filter_conv1_sigmoid(conv1_csv_path, output_csv_path, filter_csv_path, threshold=0.95):
    df = pd.read_csv(conv1_csv_path, sep='\t')    
    values = df['conv1_channel_value'].values
    values = values / df['conv1_channel_value'].mean()
    log_values = np.log1p(values)
    sigmoid_values = sigmoid(log_values)
    
    df['sigmoid_value'] = sigmoid_values
    df.to_csv(output_csv_path, sep='\t', index=False)
    
    significant_rows = df[df['sigmoid_value'] > threshold]
    significant_rows.to_csv(filter_csv_path, sep='\t', index=False)

def process_tsv_in_memory(file_path, start_val, end_val):
    df = pd.read_csv(file_path, sep="\t", header=None)
    df_filtered = df[(df[1] >= start_val) & (df[1] <= end_val)]
    
    temp_tsv = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.tsv', 
        delete=False,
        dir=os.getcwd()
    )
    
    df_filtered.to_csv(temp_tsv, sep="\t", index=False, header=False)
    temp_tsv.close()
    
    return os.path.abspath(temp_tsv.name)

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

#file_path = config["file_path"]
file_path_origin = config["file_path"]
start = config["start"]
end = config["end"]
step = config["step"]
overlap = config["overlap"]
conv1_csv_path = config["conv1_csv_path"]
output_csv_path = config["output_csv_path"]
filter_csv_path = config["filter_csv_path"]
ensure_directories_exist(conv1_csv_path, output_csv_path, filter_csv_path)

file_path = process_tsv_in_memory(file_path_origin, start, end)

images, min_x_values = generate_images(file_path, start, end, step, overlap)

model = Sequential([
    Conv2D(1, (50, 100), strides=(1, 1), activation='relu', input_shape=(1000, 100, 1), name='conv1'),
])

# 设置卷积核
conv_layer = model.get_layer('conv1')
kernel_image = np.loadtxt("kernel_standard_y30.scale.csv", delimiter="\t")
kernel_image *= 1000
fixed_kernel = np.zeros((50, 100, 1, 1))
fixed_kernel[:, :, 0, 0] = kernel_image
fixed_bias = np.zeros(1)

conv_layer.set_weights([fixed_kernel, fixed_bias])
conv_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer.output)

# Check if images were generated
if len(images) == 0:
    print(f"No images generated for {config_file_path}, skipping processing")
    sys.exit(0)

conv_output = conv_model.predict(images)

# 反推
save_conv1_output_to_csv(conv1_csv_path, conv_output, min_x_values)

# 过滤
filter_conv1_sigmoid(conv1_csv_path, output_csv_path, filter_csv_path, threshold=0.90)
