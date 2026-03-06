import numpy as np
import pandas as pd
import csv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json
import sys
from pathlib import Path as PathlibPath

def filter_y(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['detected_point_absolute'] != 'no_detected_point']
    df['detected_point_absolute'] = df['detected_point_absolute'].apply(eval)
    df = df[df['detected_point_absolute'].apply(lambda point: point[1] >= 10)]
    return df

def round_detected_points(filtered_df):
    def round_point(point):
        x, y = point
        return (round(x), round(y))

    def safe_eval(value):
        try:
            if isinstance(value, str):
                return eval(value)
            else:
                return value
        except Exception as e:
            print(f"解析失败的值: {value}")  # 输出触发异常的值
            raise e

    filtered_df['detected_point_absolute'] = (
        filtered_df['detected_point_absolute']
        .apply(safe_eval)  # 确保字符串被正确解析
        .apply(round_point)  # 对解析后的点进行四舍五入
    )
    return filtered_df

def detecte_point_expend(filtered_df, bed_file_path, output_csv_path="detecte_point_expend.csv"):
    csv_data = []

    for idx, row in filtered_df.iterrows():
        x_abs, y_abs = row['detected_point_absolute']

        # 20×20范围计算（中心向两侧各扩展10单位，总长度20）
        local_x_start = x_abs - 50
        local_x_end = x_abs + 50
        local_y_start = y_abs - 10
        local_y_end = y_abs + 10

        csv_data.append([
            row['image_index'],          # 原始图像索引（image_pre）
            row['detected_point_absolute'],  # 原始绝对坐标点
            local_x_start,               # 局部X起始
            local_x_end,                 # 局部X结束
            local_y_start,               # 局部Y起始
            local_y_end                  # 局部Y结束
        ])

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow([
            "image_pre",
            "detected_point_absolute",
            "local_x_start",
            "local_x_end",
            "local_y_start",
            "local_y_end"
        ])
        writer.writerows(csv_data)
    result_df = pd.DataFrame(
        csv_data,
        columns=[
            "image_pre",
            "detected_point_absolute",
            "local_x_start",
            "local_x_end",
            "local_y_start",
            "local_y_end"
        ]
    )
    return result_df

def process_csv(expend_df):
    # 提取需要的列并更名
    df = expend_df[['local_x_start', 'local_x_end']].copy()
    df = df.rename(columns={
        'local_x_start': 'start_y',
        'local_x_end': 'end_y'
    })

    # 去重并按start_y排序
    df = df.drop_duplicates(subset=['start_y', 'end_y'])
    df = df.sort_values(by="start_y").reset_index(drop=True)

    # 处理空数据框情况
    if df.empty:
        result = df.copy()
        result['range'] = 0
        result['start_x'] = 0
        result['end_x'] = 100
        return result

    # 初始化分组，从0开始
    df['group'] = 0
    current_group = 0

    # 仅比较当前行与上一行（原始行比较）
    for i in range(1, len(df)):
        # 上一行的end_y
        prev_end = df.loc[i-1, 'end_y']
        # 当前行的start_y
        current_start = df.loc[i, 'start_y']

        # 如果当前行的start_y小于上一行的end_y，则合并到同一组
        if current_start < prev_end:
            df.loc[i, 'group'] = current_group
        else:
            # 否则创建新组
            current_group += 1
            df.loc[i, 'group'] = current_group

    # 按组聚合，取每组最小的start_y和最大的end_y
    result = df.groupby('group').agg({
        'start_y': 'min',
        'end_y': 'max'
    }).reset_index(drop=True)

    # 计算range和添加x坐标
    result['range'] = result['end_y'] - result['start_y']
    result['start_x'] = 0
    result['end_x'] = 100

    return result

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

def filter_csv_by_sigmoid(csv_path, sigmoid_output_dir, threshold=0.95):
    df = pd.read_csv(csv_path, sep='\t')
    values = df['conv3_channel_value'].values
    values = values / df['conv3_channel_value'].mean()
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

def generate_image_batches(file_path, result, max_range, chunk_size=500, batch_size=500):
    if result.empty:
        return

    bed_all = pd.read_csv(file_path, sep='\t', header=None, usecols=[1, 2])
    result_reset = result.reset_index(drop=True)
    n = len(result_reset)
    step = max(1, int(chunk_size))
    batch_limit = max(1, int(batch_size))
    images = []
    min_x_values = []
    image_index = 0
    batch_start_index = 0

    for start_idx in range(0, n, step):
        end_idx = min(start_idx + step, n)
        chunk = result_reset.iloc[start_idx:end_idx]
        chunk_min = float(chunk['start_y'].min())
        chunk_max = float(chunk['end_y'].max())
        bed_chunk = bed_all[(bed_all[1] >= chunk_min) & (bed_all[1] <= chunk_max)]

        if bed_chunk.empty:
            continue

        for _, row in chunk.iterrows():
            start = row['start_y']
            end = row['end_y']
            if pd.isna(start) or pd.isna(end):
                print(f"Invalid start or end value in row: {row}")
                continue
            subset = bed_chunk[(bed_chunk[1] >= start) & (bed_chunk[1] <= end)]
            if subset.empty:
                print(f"No data points found for range {start}-{end}. Skipping...")
                continue

            data_points = list(zip(subset[1], subset[2]))
            min_x = int(subset[1].min())
            adjusted_data_points = [(int(x - min_x), int(y)) for x, y in data_points]

            image = create_image(adjusted_data_points, size=(int(row['range']), 100))

            pad_height = max_range - image.shape[0]
            pad_top = 0
            pad_bottom = pad_height
            padded_image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

            padded_image = np.expand_dims(padded_image, axis=0)
            padded_image = np.expand_dims(padded_image, axis=-1)

            images.append(padded_image)
            min_x_values.append(min_x)
            image_index += 1

            if len(images) >= batch_limit:
                yield np.vstack(images), min_x_values, batch_start_index
                images = []
                min_x_values = []
                batch_start_index = image_index

    if images:
        yield np.vstack(images), min_x_values, batch_start_index

def get_y_values(expend_df):
    unique_y_set = set()    
    for _, row in expend_df.iterrows():
        # 获取当前行的local_y_start和local_y_end（确保为整数，避免浮点误差）
        y_start = int(row["local_y_start"])
        y_end = int(row["local_y_end"])
        y_range = range(y_start, y_end + 1)
        filtered_y = [y for y in y_range if y >= 10]
        unique_y_set.update(filtered_y)
    y_values = sorted(list(unique_y_set))
    count = len(y_values)
    return y_values, count

def format_y_values(y_values):
    formatted_y_list = [f"y{int(value)}" for value in y_values]
    return formatted_y_list

def save_conv3_output_to_csv(base_dir, conv_output, min_x_values, kernel_height=10, kernel_width=20, stride_height=1, stride_width=1):
    image_dir = os.path.join(base_dir, "conv3_image")
    sigmoid_dir = os.path.join(base_dir, "conv3_image_sigmoid")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sigmoid_dir, exist_ok=True)
    
    for i, output in enumerate(conv_output):
        conv3_channel = output[:, :, 0]
        conv3_positions = np.argwhere(conv3_channel)

        input_positions = []
        for pos in conv3_positions:
            output_y, output_x = pos
            start_y = output_y * stride_height + min_x_values[i]
            start_x = output_x * stride_width
            end_y = start_y + kernel_height
            end_x = start_x + kernel_width
            conv3_channel_value = conv3_channel[output_y, output_x]
            input_positions.append((start_y, start_x, end_y, end_x, conv3_channel_value))

        region_start = min_x_values[i]
        region_end = min_x_values[i] + (1000 * stride_height)

        image_csv_path = os.path.join(image_dir, f"image_{i}.csv")
        with open(image_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(["start_y", "start_x", "end_y", "end_x", "conv3_channel_value", "image", "region_start", "region_end"])
            for region in input_positions:
                writer.writerow([*region, i, region_start, region_end])
        filter_csv_by_sigmoid(image_csv_path)

def generate_fixed_kernel_new(kernel_image,
                          y_values,
                          final_shape=(100, 100)):
    results = {}
    for y in y_values:
        sub_matrix = kernel_image
        final_matrix = np.zeros(final_shape)
        final_matrix[:, :] = sub_matrix
        results[f"final_matrix_y{y}"] = final_matrix

    # 修改每个矩阵的数值
    for key, matrix in results.items():
        matrix[matrix > 0] = 0

        x = int(key.split('_y')[-1])
        y = 50
        additional_points = calculate_points_inner_tranxy({'point': (x, y)}, x3=100)
        for region_name, points in additional_points.items():
            polygon = Path(points)
            # 遍历矩阵中的每个点，判断是否在多边形内
            for i in range(matrix.shape[0]):  # 遍历 y 方向
                for j in range(matrix.shape[1]):  # 遍历 x 方向
                    if polygon.contains_point((j, i)):  # 注意 (x, y) 顺序
                        matrix[i, j] = 1  # 将在多边形内的点赋值为5

    num_kernels = len(y_values)
    fixed_kernel = np.zeros((final_shape[0], final_shape[1], 1, num_kernels))
    for i, y in enumerate(y_values):
        fixed_kernel[:, :, 0, i] = results[f"final_matrix_y{y}"]
    return fixed_kernel, num_kernels, results

    
def process_sigmoid_files(input_base_dir, output_file, channel_name_y_values, threshold=0.7):
    all_filtered_rows = pd.DataFrame()
    for channel_name in channel_name_y_values:
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

def calculate_points_inner_tranxy(point, x3=100):
    x, y = point['point']
    slope_left = -1 / -2.0
    slope_right = -1 / 2.0
    # 计算斜率为 slope_left 的直线与 x=100 的交点
    y_inner_left = y + slope_left * (x3 - x)
    point_inner_left = (x3, y_inner_left)

    # 计算斜率为 slope_right 的直线与 x=100 的交点
    y_inner_right = y + slope_right * (x3 - x)
    point_inner_right = (x3, y_inner_right)

    point = (x, y)
    points_inner = [point, (x3, y_inner_left), (x3, y_inner_right)]
    return {
        "points_inner": points_inner
    }

def process_conv_output(conv_output, base_dir, min_x_values, channel_name_y_values, kernel_height=20, kernel_width=100, stride_height=1, stride_width=1, image_offset=0):
    for i, output in enumerate(conv_output):
        image_index = image_offset + i
        for channel_idx, channel_name in enumerate(channel_name_y_values):
            conv_channel = output[:, :, channel_idx]
            conv_positions = np.argwhere(conv_channel)

            input_positions = []
            for pos in conv_positions:
                output_y, output_x = pos
                start_y = output_y * stride_height + min_x_values[i]
                start_x = output_x * stride_width
                end_y = start_y + kernel_height
                end_x = start_x + kernel_width
                x_value = (start_y + end_y) / 2
                conv_channel_value = conv_channel[output_y, output_x]

                input_positions.append((start_y, start_x, end_y, end_x, x_value, conv_channel_value))

            region_start = min_x_values[i]
            region_end = min_x_values[i] + (1000 * stride_height)

            image_csv_dir = os.path.join(base_dir, f"conv3_image_{channel_name}")
            sigmoid_output_dir = os.path.join(base_dir, f"conv3_image_sigmoid_{channel_name}")
            os.makedirs(image_csv_dir, exist_ok=True)
            os.makedirs(sigmoid_output_dir, exist_ok=True)

            image_csv_path = os.path.join(image_csv_dir, f"image_{image_index}.csv")
            ensure_directories_exist(image_csv_path)
            with open(image_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(["start_y", "start_x", "end_y", "end_x", "x_value", "conv3_channel_value", "image", "region_start", "region_end", "kernel_y_value"])
                for region in input_positions:
                    kernel_y_value = int(channel_name[1:])
                    writer.writerow([*region, image_index, region_start, region_end, kernel_y_value])

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

file_path_origin = config["file_path"]
detecte_point_path = config["detecte_point_path"]
detecte_point_expend_path = config["detecte_point_expend_path"]
detecte_point_merge_path = config["detecte_point_merge_path"]
region_base_dir_V = config["region_base_dir_V"]

ensure_directories_exist(detecte_point_expend_path)
ensure_directories_exist(detecte_point_merge_path)

filtered_df = filter_y(detecte_point_path)
filtered_df = round_detected_points(filtered_df)
expend_df = detecte_point_expend(filtered_df, file_path_origin, detecte_point_expend_path)
result = process_csv(expend_df)
result.to_csv(detecte_point_merge_path, sep='\t', index=False)

#result = pd.read_csv(detecte_point_merge_path, sep='\t')
expend_df = pd.read_csv(detecte_point_expend_path, sep='\t')
y_values, count_y_values = get_y_values(expend_df)
channel_name_y_values = format_y_values(y_values)

chunk_size = int(config.get("chunk_size", 500))
batch_size = 6000
max_range = result['range'].max()

# 设置卷积核
kernel_height=100
kernel_width=100
#kernel_image = np.loadtxt("/home/nwh/software/temp/loMNase_K562/pre_test_2/16800-17800kb/kernel/right.amplified_probabilities_square_scaled.txt", delimiter="\t")
kernel_image = np.zeros((kernel_height, kernel_width))
fixed_kernel, num_kernels, results = generate_fixed_kernel_new(kernel_image, y_values, final_shape=(kernel_height, kernel_width))
#plot_results_heatmaps(results)

model = Sequential([
    Conv2D(count_y_values, (kernel_height, kernel_width), strides=(1, 1),activation='relu', input_shape=(max_range, 100, 1), name='conv3'),
])
conv_layer = model.get_layer('conv3')
fixed_bias = np.zeros(num_kernels)
conv_layer.set_weights([fixed_kernel, fixed_bias])
conv_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer.output)
base_dir = region_base_dir_V
for images, min_x_values, batch_start_index in generate_image_batches(
    file_path_origin,
    result,
    max_range,
    chunk_size=chunk_size,
    batch_size=batch_size
):
    conv_output = conv_model.predict(images)

    process_conv_output(
        conv_output=conv_output,
        base_dir=base_dir,
        min_x_values=min_x_values,
        channel_name_y_values=channel_name_y_values,
        kernel_height=100,
        kernel_width=100,
        stride_height=1,
        stride_width=1,
        image_offset=batch_start_index
    )
    del images, conv_output
