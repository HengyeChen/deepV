import os
import pandas as pd
import numpy as np
import sys
import json

def merge_conv3_images(conv3_image_dir):
    merged_df = pd.DataFrame()
    for entry in os.scandir(conv3_image_dir):
        if entry.is_dir() and entry.name.startswith("conv3_image_sigmoid_y"):
            for file_name in os.listdir(entry.path):
                if file_name.startswith("image_") and file_name.endswith(".csv"):
                    file_path = os.path.join(entry.path, file_name)
                    df = pd.read_csv(file_path, sep='\t')
                    df = df[['x_value', 'kernel_y_value', 'conv3_channel_value']]
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    merged_df = merged_df.sort_values(
        by=['x_value', 'kernel_y_value'],
        ascending=[True, True],
        key=lambda x: pd.to_numeric(x, errors='coerce')
    ).reset_index(drop=True)
    
    return merged_df

def calculate_slope(x, y):
    return np.gradient(y, x)

def get_sorted_y_values(conv3_dir):
    y_values = []
    for entry in os.scandir(conv3_dir):
        if entry.is_dir() and entry.name.startswith("conv3_image_sigmoid_y"):
            suffix = entry.name.replace("conv3_image_sigmoid_y", "")
            if suffix.isdigit():
                y_values.append(int(suffix))
    y_values.sort()
    return y_values

def get_slope(merged_df, y_values):
    slopes_data = []
    grouped = merged_df.groupby(['x_value'])

    for x_value, group in grouped:
        if isinstance(x_value, (tuple, list)):
            x_val = x_value[0]
        else:
            x_val = x_value

        y = group['kernel_y_value'].values
        conv3_channel_value = group['conv3_channel_value'].values
        y_to_value = dict(zip(y, conv3_channel_value))
        filled_values = [y_to_value.get(y_val, 0) for y_val in y_values]
        slopes = calculate_slope(y_values, filled_values)
        # 去掉 image 列，只保留 x_value 和 slopes
        slopes_data.append([x_val] + list(slopes))
    
    column_names = ['x'] + [f"y_{val}" for val in y_values]
    slopes_df = pd.DataFrame(slopes_data, columns=column_names)
    return slopes_df

def calculate_slope_diff(filtered_slopes_df, output_file):
    y_columns = sorted([col for col in filtered_slopes_df.columns if col.startswith('y_')],
                       key=lambda x: int(x.split('_')[1]))

    differences_df = filtered_slopes_df[['x']].copy()
    for i in range(len(y_columns) - 1):
        col_current = y_columns[i]
        col_next = y_columns[i + 1]
        differences_df[col_next] = (filtered_slopes_df[col_next] - filtered_slopes_df[col_current]).round(2)

    differences_df.to_csv(output_file, sep='\t', index=False)
    return differences_df

def slope_y_value(slope_diff_df):
    y_columns = [col for col in slope_diff_df.columns if col.startswith('y_')]

    new_column = []
    for _, row in slope_diff_df.iterrows():
        y_values = row[y_columns].values
        # 计算绝对值的最大值并取一半
        max_abs_value = np.max(np.abs(y_values))
        threshold = max_abs_value / 2

        found = None
        for col, value in zip(y_columns, y_values):
            if value < 0 and abs(value) >= threshold:
                found = int(col.split('_')[1])
                break
        new_column.append(found)

    slope_diff_df['slope_y_value'] = pd.Series(new_column, dtype='Int64')
    return slope_diff_df

def final_point_filter(point_filter_file, slope_diff_df, merged_df):
    filtered_point_filter_df = pd.read_csv(point_filter_file, sep="\t")

    final_filtered_rows = []
    grouped = filtered_point_filter_df.groupby(['image', 'x_value'])
    for (image, x_value), group in grouped:
        slope_y_value = slope_diff_df.loc[(slope_diff_df['x'] == x_value), 'slope_y_value']
        if slope_y_value.empty:
            continue  # 如果没有对应的 slope_y_value，跳过

        slope_y_value = slope_y_value.iloc[0]  # 获取对应的 slope_y_value

        # 保留 y_value 在 slope_y_value ± 5 范围内的行
#        filtered_group = group[(group['y_value'] >= slope_y_value - 5) & (group['y_value'] <= slope_y_value + 5)]
        filtered_group = group[(group['y_value'] >= slope_y_value - 10) & (group['y_value'] <= slope_y_value + 10)]

        if not filtered_group.empty:
            final_filtered_rows.append(filtered_group)
        else:
            print(f"Warning: No rows retained for image {image} and x_value {x_value}")

    if final_filtered_rows:
        result_df = pd.concat(final_filtered_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=filtered_point_filter_df.columns)  # 返回空 DataFrame

    result_df = result_df.merge(
        merged_df[['x_value', 'kernel_y_value', 'conv3_channel_value']],
        how='left',
        left_on=['x_value', 'y_value'],
        right_on=['x_value', 'kernel_y_value']
    ).drop_duplicates()
    result_df = result_df.drop(columns=['kernel_y_value'])

    # 进一步过滤
    # 1. 只保留 y_value 在 [10, 60] 范围内的行
    result_df = result_df[(result_df['y_value'] >= 10) & (result_df['y_value'] <= 80)]
    # 2. 如果有多行的 image 和 point 相同，将 kernel 列的值用 ',' 连接，并只保留一行
    result_df = result_df.groupby(['image', 'point', 'conv3_channel_value'], as_index=False).agg({
        'x_value': 'first',
        'y_value': 'first',
        'kernel': lambda x: ','.join(x)
    })
    return result_df

def final_point_filter_test(point_filter_file, slope_diff_df, merged_df):
    point_filter_df = pd.read_csv(point_filter_file, sep="\t")

    # 只保留 point_filter_file 中 image 和 x_value 在 slope_diff_df 中存在的行
    valid_x_values = slope_diff_df[['image', 'x']].drop_duplicates()
    filtered_point_filter_df = point_filter_df.merge(valid_x_values, how='inner', left_on=['image', 'x_value'], right_on=['image', 'x'])
    filtered_point_filter_df = filtered_point_filter_df.drop(columns=['x'])  # 删除多余的列

    # 打印过滤步骤 1 的结果
    print("After filtering valid_x_values (Step 1):")
    print(filtered_point_filter_df[filtered_point_filter_df['image'] == 89])

    final_filtered_rows = []
    grouped = filtered_point_filter_df.groupby(['image', 'x_value'])
    for (image, x_value), group in grouped:
        slope_y_value = slope_diff_df.loc[(slope_diff_df['image'] == image) & (slope_diff_df['x'] == x_value), 'slope_y_value']
        if slope_y_value.empty:
            continue  # 如果没有对应的 slope_y_value，跳过

        slope_y_value = slope_y_value.iloc[0]  # 获取对应的 slope_y_value

        # 保留 y_value 在 slope_y_value ± 10 范围内的行
        filtered_group = group[(group['y_value'] >= slope_y_value - 10) & (group['y_value'] <= slope_y_value + 10)]

        # 打印过滤步骤 2 的结果
        if image == 89:
            print(f"After filtering slope_y_value ± 10 (Step 2) for x_value={x_value}:")
            print(filtered_group)

        if not filtered_group.empty:
            final_filtered_rows.append(filtered_group)
        else:
            print(f"Warning: No rows retained for image {image} and x_value {x_value}")

    if final_filtered_rows:
        result_df = pd.concat(final_filtered_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=filtered_point_filter_df.columns)  # 返回空 DataFrame

    # 打印过滤步骤 3 的结果
    print("After concatenating filtered groups (Step 3):")
    print(result_df[result_df['image'] == 89])

    result_df = result_df.merge(
        merged_df[['image', 'x_value', 'kernel_y_value', 'conv3_channel_value']],
        how='left',
        left_on=['image', 'x_value', 'y_value'],
        right_on=['image', 'x_value', 'kernel_y_value']
    ).drop_duplicates()
    result_df = result_df.drop(columns=['kernel_y_value'])

    # 打印过滤步骤 4 的结果
    print("After merging with merged_df (Step 4):")
    print(result_df[result_df['image'] == 89])

    # 进一步过滤
    # 1. 只保留 y_value 在 [10, 80] 范围内的行
    result_df = result_df[(result_df['y_value'] >= 10) & (result_df['y_value'] <= 80)]

    # 打印过滤步骤 5 的结果
    print("After filtering y_value in [10, 80] (Step 5):")
    print(result_df[result_df['image'] == 89])

    # 2. 如果有多行的 image 和 point 相同，将 kernel 列的值用 ',' 连接，并只保留一行
    result_df = result_df.groupby(['image', 'point', 'conv3_channel_value'], as_index=False).agg({
        'x_value': 'first',
        'y_value': 'first',
        'kernel': lambda x: ','.join(x)
    })

    # 打印过滤步骤 6 的结果
    print("After grouping and aggregating (Step 6):")
    print(result_df[result_df['image'] == 89])

    return result_df

def final_point(final_point_all_df, slope_diff_df):
    filtered_rows = []
    grouped = final_point_all_df.groupby(['image', 'x_value'])
    for (image, x_value), group in grouped:
        slope_y_value = slope_diff_df.loc[
            (slope_diff_df['image'] == image) & (slope_diff_df['x'] == x_value), 'slope_y_value'
        ]
        if slope_y_value.empty:
            continue
        slope_y_value = slope_y_value.iloc[0]
        filtered_group = group[group['y_value'] == slope_y_value]

        if not filtered_group.empty:
            filtered_rows.append(filtered_group)

    if filtered_rows:
        result_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=final_point_all_df.columns)
    return result_df

def process_point_and_slope(point_df, slope_diff_df):
    merged_df = point_df.merge(
        slope_diff_df[['x', 'slope_y_value']],
        how='left',
        left_on=['x_value'],
        right_on=['x']
    )

    # 打印 slope_y_value 为 NaN 的行
#    print("Rows with NaN in 'slope_y_value':")
#    print(merged_df[merged_df['slope_y_value'].isna()])
    # 过滤掉 slope_y_value 列中包含 NaN 的行
#    merged_df = merged_df[merged_df['slope_y_value'].notna()]
    
    merged_df['y_value'] = merged_df['slope_y_value'].astype(int)
    def update_point(point, slope_y_value):
        if pd.notna(point):
            x, _ = eval(point)
            return f"({x}, {int(slope_y_value)})"
        return point
    merged_df['point'] = merged_df.apply(
        lambda row: update_point(row['point'], row['slope_y_value']),
        axis=1
    )
#    merged_df = merged_df.drop(columns=['x', 'slope_y_value', 'conv3_channel_value'])
    merged_df = merged_df.drop(columns=['x', 'slope_y_value'])
   

    # 对于fix y后的数值，只保留 y_value 在 [10, 60] 范围内的行
    merged_df = merged_df[(merged_df['y_value'] >= 10) & (merged_df['y_value'] <= 80)]
    # 去掉 kernel 列的值，去掉 kernel 列的值
    merged_df = merged_df.drop(columns=['kernel'])
    merged_df = merged_df.drop_duplicates(subset=['image', 'point'])
    return merged_df

def process_point_and_slope_test(point_df, slope_diff_df):
    # 合并 point_df 和 slope_diff_df
    merged_df = point_df.merge(
        slope_diff_df[['image', 'x', 'slope_y_value']],
        how='left',
        left_on=['image', 'x_value'],
        right_on=['image', 'x']
    )

    # 打印合并后的结果中 image == 89 的行
    print("After merging point_df and slope_diff_df (Step 1):")
    print(merged_df[merged_df['image'] == 89])

    # 打印 slope_y_value 为 NaN 的行
    print("Rows with NaN in 'slope_y_value' (Step 2):")
    print(merged_df[merged_df['slope_y_value'].isna()])

    # 转换 slope_y_value 为整数
    merged_df['y_value'] = merged_df['slope_y_value'].astype(int)

    # 打印转换 y_value 后的结果
    print("After converting 'slope_y_value' to 'y_value' (Step 3):")
    print(merged_df[merged_df['image'] == 89])

    # 更新 point 列
    def update_point(point, slope_y_value):
        if pd.notna(point):
            x, _ = eval(point)
            return f"({x}, {int(slope_y_value)})"
        return point

    merged_df['point'] = merged_df.apply(
        lambda row: update_point(row['point'], row['slope_y_value']),
        axis=1
    )

    # 打印更新 point 列后的结果
    print("After updating 'point' column (Step 4):")
    print(merged_df[merged_df['image'] == 89])

    # 删除多余的列
    merged_df = merged_df.drop(columns=['x', 'slope_y_value', 'conv3_channel_value'])

    # 打印删除多余列后的结果
    print("After dropping unnecessary columns (Step 5):")
    print(merged_df[merged_df['image'] == 89])

    # 只保留 y_value 在 [10, 80] 范围内的行
    merged_df = merged_df[(merged_df['y_value'] >= 10) & (merged_df['y_value'] <= 80)]

    # 打印过滤 y_value 范围后的结果
    print("After filtering 'y_value' in [10, 80] (Step 6):")
    print(merged_df[merged_df['image'] == 89])

    # 删除 kernel 列
    merged_df = merged_df.drop(columns=['kernel'])

    # 打印删除 kernel 列后的结果
    print("After dropping 'kernel' column (Step 7):")
    print(merged_df[merged_df['image'] == 89])

    # 去重
    merged_df = merged_df.drop_duplicates(subset=['image', 'point'])

    # 打印去重后的结果
    print("After dropping duplicates (Step 8):")
    print(merged_df[merged_df['image'] == 89])

    return merged_df

def ensure_directories_exist(*file_paths):
    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)

    post_conv2_point_filter_file = config["point_filter_file"]
    conv3_dir = config["conv3_dir"]
    slope_output_file = config["slope_output_file"]
    slope_diff_output_file = config["slope_diff_output_file"]
    final_point_fix_file = config["final_point_fix_file"]
    ensure_directories_exist(slope_output_file)

    y_values = get_sorted_y_values(conv3_dir)

    merged_df = merge_conv3_images(conv3_dir)
#    print(merged_df)
    slopes_df = get_slope(merged_df, y_values)
#    print(slopes_df)
    slopes_df.to_csv(slope_output_file, sep="\t", index=False)

    slope_diff_df = calculate_slope_diff(slopes_df, slope_diff_output_file)
    slope_diff_df = slope_y_value(slope_diff_df)
    slope_diff_df.to_csv(slope_diff_output_file, sep='\t', index=False)

    final_point_all_df = final_point_filter(post_conv2_point_filter_file, slope_diff_df, merged_df)
    print(final_point_all_df)

    merged_df = process_point_and_slope(final_point_all_df, slope_diff_df)
    print(merged_df)
    merged_df.to_csv(final_point_fix_file, sep="\t", index=False)
