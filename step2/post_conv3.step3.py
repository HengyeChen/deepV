import os
import pandas as pd
import numpy as np
import sys
import json
cluster_dir = "/home/nwh/software/temp/loMNase_K562/test_250728/scripts"
sys.path.insert(0, os.path.abspath(cluster_dir))
from cluster import assign_v_clusters

def build_counts_by_x(file_path):
    file_df = pd.read_csv(file_path, sep="\t", header=None, names=['chr', 'x', 'y', 'count'])
    if file_df.empty:
        return {}
    file_df = file_df.groupby(['x', 'y'], as_index=False)['count'].sum()
    counts_by_x = {}
    for x_value, group in file_df.groupby('x'):
        counts_by_x[int(x_value)] = dict(zip(group['y'].astype(int), group['count']))
    return counts_by_x

def resolve_point_xy(x_value, y_value, point):
    if pd.notna(x_value) and pd.notna(y_value):
        return int(x_value), int(y_value)
    point = point.strip()[1:-1]
    x_str, y_str = point.split(',')
    return int(x_str.strip()), int(y_str.strip())

def sum_slope_counts(counts_by_x, x_values, y_value):
    right_count = 0
    left_count = 0
    for x_val, delta in x_values:
        x_counts = counts_by_x.get(x_val)
        if not x_counts:
            continue
        right_count += x_counts.get(y_value + 2 * delta, 0)
        left_count += x_counts.get(y_value - 2 * delta, 0)
    return right_count, left_count

def has_nearby_reads(x_value, y_value, counts_by_x, window=3):
    for x_val in range(x_value - window, x_value + window + 1):
        x_counts = counts_by_x.get(x_val)
        if not x_counts:
            continue
        for y_val in range(y_value - window, y_value + window + 1):
            if y_val in x_counts:
                return True
    return False

def merge_conv4_V_inner_images(conv4_V_inner_image_dir):
    merged_df = pd.DataFrame()
    for file_name in os.listdir(conv4_V_inner_image_dir):
        file_path = os.path.join(conv4_V_inner_image_dir, file_name)
        if os.path.isfile(file_path) and file_name.startswith("image_") and file_name.endswith(".csv"):
            df = pd.read_csv(file_path, sep='\t')
            df = df[['image', 'x_value', 'kernel_y_value', 'conv3_channel_value']]
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

def add_conv4_V_inner_value(point_file_df, merged_df):
    merged_result = point_file_df.merge(
        merged_df[['image', 'x_value', 'kernel_y_value', 'conv3_channel_value']],
        how='left',
        left_on=['image', 'x_value', 'y_value'],
        right_on=['image', 'x_value', 'kernel_y_value']
    )
    merged_result = merged_result.rename(columns={'conv3_channel_value': 'conv4_V_inner_value'})
    merged_result = merged_result.drop(columns=['kernel_y_value'])
    return merged_result

def add_post_conv2_V_channel_info(point_file_df, local_max_df):
    merged_result = point_file_df.merge(
        local_max_df[['image', 'y_value', 'x_left', 'x_right', 'nearest_local_max_count']],
        how='left',
        on=['image', 'y_value']
    )
    # 只保留 nearest_local_max_count >= 2 的行
    # 先不用这个指标再做过滤
#    merged_result = merged_result[merged_result['nearest_local_max_count'] >= 2]
    return merged_result

def add_point_Vchannel_score(point_channel_df):
    def calculate_score(row):
        x_value = row['x_value']
        x_left = row['x_left']
        x_right = row['x_right']
        left_diff = abs(x_value - x_left) if not pd.isna(x_left) else np.inf
        right_diff = abs(x_value - x_right) if not pd.isna(x_right) else np.inf
        return min(left_diff, right_diff)
    point_channel_df['point_Vchannel_score'] = point_channel_df.apply(calculate_score, axis=1)
    return point_channel_df

def add_conv2_V_inner_score(point_channel_df, v_inner_point_file, region_base_dir_Vinner):
    v_inner_point_df = pd.read_csv(v_inner_point_file, sep="\t")
    point_channel_df = point_channel_df.merge(
        v_inner_point_df[['image', 'point']],
        how='left',
        on=['image', 'point']
    )
    kernel_list = ['standard', 'left', 'right', 'middle']
    for kernel in kernel_list:
        conv2_scores = []
        for _, row in point_channel_df.iterrows():
            image = row['image']
            y_value = row['y_value']
            x_value = row['x_value']
            if pd.isna(x_value):
                conv2_scores.append(np.nan)
                continue
            csv_path = os.path.join(
                region_base_dir_Vinner,
                f"V_{kernel}/conv2_image_sigmoid_y{int(y_value)}/image_{int(image)}.sigmoid.csv"
            )
            if not os.path.exists(csv_path):
                conv2_scores.append(np.nan)
                continue
            conv2_df = pd.read_csv(csv_path, sep="\t")
            # 查找 start_y + 10 等于 x_value 的行
            matched_row = conv2_df[conv2_df['start_y'] + 10 == x_value]
            if not matched_row.empty:
                conv2_scores.append(matched_row.iloc[0]['conv2_channel_value'])
            else:
                conv2_scores.append(np.nan)
        # 将当前 kernel 的 conv2_V_inner_score 添加为新列
        point_channel_df[f'conv2_Vinner_value_kernel_{kernel}'] = conv2_scores
    return point_channel_df

def get_kernel_max_values(point_channel_df):
    kernel_columns = [
        'conv2_Vinner_value_kernel_standard',
        'conv2_Vinner_value_kernel_left',
        'conv2_Vinner_value_kernel_right',
        'conv2_Vinner_value_kernel_middle'
    ]
    result_rows = []
    grouped = point_channel_df.groupby('image')
    for image, group in grouped:
        for kernel_col in kernel_columns:
            max_value = group[kernel_col].max()
            if pd.notna(max_value):
                max_row = group[group[kernel_col] == max_value].iloc[0]
                result_rows.append({
                    'image': image,
                    'point': max_row['point'],
                    'x_value': max_row['x_value'],
                    'y_value': max_row['y_value'],
                    'conv4_Vinner_value': max_row['conv4_V_inner_value'],
                    'kernel': kernel_col.split('_')[-1],  # 提取 kernel 名称
                    'conv2_Vinner_value': max_value
                })
    result_df = pd.DataFrame(result_rows)
    return result_df

def choose_kernel(point_kernel_df):
    result_rows = []
    grouped = point_kernel_df.groupby('image')
    for image, group in grouped:
        max_value = group['conv2_Vinner_value'].max()
        max_rows = group[group['conv2_Vinner_value'] == max_value]
        if len(max_rows) > 1:
            print(f"Image {image} has multiple rows with the same max_value ({max_value}):")
            print(max_rows)
        result_rows.append(max_rows)
    result_df = pd.concat(result_rows, ignore_index=True)
    return result_df

def add_conv2_Vchannel_value(point_kernel_df, region_base_dir_Vchannel):
    conv2_Vchannel_values = []
    for _, row in point_kernel_df.iterrows():
        image = row['image']
        y_value = row['y_value']
        x_value = row['x_value']
        # 计算 x_value - 10
        target_start_y = x_value - 10

        csv_path = os.path.join(
            region_base_dir_Vchannel,
            f"conv2_image_sigmoid_y{int(y_value)}/image_{int(image)}.sigmoid.csv"
        )
        if not os.path.exists(csv_path):
            conv2_Vchannel_values.append(None)
            continue
        conv2_df = pd.read_csv(csv_path, sep="\t")
        # 查找 start_y 等于 target_start_y 的行
        matched_row = conv2_df[conv2_df['start_y'] == target_start_y]
        # 获取 conv2_channel_value 的值
        if not matched_row.empty:
            conv2_Vchannel_values.append(matched_row.iloc[0]['conv2_channel_value'])
        else:
            conv2_Vchannel_values.append(None)
    point_kernel_df['conv2_Vchannel_value'] = conv2_Vchannel_values
    return point_kernel_df

def add_nested_column_onenested(df):
    def check_slope(group):
        points = group[['x_value', 'y_value']].values
        nested_labels = [None] * len(points)
        label_counter = 1

        for i in range(len(points)):
            if nested_labels[i] is not None:
                continue  # 已经标注过的点跳过
            nested_labels[i] = f"nested_{label_counter}"
            x1, y1 = points[i]

            for j in range(i + 1, len(points)):
                if nested_labels[j] is not None:
                    continue  # 已经标注过的点跳过
                x2, y2 = points[j]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else None
                if slope is not None and (-2.5 <= slope <= -1.5 or 1.5 <= slope <= 2.5):
                    nested_labels[j] = f"nested_{label_counter}"

            label_counter += 1

        return pd.Series(nested_labels, index=group.index)
    # 按照 image 列分组，并对每组应用斜率检查逻辑
    df['nested'] = df.groupby('image', group_keys=False).apply(check_slope)
    return df

def add_nested_column(df):
    def check_slope(group):
        points = group[['x_value', 'y_value']].values
        nested_labels = [set() for _ in range(len(points))]  # 使用集合存储每个点的 nested 编号
        label_counter = 1

        for i in range(len(points)):
            x1, y1 = points[i]

            for j in range(i + 1, len(points)):
                x2, y2 = points[j]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else None
                if slope is not None and (-2.5 <= slope <= -1.5 or 1.5 <= slope <= 2.5):
                    # 将当前编号添加到两个点的集合中
                    nested_labels[i].add(f"nested_{label_counter}")
                    nested_labels[j].add(f"nested_{label_counter}")

            label_counter += 1

        # 将集合转换为逗号分隔的字符串
        nested_labels = [",".join(sorted(labels)) if labels else None for labels in nested_labels]
        return pd.Series(nested_labels, index=group.index)

    # 按照 image 列分组，并对每组应用斜率检查逻辑
    df['nested'] = df.groupby('image', group_keys=False).apply(check_slope)
    return df

def merge_nested_to_kernel(point_df, point_kernel_df):
    merged_df = point_kernel_df.merge(
        point_df[['image', 'point', 'nested']],
        how='left',
        on=['image', 'point']
    )
    return merged_df

def process_nested_groups(point_kernel_df):
    df1 = []  # 用于存储符合条件的行（df_nested）
    df2 = []  # 用于存储不符合条件的行

    grouped = point_kernel_df.groupby('image')

    for image, group in grouped:
        # 将 nested 列以逗号分隔转换为列表
        group['nested_list'] = group['nested'].fillna("").apply(lambda x: x.split(",") if x else [])

        # 获取所有行的 nested_list 的交集
        all_nested_values = set.intersection(*map(set, group['nested_list'])) if not group['nested_list'].empty else set()

        if all_nested_values:  # 如果交集不为空
            # 如果某值在所有行的 nested_list 中都出现
            df1.append(group)
        elif group['nested'].isna().all():
            # 如果所有行的 nested 列都没有值
            if group['point'].nunique() == 1:
                # 如果 point 列的值都相等，只保留一行
                df2.append(group.iloc[[0]])
            else:
                # 如果 point 列的值有不等的，保留所有行
                df2.append(group)
        else:
            # 如果交集为空，则该组写入 df2
            df2.append(group)
    # 将结果合并为 DataFrame
    df1 = pd.concat(df1, ignore_index=True) if df1 else pd.DataFrame(columns=point_kernel_df.columns)
    df2 = pd.concat(df2, ignore_index=True) if df2 else pd.DataFrame(columns=point_kernel_df.columns)
    return df1, df2

def filter_kernel_max_values(point_kernel_df):
    filtered_df = point_kernel_df.loc[
        point_kernel_df.groupby('image')['conv4_Vinner_value'].idxmax()
    ]
    return filtered_df

def get_min_point_from_nested_values(point_df, point_kernel_max_df):
    # 分成两部分：有 nested 值的部分和没有 nested 值的部分
    nested_df = point_kernel_max_df[point_kernel_max_df['nested'].notna()].copy()
    no_nested_df = point_kernel_max_df[point_kernel_max_df['nested'].isna()].copy()

    result_rows = []

    # 遍历有 nested 值的部分
    for _, row in nested_df.iterrows():
        image = row['image']
        nested_list = row['nested'].split(",")  # 将当前行的 nested 列分成列表
        # 在 point_df 中找到 image 列值相等的行
        matching_rows = point_df[point_df['image'] == image].copy()
        # 将 matching_rows 的 nested 列转换为列表
        matching_rows['nested_list'] = matching_rows['nested'].fillna("").apply(lambda x: x.split(",") if x else [])

        candidates = []

        # 遍历 nested 列的值
        for nested_value in nested_list:
            # 找到 nested_list 列中包含当前值的行
            candidate_rows = matching_rows[matching_rows['nested_list'].apply(lambda x: nested_value in x)]

            if not candidate_rows.empty:
                # 找到 conv4_V_inner_value 最大的行
                max_conv4_row = candidate_rows.loc[candidate_rows['conv4_V_inner_value'].idxmax()]

                # 如果 conv4_V_inner_value 一样大，比较 conv2_Vchannel_value
                max_conv4_value = max_conv4_row['conv4_V_inner_value']
                tie_rows = candidate_rows[candidate_rows['conv4_V_inner_value'] == max_conv4_value]
                if len(tie_rows) > 1:
                    max_conv2_row = tie_rows.loc[tie_rows['conv2_Vchannel_value'].idxmax()]
                    if len(tie_rows[tie_rows['conv2_Vchannel_value'] == max_conv2_row['conv2_Vchannel_value']]) > 1:
                        print(f"In nested_value: Tie detected for image {image} and nested value {nested_value}:")
                        print(tie_rows)
                    max_row = max_conv2_row
                else:
                    max_row = max_conv4_row

                # 添加 nested_value 到候选行
                max_row = max_row.to_dict()
                max_row['nested_value_used'] = nested_value
                candidates.append(max_row)

        # 对同一个 image 的所有 nested_value 的结果进行比较
        if candidates:
            for candidate in candidates:
                candidate.pop('nested_list', None)  # 如果存在 nested_list 列，则移除
            # 将 candidates 转换为 DataFrame，并根据 image 和 point 列去重
            candidates_df = pd.DataFrame(candidates).drop_duplicates(subset=['image', 'point'])
            max_conv4_row = candidates_df.loc[candidates_df['conv4_V_inner_value'].idxmax()]

            # 如果 conv4_V_inner_value 一样大，比较 conv2_Vchannel_value
            max_conv4_value = max_conv4_row['conv4_V_inner_value']
            tie_rows = candidates_df[candidates_df['conv4_V_inner_value'] == max_conv4_value]
            if len(tie_rows) > 1:
                max_conv2_row = tie_rows.loc[tie_rows['conv2_Vchannel_value'].idxmax()]
                if len(tie_rows[tie_rows['conv2_Vchannel_value'] == max_conv2_row['conv2_Vchannel_value']]) > 1:
                    print(f"In nested_list: Tie detected for image {image}:")
                    print(tie_rows)
                max_row = max_conv2_row
            else:
                max_row = max_conv4_row

            # 将最终找到的行添加到结果中
            result_rows.append({
                'image': max_row['image'],
                'point': max_row['point'],
                'x_value': max_row['x_value'],
                'y_value': max_row['y_value'],
                'kernel': row['kernel'],
                'conv4_V_inner_value': max_row['conv4_V_inner_value'],
#                'nested_final': max_row['nested'],
                'nested_final': max_row['nested_value_used']  # 保存使用的 nested_value
            })
    # 将有 nested 值的部分结果转换为 DataFrame
    result_df = pd.DataFrame(result_rows)
    # 对于没有 nested 值的部分，直接添加到结果中
    no_nested_rows = no_nested_df[['image', 'point', 'x_value', 'y_value', 'kernel', 'conv4_Vinner_value']].copy()
    no_nested_rows.rename(columns={'conv4_Vinner_value': 'conv4_V_inner_value'}, inplace=True)
    no_nested_rows['nested_final'] = None
    # 合并有 nested 值和无 nested 值的部分
    final_df = pd.concat([result_df, no_nested_rows], ignore_index=True)
    # 按照 image 列从小到大排序
    final_df = final_df.sort_values(by='image').reset_index(drop=True)

    # 根据 image 和 point 列的值，在 point_df 中找到相同的行
    final_df = final_df.merge(
        point_df[['image', 'point', 'conv2_Vchannel_value']],
        how='left',
        on=['image', 'point']
    )
    final_df.rename(columns={'conv2_Vchannel_value': 'conv2_V_channel_value'}, inplace=True)
    return final_df

def calculate_slope_points_multix(point_df, file_path=None, counts_by_x=None):
    if counts_by_x is None:
        if file_path is None:
            raise ValueError("file_path or counts_by_x must be provided")
        counts_by_x = build_counts_by_x(file_path)

    point_df['right_count'] = 0
    point_df['left_count'] = 0
    point_df['right_count_x-1'] = 0
    point_df['left_count_x-1'] = 0
    point_df['right_count_x+1'] = 0
    point_df['left_count_x+1'] = 0

    for index, row in point_df.iterrows():
        x, y = resolve_point_xy(row.get('x_value'), row.get('y_value'), row['point'])
        targets = [
            (x, y),          # 当前点
            (x - 1, y),      # 左移 1 的点
            (x + 1, y)       # 右移 1 的点
        ]

        # 计算每个目标点的斜率为 2 和 -2 的点的数目
        for i, (target_x, target_y) in enumerate(targets):
            x_values = [(x_val, x_val - target_x) for x_val in range(target_x - 100, target_x + 101)]
            right_count, left_count = sum_slope_counts(counts_by_x, x_values, target_y)
# 下面注释掉的代码可以插入的位置
            # 根据目标点更新对应的列
            if i == 0:  # 当前点
                point_df.at[index, 'right_count'] = right_count
                point_df.at[index, 'left_count'] = left_count
            elif i == 1:  # 左移 1 的点
                point_df.at[index, 'right_count_x-1'] = right_count
                point_df.at[index, 'left_count_x-1'] = left_count
            elif i == 2:  # 右移 1 的点
                point_df.at[index, 'right_count_x+1'] = right_count
                point_df.at[index, 'left_count_x+1'] = left_count
    return point_df
"""
            # 计算分子和分母
            numerator = filtered_file_df['y'] - target_y
            denominator = filtered_file_df['x'] - target_x
            # 处理 0/0 的情况，将其替换为 1/1
            numerator = numerator.where(~((numerator == 0) & (denominator == 0)), 1)
            denominator = denominator.where(~((numerator == 0) & (denominator == 0)), 1)
            # 处理 num/0 的情况，将其排除
            valid_mask = denominator != 0
            # 计算斜率在 1.9 ~ 2.1 范围内的点
            right_points = filtered_file_df[valid_mask & (
                (numerator / denominator).between(1.9, 2.1)
            )]
            right_count = right_points['count'].sum()
            # 计算斜率在 -1.9 ~ -2.1 范围内的点
            left_points = filtered_file_df[valid_mask & (
                (numerator / denominator).between(-2.1, -1.9)
            )]
            left_count = left_points['count'].sum()
"""

def calculate_slope_points(point_df, file_path=None, counts_by_x=None):
    if counts_by_x is None:
        if file_path is None:
            raise ValueError("file_path or counts_by_x must be provided")
        counts_by_x = build_counts_by_x(file_path)

    for offset in range(5):  # y+0 到 y+4
        point_df[f'left_count_y+{offset}'] = 0
        point_df[f'right_count_y+{offset}'] = 0

    # 遍历 point_df 的每一行
    for index, row in point_df.iterrows():
        x, y = resolve_point_xy(row.get('x_value'), row.get('y_value'), row['point'])
        x_values = [(x_val, x_val - x) for x_val in range(x - 50, x + 51)]

        # 针对 y+0 到 y+4 计算 left_count 和 right_count
        for offset in range(5):
            target_y = y + offset

            right_count, left_count = sum_slope_counts(counts_by_x, x_values, target_y)

            point_df.at[index, f'right_count_y+{offset}'] = right_count
            point_df.at[index, f'left_count_y+{offset}'] = left_count

    # 过滤条件：保留至少有一组 (left_count_y+offset, right_count_y+offset) 都全为 0 的行
    filter_condition = point_df.apply(
        lambda row: any(
            row[f'left_count_y+{offset}'] != 0 and row[f'right_count_y+{offset}'] != 0
#            row[f'left_count_y+{offset}'] > 1 and row[f'right_count_y+{offset}'] > 1
            for offset in range(5)
        ),
        axis=1
    )
    filtered_point_df = point_df[filter_condition].reset_index(drop=True)
    return filtered_point_df

def filter_by_reads_position_le(point_df, file_path=None, counts_by_x=None):
    if counts_by_x is None:
        if file_path is None:
            raise ValueError("file_path or counts_by_x must be provided")
        counts_by_x = build_counts_by_x(file_path)
    df_le_20 = point_df[point_df['y_value'] <= 20].copy()
    df_gt_20 = point_df[point_df['y_value'] > 20].copy()

    keep_mask = []
    for row in df_le_20.itertuples(index=False):
        x, y = resolve_point_xy(row.x_value, row.y_value, row.point)
        keep_mask.append(has_nearby_reads(x, y, counts_by_x, window=3))

    filtered_df_le_20 = df_le_20[keep_mask].copy()

    merged_df = pd.concat([filtered_df_le_20, df_gt_20], ignore_index=True)

    merged_df = merged_df.sort_values(by=['image', 'x_value']).reset_index(drop=True)
    return merged_df

def filter_by_reads_position(point_df, file_path=None, counts_by_x=None):
    if counts_by_x is None:
        if file_path is None:
            raise ValueError("file_path or counts_by_x must be provided")
        counts_by_x = build_counts_by_x(file_path)

    keep_mask = []
    for row in point_df.itertuples(index=False):
        x, y = resolve_point_xy(row.x_value, row.y_value, row.point)
        keep_mask.append(has_nearby_reads(x, y, counts_by_x, window=3))

    filtered_df = point_df[keep_mask].copy()
    filtered_df = filtered_df.sort_values(by=['image', 'x_value']).reset_index(drop=True)
    return filtered_df

def filter_top_conv4_values_onecluster(vcluster_df):
    filtered_df = vcluster_df.loc[vcluster_df.groupby(['image', 'cluster_v_region'])['conv4_V_inner_value'].idxmax()]
    columns_to_keep = ['image', 'point', 'x_value', 'y_value', 'conv4_V_inner_value', 'cluster_v_region']
    filtered_df = filtered_df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

def filter_top_conv4_values(vcluster_df):
    # 创建临时DataFrame并添加原始索引列
    temp_df = vcluster_df.reset_index().copy()  # 添加原始索引到名为'index'的列

    # 将cluster_v_region列拆分为多行（每个簇一个条目）
    temp_df['cluster'] = temp_df['cluster_v_region'].str.split(',')
    temp_df = temp_df.explode('cluster')
    temp_df['cluster'] = temp_df['cluster'].astype(str)  # 确保簇编号为字符串类型

    # 按image和簇编号分组，获取每个簇的最大conv4_V_inner_value
    max_values = temp_df.groupby(['image', 'cluster'])['conv4_V_inner_value'].max().reset_index()

    # 创建一个唯一标识，用于后续匹配
    max_values['key'] = max_values['image'].astype(str) + '_' + max_values['cluster'] + '_' + max_values['conv4_V_inner_value'].astype(str)

    # 为原始DataFrame创建相同的标识
    temp_df['key'] = temp_df['image'].astype(str) + '_' + temp_df['cluster'] + '_' + temp_df['conv4_V_inner_value'].astype(str)

    # 筛选出匹配最大conv4_V_inner_value的行
    filtered_temp = temp_df[temp_df['key'].isin(max_values['key'])]

    # 对于每个image和簇，可能有多行具有相同的最大值，只保留其中一行
    filtered_temp = filtered_temp.drop_duplicates(subset=['image', 'cluster'])

    # 提取原始索引并映射回原始DataFrame
    filtered_indices = filtered_temp['index'].tolist()
    filtered_df = vcluster_df.loc[filtered_indices]

    # 保留需要的列
    columns_to_keep = ['image', 'point', 'x_value', 'y_value', 'conv4_V_inner_value', 'cluster_v_region']
    filtered_df = filtered_df[columns_to_keep]

    return filtered_df.reset_index(drop=True)
    
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

    file_path = config["file_path"]
    final_point_fix_file = config["final_point_fix_file"]
    point_file = config["point_file"]

    point_vcluster_file = point_file.replace('.csv', '.vcluster.csv')
    point_topconv4_file = point_file.replace('.csv', '.topconv4.csv')

    ensure_directories_exist(point_file)

    point_df = pd.read_csv(final_point_fix_file, sep="\t")
#    merged_df = merge_conv4_V_inner_images(post_conv3_step2_base_dir)
#    point_df = add_conv4_V_inner_value(point_df, merged_df)
    counts_by_x = build_counts_by_x(file_path)
    point_df = calculate_slope_points(point_df, counts_by_x=counts_by_x)
#    point_df.to_csv('1.csv', sep="\t", index=False)
    point_df = point_df.rename(columns={'conv3_channel_value': 'conv4_V_inner_value'})
    point_df = filter_by_reads_position(point_df, counts_by_x=counts_by_x)
    point_df.to_csv(point_file, sep="\t", index=False)

# vcluster
    vcluster_df = assign_v_clusters(point_file)
    vcluster_df.to_csv(point_vcluster_file, sep='\t', index=False)

# topconv4
    topconv4_df = filter_top_conv4_values(vcluster_df)
    topconv4_df.to_csv(point_topconv4_file, sep='\t', index=False)
