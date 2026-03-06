import numpy as np
import pandas as pd
import os
import argparse
from typing import List, Tuple, Dict, Any
from scipy.signal import savgol_filter, find_peaks  # 导入find_peaks

def load_csv_data(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取CSV并返回DataFrame与x/y数组"""
    try:
        df = pd.read_csv(csv_path, delimiter='\t')
        if 'start_y' in df.columns:
            df['start_y_plus_10'] = df['start_y'] + 10
        if 'start_y' in df.columns and 'conv2_channel_value' in df.columns:
            x = df['start_y'].values + 10  # x坐标：start_y+10
            y = df['conv2_channel_value'].values  # y坐标：通道值
        else:
            x = np.array([])
            y = np.array([])
        return df, x, y
    except Exception as e:
        print(f"读取文件 {csv_path} 出错: {e}")
        return pd.DataFrame(), np.array([]), np.array([])

def read_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    _, x, y = load_csv_data(csv_path)
    return x, y

def read_csv_df(csv_file_path: str) -> pd.DataFrame:
    "添加start_y_plus_10列"
    df, _, _ = load_csv_data(csv_file_path)
    return df

# ---- 极大值检测（基于find_peaks优化）----
def _detect_extrema_from_arrays(x: np.ndarray, y: np.ndarray,
                                min_relative_height: float = 0.5) -> Dict[str, Any]:
    if len(y) < 3:
        return {'x': x, 'y': y, 'max_indices': []}

    # 1. 使用find_peaks获取潜在极大值（基础筛选）
    # 补充首尾点处理（find_peaks默认不检测首尾，需单独判断）
    potential_max_indices, _ = find_peaks(y)
    potential_max_indices = list(potential_max_indices)
    
    # 处理起始点：若一阶差分≤0，视为潜在极大值
    dy = np.diff(y)
    if len(dy) > 0 and dy[0] <= 0:
        potential_max_indices.insert(0, 0)
    # 处理终止点：若最后一阶差分>0，视为潜在极大值
    if len(dy) > 0 and dy[-1] > 0:
        potential_max_indices.append(len(y)-1)
    # 去重并排序
    potential_max_indices = sorted(list(set(potential_max_indices)))

    # 2. 筛选显著极大值（保留原相对高度逻辑）
    significant_max_indices = []
    global_min = np.min(y)
    # 重新获取潜在极小值（用于基准计算）
    potential_min_indices, _ = find_peaks(-y)  # 极小值是-y的极大值
    potential_min_index_set = set(potential_min_indices)

    for max_idx in potential_max_indices:
        # 寻找左右最近的极小值（同原逻辑）
        left_min_idx = None
        for i in range(max_idx - 1, -1, -1):
            if i in potential_min_index_set:
                left_min_idx = i
                break
        right_min_idx = None
        for i in range(max_idx + 1, len(y)):
            if i in potential_min_index_set:
                right_min_idx = i
                break

        # 计算基准值和相对高度
        if left_min_idx is not None and right_min_idx is not None:
            base_val = min(y[left_min_idx], y[right_min_idx])
        elif left_min_idx is not None:
            base_val = y[left_min_idx]
        elif right_min_idx is not None:
            base_val = y[right_min_idx]
        else:
            base_val = global_min

        peak_val = y[max_idx]
        relative_height = (peak_val - base_val) / base_val if base_val != 0 else float('inf')
        if relative_height >= min_relative_height:
            significant_max_indices.append(max_idx)

    return {
        'x': x, 'y': y,
        'max_indices': significant_max_indices,
        'potential_max_indices': potential_max_indices,
        'min_indices': potential_min_indices
    }

def detect_extrema(csv_file_path: str, min_relative_height: float = 0.5) -> Dict[str, Any]:
    _, x, y = load_csv_data(csv_file_path)
    return _detect_extrema_from_arrays(x, y, min_relative_height)

# ---- 极小值检测（基于find_peaks优化）----
def _detect_minima_from_arrays(df: pd.DataFrame, x: np.ndarray, y: np.ndarray,
                               sensitivity: float = 2.0, window_size: int = 7) -> Dict[str, Any]:
    if len(y) < window_size:
        print(f"数据长度不足 ({len(y)} < {window_size})，无法检测极小值")
        return {'x': x, 'y': y, 'min_indices': []}

    # 1. 数据平滑（保留原逻辑）
    y_smooth = savgol_filter(y, window_length=window_size, polyorder=3)

    # 2. 计算自适应阈值（保留原逻辑）
    local_std = (
        pd.Series(y)
        .rolling(window=window_size, center=True, min_periods=1)
        .std(ddof=0)
        .to_numpy()
    )
    threshold = sensitivity * local_std

    # 3. 使用find_peaks检测潜在极小值（通过-y转化为极大值问题）
    # 结合一阶导数变化幅度阈值
    dy = np.gradient(y_smooth)
    peak_threshold = [0, np.max(dy)]  # 仅筛选上升沿的峰值
    potential_min_indices, _ = find_peaks(
        -y_smooth,  # 极小值 → -y的极大值
        height=(-np.max(y_smooth), -np.min(y_smooth)),  # 高度范围
        threshold=(threshold.min(), threshold.max())  # 阈值范围
    )
    potential_min_index_set = set(potential_min_indices)

    # 4. 多尺度验证（保留原逻辑，用find_peaks简化单尺度检测）
    scales = [5, window_size, min(11, len(y)-1)]
    scale_votes = np.zeros(len(y), dtype=int)
    for s in scales:
        if s >= len(y) or s % 2 == 0:  # 确保窗口为奇数
            continue
        y_smooth_scale = savgol_filter(y, window_length=s, polyorder=3)
        # 单尺度检测极小值
        scale_min_indices, _ = find_peaks(-y_smooth_scale)
        for idx in scale_min_indices:
            if idx not in potential_min_index_set:
                potential_min_index_set.add(idx)
            scale_votes[idx] += 1

    # 5. 后处理（保留原逻辑）
    # 按支持尺度数量排序并去重
    potential_min_indices = sorted(
        potential_min_index_set,
        key=lambda idx: -scale_votes[idx]
    )
    potential_min_indices = list(dict.fromkeys(potential_min_indices))
    # 去除相邻过近的点
    significant_min_indices = []
    if potential_min_indices:
        potential_min_indices.sort()
        significant_min_indices.append(potential_min_indices[0])
        for idx in potential_min_indices[1:]:
            if idx - significant_min_indices[-1] > window_size // 2:
                significant_min_indices.append(idx)

    # 6. 转折点验证（保留原逻辑）
    filtered_min_indices = []
    minima_info = []
    start_y = df['start_y'].to_numpy()
    values = df['conv2_channel_value'].to_numpy()
    is_sorted = np.all(start_y[:-1] <= start_y[1:])

    def last_index_less_than(value: float) -> int:
        if is_sorted:
            idx = np.searchsorted(start_y, value, side='left') - 1
            return idx if idx >= 0 else -1
        indices = np.flatnonzero(start_y < value)
        return int(indices[-1]) if indices.size else -1

    def first_index_greater_than(value: float) -> int:
        if is_sorted:
            idx = np.searchsorted(start_y, value, side='right')
            return idx if idx < len(start_y) else -1
        indices = np.flatnonzero(start_y > value)
        return int(indices[0]) if indices.size else -1

    def indices_equal_to(value: float) -> np.ndarray:
        if is_sorted:
            left = np.searchsorted(start_y, value, side='left')
            right = np.searchsorted(start_y, value, side='right')
            return np.arange(left, right) if left < right else np.array([], dtype=int)
        return np.flatnonzero(start_y == value)

    for min_idx in significant_min_indices:
        x_lowest = start_y[min_idx]

        # 向左寻找转折点
        x_left_pre = None
        max_idx_left = last_index_less_than(x_lowest)
        if max_idx_left >= 0:
            for i in range(max_idx_left, -1, -1):
                if i == 0:
                    if values[i] >= values[min_idx]:
                        x_left_pre = start_y[i]
                    break
                elif values[i] > values[i - 1]:
                    break
                else:
                    x_left_pre = start_y[i - 1]

        # 向右寻找转折点
        x_right_pre = None
        min_idx_right = first_index_greater_than(x_lowest)
        if min_idx_right >= 0:
            for i in range(min_idx_right, len(df)):
                if i == len(df) - 1:
                    if values[i] >= values[min_idx]:
                        x_right_pre = start_y[i]
                    break
                elif values[i] < values[i - 1]:
                    break
                else:
                    x_right_pre = start_y[i]

        # 验证并保存
        x_left = None
        if x_left_pre is not None:
            left_prev_idx = last_index_less_than(x_left_pre)
            left_equal_indices = indices_equal_to(x_left_pre)
            if left_equal_indices.size:
                left_value_ok = True
                if left_prev_idx >= 0:
                    left_value_ok = np.all(values[left_equal_indices] >= values[left_prev_idx])
                if left_value_ok:
                    x_left = x_left_pre + 10

        x_right = None
        if x_right_pre is not None:
            right_next_idx = first_index_greater_than(x_right_pre)
            right_equal_indices = indices_equal_to(x_right_pre)
            if right_equal_indices.size:
                right_value_ok = True
                if right_next_idx >= 0:
                    right_value_ok = np.all(values[right_equal_indices] >= values[right_next_idx])
                if right_value_ok:
                    x_right = x_right_pre + 10

        if x_left is not None or x_right is not None:
            filtered_min_indices.append(min_idx)
            minima_info.append({
                'index': min_idx,
                'x_lowest': x_lowest + 10,
                'x_left': x_left,
                'x_right': x_right
            })

    return {
        'x': x, 'y': y,
        'min_indices': filtered_min_indices,
        'minima_info': minima_info,
        'method': 'adaptive'
    }

def detect_minima(csv_file_path: str, sensitivity: float = 2.0, window_size: int = 7) -> Dict[str, Any]:
    df, x, y = load_csv_data(csv_file_path)
    return _detect_minima_from_arrays(df, x, y, sensitivity, window_size)

def process_files(max_csv_path: str, min_csv_path: str, detect_type: str,
                  output_file: str, min_relative_height: float = 0.3,
                  sensitivity: float = 0.5, window_size: int = 5) -> None:
    with open(output_file, 'w') as f:
        f.write("type\tx\ty\tx_left\tx_right\n")

        if detect_type in ['max', 'both'] and max_csv_path:
            print(f"检测极大值（文件：{max_csv_path}）")
            _, max_x, max_y = load_csv_data(max_csv_path)
            max_result = _detect_extrema_from_arrays(max_x, max_y, min_relative_height)
            print(f"检测到 {len(max_result['max_indices'])} 个显著极大值点")
            for i in max_result['max_indices']:
                f.write(f"max\t{max_result['x'][i]}\t{max_result['y'][i]}\t\t\n")

        if detect_type in ['min', 'both'] and min_csv_path:
            print(f"检测极小值（文件：{min_csv_path}）")
            min_df, min_x, min_y = load_csv_data(min_csv_path)
            min_result = _detect_minima_from_arrays(min_df, min_x, min_y, sensitivity, window_size)
            print(f"检测到 {len(min_result['min_indices'])} 个显著极小值点")
            for info in min_result['minima_info']:
                f.write(f"min\t{info['x_lowest']}\t{min_result['y'][info['index']]}\t{info['x_left'] or ''}\t{info['x_right'] or ''}\n")

def main():
    parser = argparse.ArgumentParser(description='CSV极值检测系统（基于find_peaks优化）')
    parser.add_argument('--max_csv_path', type=str, help='用于极大值检测的CSV文件路径')
    parser.add_argument('--min_csv_path', type=str, help='用于极小值检测的CSV文件路径')
    parser.add_argument('--detect_type', choices=['max', 'min', 'both'], default='both', help='检测类型')
    parser.add_argument('--output_file', type=str, default='extrema.csv', help='结果输出文件路径')
    parser.add_argument('--min_relative_height', type=float, default=0.3, help='极大值最小相对高度')
    parser.add_argument('--sensitivity', type=float, default=0.5, help='极小值检测灵敏度')
    parser.add_argument('--window_size', type=int, default=5, help='极小值检测窗口大小（奇数）')
    args = parser.parse_args()

    process_files(
        args.max_csv_path,
        args.min_csv_path,
        args.detect_type,
        args.output_file,
        min_relative_height=args.min_relative_height,
        sensitivity=args.sensitivity,
        window_size=args.window_size
    )

if __name__ == "__main__":
    main()
