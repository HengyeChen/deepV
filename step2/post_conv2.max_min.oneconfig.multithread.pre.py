import os
import re
import json
import sys
import pandas as pd
import importlib.util
from multiprocessing import Pool

_WORKER_MODULE = None
_WORKER_PATH = os.path.join(os.path.dirname(__file__), "post_conv2.max_min.py")

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

def collect_inner_files(region_base_dir, y_values, labels):
    inner_index = {}
    image_nums = set()
    for label in labels:
        for y_value in y_values:
            dir_path = os.path.join(
                region_base_dir,
                f"V_{label}/conv2_image_sigmoid_y{y_value}"
            )
            if not os.path.isdir(dir_path):
                continue
            for entry in os.scandir(dir_path):
                if not entry.is_file():
                    continue
                match = re.match(r"image_(\d+)\.sigmoid\.csv", entry.name)
                if match:
                    image_num = int(match.group(1))
                    image_nums.add(image_num)
                    inner_index.setdefault((image_num, y_value), []).append({
                        "label": label,
                        "file_path": entry.path
                    })

    return inner_index, image_nums

def collect_channel_files(region_base_dir, y_values):
    channel_index = {}
    for y_value in y_values:
        dir_path = os.path.join(
            region_base_dir,
            f"conv2_image_sigmoid_y{y_value}"
        )
        if not os.path.isdir(dir_path):
            continue
        for entry in os.scandir(dir_path):
            if not entry.is_file():
                continue
            match = re.match(r"image_(\d+)\.sigmoid\.csv", entry.name)
            if match:
                image_num = int(match.group(1))
                channel_index[(image_num, y_value)] = entry.path

    return channel_index

def generate_file_pairings(inner_index, channel_index):
    pairings = []
    for (image_num, y_value), inner_files in inner_index.items():
        channel_file = channel_index.get((image_num, y_value))
        if not channel_file:
            continue
        for inner in inner_files:
            pairings.append({
                "image": image_num,
                "y_value": y_value,
                "inner_label": inner["label"],
                "inner_file": inner["file_path"],
                "channel_file": channel_file
            })

    return pairings

def _load_worker_module():
    global _WORKER_MODULE
    if _WORKER_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "post_conv2_max_min",
            _WORKER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _WORKER_MODULE = module
    return _WORKER_MODULE

def _init_worker():
    _load_worker_module()

def run_task(task_group):
    module = _load_worker_module()
    for task in task_group:
        module.process_files(*task)

def manage_tasks(tasks, max_concurrent_tasks=50):
    """管理并限制后台运行的任务数"""
    if not tasks:
        return
    with Pool(processes=max_concurrent_tasks, initializer=_init_worker) as pool:
        pool.map(run_task, tasks)

def ensure_directories_exist(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def main():
    if len(sys.argv) != 2:
        print("python script.py <config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)

    region_base_dir_Vinner = config["region_base_dir_Vinner"]
    region_base_dir_Vchannel = config["region_base_dir_Vchannel"]
    output_path_prefix = config["output_path_prefix"]
    detecte_point_expend_path = config["detecte_point_expend_path"]
    detect_type = "both"

    expend_df = pd.read_csv(detecte_point_expend_path, sep='\t')
    y_values, _ = get_y_values(expend_df)

    # 获取目录的最后部分
    root_dir = region_base_dir_Vinner.split(os.sep)[0]

    labels = ["standard", "right", "left", "middle"]
    inner_index, image_nums = collect_inner_files(region_base_dir_Vinner, y_values, labels)
    if not image_nums:
        print("not find image file")
        sys.exit(1)
    image_nums_sorted = sorted(image_nums)
    print(f"找到 {len(image_nums_sorted)} 个image，范围: {image_nums_sorted[0]} 到 {image_nums_sorted[-1]}")

    # 2. 生成inner和channel文件的配对
    channel_index = collect_channel_files(region_base_dir_Vchannel, y_values)
    pairings = generate_file_pairings(inner_index, channel_index)

    # 3. 生成并运行处理命令
    label_order = {label: index for index, label in enumerate(labels)}
    tasks_by_image = {}
    for row in pairings:
        max_csv_path = row["inner_file"]
        min_csv_path = row["channel_file"]
        inner_label = row["inner_label"]

        # 从 max_csv_path 中提取 image 和 y 信息
        match = re.search(r"conv2_image_sigmoid_y(\d+)/image_(\d+)\.sigmoid\.csv", max_csv_path)
        if match:
            y_value = int(match.group(1))
            image_num = int(match.group(2))
        else:
            print(f"无法从路径中提取 y 和 image 信息: {max_csv_path}")
            continue

        output_file = f"{root_dir}/post_conv2/max_min/{output_path_prefix}/image_{image_num}.y_{y_value}.kernel_{inner_label}.max_min.csv"
        ensure_directories_exist(output_file)

        task = (
            max_csv_path,
            min_csv_path,
            detect_type,
            output_file
        )
        tasks_by_image.setdefault(image_num, []).append((
            y_value,
            label_order.get(inner_label, len(label_order)),
            task
        ))

    print("Start processing files...")
    grouped_tasks = []
    for image_num in sorted(tasks_by_image):
        entries = sorted(tasks_by_image[image_num], key=lambda item: (item[0], item[1]))
        grouped_tasks.append([entry[2] for entry in entries])
    manage_tasks(grouped_tasks, max_concurrent_tasks=5)

if __name__ == "__main__":
    main()
