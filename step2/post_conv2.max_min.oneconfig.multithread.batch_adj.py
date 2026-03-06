import os
import re
import json
import sys
import pandas as pd
import importlib.util
from multiprocessing import Pool

_WORKER_MODULE = None
_WORKER_PATH = os.path.join(os.path.dirname(__file__), "post_conv2.max_min.batch.py")
MAX_CONCURRENT_TASKS = 10
MIN_RELATIVE_HEIGHT = 0.3
SENSITIVITY = 0.5
WINDOW_SIZE = 5

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

def parse_target_images(config):
    target_images = config.get("target_images")
    if target_images is None:
        target_images = config.get("target_image")
    if target_images is None:
        start = config.get("target_image_start")
        end = config.get("target_image_end")
        if start is None and end is None:
            return set()
        if start is None:
            start = end
        if end is None:
            end = start
        start = int(start)
        end = int(end)
        if end < start:
            print("target_image_end must be >= target_image_start")
            sys.exit(1)
        return set(range(start, end + 1))
    if isinstance(target_images, int):
        return {target_images}
    if isinstance(target_images, list):
        return {int(value) for value in target_images}
    if isinstance(target_images, str):
        parts = [value for value in re.split(r"[\s,]+", target_images.strip()) if value]
        return {int(value) for value in parts}
    return set()

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

def run_task(task):
    module = _load_worker_module()
    (
        image_num,
        y_value,
        inner_label,
        max_csv_path,
        min_csv_path,
        detect_type,
        min_relative_height,
        sensitivity,
        window_size,
    ) = task
    rows = module.collect_rows(
        max_csv_path,
        min_csv_path,
        detect_type,
        min_relative_height=min_relative_height,
        sensitivity=sensitivity,
        window_size=window_size,
        verbose=False,
    )
    return (image_num, y_value, inner_label, rows)

def manage_tasks(tasks, max_concurrent_tasks=50):
    """管理并限制后台运行的任务数"""
    if not tasks:
        return []
    with Pool(processes=max_concurrent_tasks, initializer=_init_worker) as pool:
        return pool.map(run_task, tasks)

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
    target_images = parse_target_images(config)

    inner_index, image_nums = collect_inner_files(region_base_dir_Vinner, y_values, labels)
    if not image_nums:
        print("not find image file")
        sys.exit(1)
    image_nums_sorted = sorted(image_nums)
    print(f"找到 {len(image_nums_sorted)} 个image，范围: {image_nums_sorted[0]} 到 {image_nums_sorted[-1]}")
    #if target_images:
     #   missing_images = sorted(target_images.difference(image_nums))
      #  if missing_images:
       #     print(f"目标image未找到: {missing_images}")
        #    sys.exit(1)
    # 找到这段代码（大约在main函数的中间位置）
    if target_images:
         missing_images = sorted(target_images.difference(image_nums))
         if missing_images:
             print(f"目标image未找到: {missing_images}")
        # sys.exit(1)  # 注释掉这行，改为继续处理存在的image
        
        # 改为只处理存在的image
             target_images = target_images.intersection(image_nums)
             if not target_images:
                 print("没有可处理的image，退出")
                 sys.exit(1)
             else:
                 print(f"将处理存在的 {len(target_images)} 个image")
    # 2. 生成inner和channel文件的配对
    channel_index = collect_channel_files(region_base_dir_Vchannel, y_values)
    pairings = generate_file_pairings(inner_index, channel_index)

    # 3. 生成并运行处理命令（按 image 范围输出大文件）
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

        if target_images and image_num not in target_images:
            continue

        task = (
            image_num,
            y_value,
            inner_label,
            max_csv_path,
            min_csv_path,
            detect_type,
            MIN_RELATIVE_HEIGHT,
            SENSITIVITY,
            WINDOW_SIZE,
        )
        tasks_by_image.setdefault(image_num, []).append((
            y_value,
            label_order.get(inner_label, len(label_order)),
            task
        ))

    print("Start processing files...")
    if not tasks_by_image:
        print("no tasks to run")
        return

    image_nums_sorted = sorted(tasks_by_image)
    batch_start = image_nums_sorted[0]
    batch_end = image_nums_sorted[-1]
    output_file = f"{root_dir}/post_conv2/max_min/{output_path_prefix}/batch_{batch_start}_{batch_end}.max_min.csv"
    ensure_directories_exist(output_file)

    batch_tasks = []
    for image_num in image_nums_sorted:
        entries = sorted(tasks_by_image[image_num], key=lambda item: (item[0], item[1]))
        batch_tasks.extend([entry[2] for entry in entries])

    results = manage_tasks(batch_tasks, max_concurrent_tasks=MAX_CONCURRENT_TASKS)
    with open(output_file, "w") as f:
        f.write("image\ty_value\tkernel\ttype\tx\ty\tx_left\tx_right\n")
        for image_num, y_value, inner_label, rows in results:
            if not rows:
                continue
            for row in rows:
                row_type, x_val, y_val, x_left, x_right = row
                f.write(
                    f"{image_num}\t{y_value}\t{inner_label}\t{row_type}\t{x_val}\t{y_val}\t{x_left}\t{x_right}\n"
                )
    print(f"Wrote {output_file}")

if __name__ == "__main__":
    main()
