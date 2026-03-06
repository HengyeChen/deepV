import os
import json
import argparse

'''
def generate_config_files_conv1(base_dir, start, end, step, overlap, max_end, threshold, file_path):
    current_start = start
    current_end = end
    threshold = threshold
    config_dir = os.path.join(base_dir, "config/conv1")
    os.makedirs(config_dir, exist_ok=True)

    conv1_files = []  # 用于存储生成的 filter_csv_path 文件名

    while current_end < max_end:
        config_filename = f"conv1.{current_start // 1000}-{current_end // 1000}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        filter_csv_path = f"{base_dir}/conv1/conv1.sigmoid{threshold}.{current_start // 1000}-{current_end // 1000}kb.csv"
        conv1_files.append(os.path.basename(filter_csv_path))  # 提取文件名并加入列表

        config_content = {
            "file_path": file_path,
            "start": current_start,
            "end": current_end,
            "step": step,
            "overlap": overlap,
            "conv1_csv_path": f"{base_dir}/conv1/conv1.{current_start // 1000}-{current_end // 1000}kb.csv",
            "output_csv_path": f"{base_dir}/conv1/conv1.sigmoid.{current_start // 1000}-{current_end // 1000}kb.csv",
            "filter_csv_path": filter_csv_path
        }

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)

        print(f"Generated: {config_filepath}")

        current_start = current_end - overlap
        current_end = current_start + 1000000
    return conv1_files
'''

def generate_config_files_conv1(base_dir, start, end, step, overlap, max_end, threshold, file_path):
    current_start = start
    current_end = end
    threshold = threshold
    config_dir = os.path.join(base_dir, "config/conv1")
    os.makedirs(config_dir, exist_ok=True)

    conv1_files = []
    is_first_file = True  # 标记是否是第一个文件

    while current_end < max_end:
        if is_first_file:
            # 第一个文件：使用原始逻辑，不强制以9结尾
            kb_start = current_start // 1000
            kb_end = current_end // 1000
        else:
            # 非第一个文件：强制构造以9结尾的数值
            kb_start = (current_start // 1000) // 10 * 10 + 9
            kb_end = kb_start + 1000
            kb_start = kb_end - 1000
            kb_start = int(str(kb_start)[:-1] + "9")
            kb_end = int(str(kb_end)[:-1] + "9")
        
        config_filename = f"conv1.{kb_start}-{kb_end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        filter_csv_path = f"{base_dir}/conv1/conv1.sigmoid{threshold}.{kb_start}-{kb_end}kb.csv"
        conv1_files.append(os.path.basename(filter_csv_path))

        config_content = {
            "file_path": file_path,
            "start": current_start,
            "end": current_end,
            "step": step,
            "overlap": overlap,
            "conv1_csv_path": f"{base_dir}/conv1/conv1.{kb_start}-{kb_end}kb.csv",
            "output_csv_path": f"{base_dir}/conv1/conv1.sigmoid.{kb_start}-{kb_end}kb.csv",
            "filter_csv_path": filter_csv_path
        }

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)

        print(f"Generated: {config_filepath}")

        is_first_file = False
        current_start = current_end - overlap
        current_end = current_start + 1000000
    return conv1_files

def generate_config_files_conv2(conv1_files, base_dir, file_path):
    config_dir = os.path.join(base_dir, "config/conv2")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "file_path": file_path,
            "conv1_file_path": f"{base_dir}/conv1/{conv1_file}",
            "conv1_merge_path": f"{base_dir}/conv2/conv1_merge/conv1.{start}-{end}kb.merge.csv",
            "region_base_dir_Vinner": f"{base_dir}/conv2/V_inner/{base_dir.split('_')[0]}_{start}-{end}kb",
            "region_base_dir_Vchannel": f"{base_dir}/conv2/V_channel/{base_dir.split('_')[0]}_{start}-{end}kb"
        }

        config_filename = f"conv2.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_post_conv2(conv1_files, base_dir):
    config_dir = os.path.join(base_dir, "config/post_conv2")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "region_base_dir_Vinner": f"{base_dir}/conv2/V_inner/{base_dir.split('_')[0]}_{start}-{end}kb",
            "region_base_dir_Vchannel": f"{base_dir}/conv2/V_channel/{base_dir.split('_')[0]}_{start}-{end}kb",
            "output_path_prefix": f"{base_dir.split('_')[0]}_{start}-{end}kb",
        }

        config_filename = f"post_conv2.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_conv3(conv1_files, base_dir, file_path):
    config_dir = os.path.join(base_dir, "config/conv3")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "file_path": file_path,
            "point_filter_file": f"{base_dir}/post_conv2/pair/{base_dir.split('_')[0]}_{start}-{end}kb.pair.csv",
            "conv3_base_dir": f"{base_dir}/conv3/{base_dir.split('_')[0]}_{start}-{end}kb"
#            "conv1_file_path": f"{base_dir}/conv1/{conv1_file}",
#            "conv1_merge_path": f"{base_dir}/conv2/conv1_merge/conv1.{start}-{end}kb.merge.csv",
#            "region_base_dir_Vinner": f"{base_dir}/conv2/V_inner/{base_dir.split('_')[0]}_{start}-{end}kb",
#            "region_base_dir_Vchannel": f"{base_dir}/conv2/V_channel/{base_dir.split('_')[0]}_{start}-{end}kb"
        }

        config_filename = f"conv3.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_post_conv3_step1(conv1_files, base_dir):
    config_dir = os.path.join(base_dir, "config/post_conv3")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "point_filter_file": f"{base_dir}/post_conv2/pair/{base_dir.split('_')[0]}_{start}-{end}kb.pair.csv",
            "conv3_dir": f"{base_dir}/conv3/{base_dir.split('_')[0]}_{start}-{end}kb",
            "slope_output_file": f"{base_dir}/post_conv3/step1.slope/slope.{base_dir.split('_')[0]}_{start}-{end}kb.csv",
            "slope_diff_output_file": f"{base_dir}/post_conv3/step1.slope/slope.diff.{base_dir.split('_')[0]}_{start}-{end}kb.csv",
            "final_point_fix_file": f"{base_dir}/post_conv3/step1.slope/point.fix.{base_dir.split('_')[0]}_{start}-{end}kb.csv",
        }

        config_filename = f"post_conv3.step1.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_post_conv3_step2(conv1_files, base_dir, file_path):
    config_dir = os.path.join(base_dir, "config/post_conv3")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "file_path": file_path,
            "final_point_fix_file": f"{base_dir}/post_conv3/step1.slope/point.fix.{base_dir.split('_')[0]}_{start}-{end}kb.csv",
            "post_conv3_step2_base_dir": f"{base_dir}/post_conv3/step2.conv3_fix/{base_dir.split('_')[0]}_{start}-{end}kb"
        }

        config_filename = f"post_conv3.step2.conv3.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_post_conv3_step3(conv1_files, base_dir, file_path):
    config_dir = os.path.join(base_dir, "config/post_conv3")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "file_path": file_path,
            "final_point_fix_file": f"{base_dir}/post_conv3/step2.slope_fix/{base_dir.split('_')[0]}_{start}-{end}kb/slope_diff.csv",
            "point_file": f"{base_dir}/post_conv3/step3.filter/point.fix.{base_dir.split('_')[0]}_{start}-{end}kb.csv"
        }

        config_filename = f"post_conv3.step3.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_post_conv3_step2_slope(conv1_files, base_dir):
    config_dir = os.path.join(base_dir, "config/post_conv3")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "point_filter_file": f"{base_dir}/post_conv3/step1.slope/point.fix.{base_dir.split('_')[0]}_{start}-{end}kb.csv",
            "post_conv3_step2_conv_dir": f"{base_dir}/post_conv3/step2.conv3_fix/{base_dir.split('_')[0]}_{start}-{end}kb",
            "post_conv3_step2_slope_dir": f"{base_dir}/post_conv3/step2.slope_fix/{base_dir.split('_')[0]}_{start}-{end}kb",
        }

        config_filename = f"post_conv3.step2.slope.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


def generate_config_files_binding_hotspot_conv_step2(conv1_files, base_dir, file_path):
    config_dir = os.path.join(base_dir, "config/binding_hotspot")
    os.makedirs(config_dir, exist_ok=True)

    for conv1_file in conv1_files:
        range_info = conv1_file.split(".")[3]  # 提取类似 "10000-11000kb"
        start, end = range_info.replace("kb", "").split("-")

        config_content = {
            "file_path": file_path,
            "final_point_fix_file": f"{base_dir}/binding_hotspot/{base_dir.split('_')[0]}_{start}-{end}kb/expend_point.csv",
            "binding_hotspot_conv_base_dir": f"{base_dir}/binding_hotspot/{base_dir.split('_')[0]}_{start}-{end}kb/step2.conv"
        }

        config_filename = f"binding_hotspot.step2.{start}-{end}kb.config"
        config_filepath = os.path.join(config_dir, config_filename)

        with open(config_filepath, "w") as f:
            json.dump(config_content, f, indent=4)
        print(f"Generated config: {config_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config files for chromosome regions.")
    parser.add_argument("chr", type=str, help="Chromosome name (e.g., chr1,chrX)")
    parser.add_argument("start_kb", type=int, help="Start position (kb)")
    parser.add_argument("end_kb", type=int, help="End position (kb)")
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    # 构建 base_dir，例如 chr1_0_10000kb 这样的格式
    base_dir = f"{args.chr}_{args.start_kb}_{args.end_kb}kb"
    # 计算 start、end、max_end 等参数，注意单位转换（kb 转成 bp，假设 1kb = 1000bp ）
    start = args.start_kb * 1000
    end = start + 1000000  # 按照需求，end = start + 1000000
    max_end = args.end_kb * 1000  # max_end 为第三个参数（end_kb）*1000
    step = 1000
    overlap = 50
    threshold = 0.90
    # 构建 file_path，替换成实际需要的路径格式，这里示例使用传入的染色体等信息构建路径
#    file_path = f"/home/nwh/software/temp/loMNase_K562/data/{args.chr}/loMNase_K562.{args.chr}.{args.start_kb}-{args.end_kb}kb.bed"
#    file_path = f"/home/nwh/software/temp/loMNase_K562/data/loMNase_K562.chr4.sort.bed"
    file_path = args.file_path

    conv1_files = generate_config_files_conv1(
        base_dir=base_dir,
        start=start,
        end=end,
        step=step,
        overlap=overlap,
        max_end=max_end,
        threshold=threshold,
        file_path=file_path
    )

    generate_config_files_conv2(
        conv1_files=conv1_files,
        base_dir=base_dir,
        file_path=file_path
    )

'''
    generate_config_files_post_conv2(
        conv1_files=conv1_files,
        base_dir=base_dir,
    )

    generate_config_files_conv3(
        conv1_files=conv1_files,
        base_dir=base_dir,
        file_path=file_path
    )

    generate_config_files_post_conv3_step1(
        conv1_files=conv1_files,
        base_dir=base_dir,
    )

    generate_config_files_post_conv3_step2(
        conv1_files=conv1_files,
        base_dir=base_dir,
        file_path=file_path
    )

    generate_config_files_post_conv3_step2_slope(
        conv1_files=conv1_files,
        base_dir=base_dir,
    )

    generate_config_files_post_conv3_step3(
        conv1_files=conv1_files,
        base_dir=base_dir,
        file_path=file_path
    )

    generate_config_files_binding_hotspot_conv_step2(
        conv1_files=conv1_files,
        base_dir=base_dir,
        file_path=file_path
    )
'''
