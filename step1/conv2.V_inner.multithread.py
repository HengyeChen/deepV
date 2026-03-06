from multiprocessing import Pool
import os
import argparse

def process_config(config_file):
    os.system(f"python conv2.V_inner.py {config_file}")

def generate_config_files(chr, start_kb, end_kb):
    base_dir = f"{chr}_{start_kb}_{end_kb}kb"
    config_files = []

    end = end_kb // 1000
    for n in range(1, end+1):  # 生成十个文件
        if n == 1:  # 如果是第一个文件
            current_start = start_kb  # 不减 1
        else:
            current_start = start_kb + (n - 1) * 1000 - 1  # 其他文件减 1

        if current_start < 0:  # 检查是否小于 0
            current_start = 0
        current_end = current_start + 1000  # end = start + 1000

        # 生成区间字符串（例如：10000-11000kb）
        range_str = f"{current_start}-{current_end}kb"

        config_path = os.path.join(
            base_dir,
            "config/conv2",
            f"conv2.{range_str}.config"
        )
        config_files.append(config_path)

    return config_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config files for chromosome regions.")
    parser.add_argument("chr", type=str, help="Chromosome name (e.g., chr1,chrX)")
    parser.add_argument("start_kb", type=int, help="Start position (kb)")
    parser.add_argument("end_kb", type=int, help="End position (kb)")
    args = parser.parse_args()

    # 生成配置文件列表
    config_files = generate_config_files(
        chr=args.chr,
        start_kb=args.start_kb,
        end_kb=args.end_kb
    )

    for file in config_files:
        print(file)

    # 多进程处理（建议根据服务器性能调整进程数）
    with Pool(processes=10) as pool:  # 推荐进程数=CPU核心数
        pool.map(process_config, config_files)
