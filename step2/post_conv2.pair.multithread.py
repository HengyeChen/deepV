from multiprocessing import Pool
import os
import argparse

def process_config(config_file):
    os.system(f"python scripts/post_conv2.pair.py {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多进程处理单个配置文件")
    parser.add_argument("config_file", help="需要处理的配置文件路径")
    args = parser.parse_args()

    with Pool(processes=20) as pool:  # 单个文件处理，进程数设为1
        pool.map(process_config, [args.config_file])  # 包装为列表以适配map

