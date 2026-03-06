#!/usr/bin/env python3
import argparse
import json
import os
import re


def write_config(config_dir, filename, content):
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, filename)
    with open(config_path, "w") as f:
        json.dump(content, f, indent=4)
    print(f"Generated: {config_path}")


def parse_target_images_arg(target_images):
    if target_images is None:
        return None
    parts = [value for value in re.split(r"[\s,]+", target_images.strip()) if value]
    if not parts:
        return None
    if len(parts) == 1:
        return int(parts[0])
    return [int(value) for value in parts]


def generate_configs(
    chr_name,
    start_kb,
    end_kb,
    file_path,
    config_dir,
    result_dir,
    detecte_point_path,
    target_images=None,
    target_image_start=None,
    target_image_end=None,
):
    range_str = f"{start_kb}-{end_kb}kb"
    range_label = f"{chr_name}_{range_str}"

    conv2_config = {
        "file_path": file_path,
        "detecte_point_path": detecte_point_path,
        "detecte_point_expend_path": f"{result_dir}/conv2/detected_points.expend.csv",
        "detecte_point_merge_path": f"{result_dir}/conv2/detected_points.merge.csv",
        "region_base_dir_Vinner": f"{result_dir}/conv2/V_inner",
        "region_base_dir_Vchannel": f"{result_dir}/conv2/V_channel",
    }
    write_config(config_dir, f"conv2.{range_str}.config", conv2_config)

    conv3_new_config = {
        "file_path": file_path,
        "detecte_point_path": detecte_point_path,
        "detecte_point_expend_path": f"{result_dir}/conv3/detected_points.expend.csv",
        "detecte_point_merge_path": f"{result_dir}/conv3/detected_points.merge.csv",
        "region_base_dir_V": f"{result_dir}/conv3",
    }
    write_config(config_dir, f"conv3.new.{range_str}.config", conv3_new_config)

    post_conv2_config = {
        "detecte_point_expend_path": f"{result_dir}/conv2/detected_points.expend.csv",
        "region_base_dir_Vinner": f"{result_dir}/conv2/V_inner",
        "region_base_dir_Vchannel": f"{result_dir}/conv2/V_channel",
        "output_path_prefix": range_label,
    }
    normalized_images = parse_target_images_arg(target_images)
    if normalized_images is not None:
        post_conv2_config["target_images"] = normalized_images
    if target_image_start is not None:
        post_conv2_config["target_image_start"] = target_image_start
    if target_image_end is not None:
        post_conv2_config["target_image_end"] = target_image_end
    write_config(config_dir, f"post_conv2.{range_str}.config", post_conv2_config)

    post_conv3_step1_config = {
        "point_filter_file": f"{result_dir}/post_conv2/pair/{range_label}.pair.csv",
        "conv3_dir": f"{result_dir}/conv3",
        "slope_output_file": f"{result_dir}/post_conv3/step1.slope/slope.{range_label}.csv",
        "slope_diff_output_file": f"{result_dir}/post_conv3/step1.slope/slope.diff.{range_label}.csv",
        "final_point_fix_file": f"{result_dir}/post_conv3/step1.slope/point.fix.{range_label}.csv",
    }
    write_config(config_dir, f"post_conv3.step1.{range_str}.config", post_conv3_step1_config)

    post_conv3_step3_config = {
        "file_path": file_path,
        "final_point_fix_file": f"{result_dir}/post_conv3/step1.slope/point.fix.{range_label}.csv",
        "point_file": f"{result_dir}/post_conv3/step3.filter/point.fix.{range_label}.csv",
    }
    write_config(config_dir, f"post_conv3.step3.{range_str}.config", post_conv3_step3_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate config files for post-train adjust point pipeline.")
    parser.add_argument("chr", type=str, help="Chromosome name (e.g., chr1)")
    parser.add_argument("start_kb", type=int, help="Start position in kb")
    parser.add_argument("end_kb", type=int, help="End position in kb")
    parser.add_argument("file_path", type=str, help="Input BED file path")
    parser.add_argument("--config-dir", default="config", help="Directory to write config files")
    parser.add_argument("--result-dir", default="result", help="Result directory prefix used in configs")
    parser.add_argument(
        "--detecte-point-path",
        default="detected_points.jointcost.processed.tsv",
        help="Detected point input path used in configs",
    )
    parser.add_argument(
        "--target-images",
        default=None,
        help="Target image IDs for post_conv2 (comma/space separated)",
    )
    parser.add_argument(
        "--target-image-start",
        type=int,
        default=None,
        help="Target image start (inclusive) for post_conv2",
    )
    parser.add_argument(
        "--target-image-end",
        type=int,
        default=None,
        help="Target image end (inclusive) for post_conv2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_kb <= args.start_kb:
        raise ValueError("end_kb must be greater than start_kb")
    if (
        args.target_image_start is not None
        and args.target_image_end is not None
        and args.target_image_end < args.target_image_start
    ):
        raise ValueError("target_image_end must be >= target_image_start")
    generate_configs(
        chr_name=args.chr,
        start_kb=args.start_kb,
        end_kb=args.end_kb,
        file_path=args.file_path,
        config_dir=args.config_dir,
        result_dir=args.result_dir,
        detecte_point_path=args.detecte_point_path,
        target_images=args.target_images,
        target_image_start=args.target_image_start,
        target_image_end=args.target_image_end,
    )


if __name__ == "__main__":
    main()
