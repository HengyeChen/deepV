#!/usr/bin/env python3
"""
Process detected points table (CSV/TSV):
- Filter out rows where detected_point_absolute.y > Y_MAX (default 60)
- Group by image_index
- Within each group, cluster points using a greedy average-distance rule and
  keep only one row per cluster: the one with the highest detection_confidence.
  Average-distance threshold defaults to 2.

Usage:
  python3 process_detected_points.py INPUT_PATH [--y-max 60] [--avg-dist-threshold 2] [--output-dir RESULT_DIR]

Notes:
- INPUT_PATH may be .tsv or .csv. Delimiter inferred from extension (tsv->"\t", csv->",").
- The script creates an output file under ./result (next to this script) by default.
- Required columns: image_index, detected_point_absolute, detection_confidence
- The detected_point_absolute is parsed by extracting the first two numbers found in the string.
"""
from __future__ import annotations
import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

Number = float
Point = Tuple[Number, Number]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter and de-duplicate detected points by average distance per image group.")
    p.add_argument("input", help="Path to input CSV/TSV file")
    p.add_argument("--y-max", type=float, default=60.0, help="Max allowed Y value (rows with y > this are dropped). Default: 60")
    p.add_argument("--avg-dist-threshold", "-t", type=float, default=2.0, help="Average distance threshold to group close points. Default: 2.0")
    p.add_argument("--output-dir", "-o", default=None, help="Output directory. Default: a 'result' folder next to this script")
    return p.parse_args()


# Regex to extract numbers; use single backslashes in the raw string
_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_point(val: str) -> Point | None:
    """Parse a detected_point_absolute cell into (x, y).
    Strategy: extract first two numeric tokens from the string. Returns None if parsing fails.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    nums = _num_re.findall(s)
    if len(nums) < 2:
        return None
    try:
        x = float(nums[0])
        y = float(nums[1])
        return (x, y)
    except ValueError:
        return None


def euclidean(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def cluster_indices(points: List[Point], threshold: float) -> List[List[int]]:
    """Greedy average-distance clustering.
    For each point in order, try to add it to an existing cluster if the average
    distance to that cluster's members is <= threshold; otherwise start a new cluster.
    Returns clusters as lists of original indices.
    """
    clusters: List[List[int]] = []
    for i, p in enumerate(points):
        placed = False
        for cluster in clusters:
            # average distance from p to all points in cluster
            avg = 0.0
            for j in cluster:
                avg += euclidean(p, points[j])
            avg /= len(cluster)
            if avg <= threshold:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def infer_delimiter(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".tsv":
        return "\t"
    if suf == ".csv":
        return ","
    # Fallback: default to tab to be conservative
    return "\t"


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 2

    # Resolve output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (Path(__file__).resolve().parent / "result")
    out_dir.mkdir(parents=True, exist_ok=True)

    delim = infer_delimiter(in_path)

    # Read
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            print("Error: input file appears to have no header.", file=sys.stderr)
            return 2
        fieldnames = list(reader.fieldnames)
        if "image_index" not in fieldnames or "detected_point_absolute" not in fieldnames:
            print("Error: required columns missing. Need 'image_index' and 'detected_point_absolute'.", file=sys.stderr)
            return 2
        for row in reader:
            rows.append(row)

    # Parse points and filter by Y
    parsed_points: List[Point | None] = []
    keep_mask_y: List[bool] = []
    drop_unparsable = 0
    drop_y = 0
    for row in rows:
        p = parse_point(row.get("detected_point_absolute"))
        parsed_points.append(p)
        if p is None:
            # Unparsable rows are dropped conservatively
            keep_mask_y.append(False)
            drop_unparsable += 1
            continue
        if p[1] > float(args.y_max):
            keep_mask_y.append(False)
            drop_y += 1
        else:
            keep_mask_y.append(True)

    # Apply Y filter
    filtered_rows = [r for r, keep in zip(rows, keep_mask_y) if keep]
    filtered_points = [p for p, keep in zip(parsed_points, keep_mask_y) if keep and p is not None]

    # Group by image_index
    groups: Dict[str, List[int]] = {}
    for idx, r in enumerate(filtered_rows):
        key = str(r.get("image_index"))
        groups.setdefault(key, []).append(idx)

    # For each group, cluster by average distance and keep one per cluster (highest detection_confidence)
    keep_indices_set = set()
    for key, idxs in groups.items():
        pts = [filtered_points[i] for i in idxs]
        clusters = cluster_indices(pts, float(args.avg_dist_threshold))
        for cluster in clusters:
            # Select the row with the highest detection_confidence within this cluster
            best_local_idx = None
            best_conf = float('-inf')
            for local_idx in cluster:
                row = filtered_rows[idxs[local_idx]]
                try:
                    conf = float(row.get("detection_confidence", ""))
                except (TypeError, ValueError):
                    conf = float('-inf')
                if conf > best_conf:
                    best_conf = conf
                    best_local_idx = local_idx
            if best_local_idx is not None:
                keep_indices_set.add(idxs[best_local_idx])

    # Sort kept indices to preserve file order
    kept_sorted = sorted(keep_indices_set)
    output_rows = [filtered_rows[i] for i in kept_sorted]

    # Round detected_point_absolute to integer (四舍五入, 0.5 进位)
    def round_point_str(val: str) -> str:
        s = str(val)
        nums = _num_re.findall(s)
        if len(nums) < 2:
            return s  # fallback, should not happen due to earlier filtering
        try:
            x = int(Decimal(nums[0]).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            y = int(Decimal(nums[1]).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        except (InvalidOperation, ValueError):
            return s
        return f"({x}, {y})"

    for r in output_rows:
        r["detected_point_absolute"] = round_point_str(r.get("detected_point_absolute", ""))

    # Write output
    out_name = f"{in_path.stem}.filter{in_path.suffix}"
    out_path = out_dir / out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delim)
        writer.writeheader()
        writer.writerows(output_rows)

    # Brief report to stderr
    print(
        f"Read {len(rows)} rows; dropped {drop_unparsable} unparsable, {drop_y} with y>{args.y_max}. "
        f"Wrote {len(output_rows)} rows to {out_path}",
        file=sys.stderr,
    )

    # Also print the output path to stdout for convenience
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
