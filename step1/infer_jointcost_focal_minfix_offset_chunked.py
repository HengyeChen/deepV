#!/usr/bin/env python3
"""
Chunked inference for the joint-cost + focal + offset CNN.

Goal
- Avoid repeatedly scanning a large BED for every image window.
- Split --input into chunks of N rows (default 500), extract the
  corresponding subset from --bed once per chunk, and run inference on
  that chunk using a preloaded model.
  - Optionally write the per-chunk BED slices to disk for reuse/debugging.

Usage (example)
  python scripts/infer_jointcost_focal_minfix_offset_chunked.py \
    --weights scripts/final_jointcost_weights.h5 \
    --input chr4_0_200000kb/conv2/chr4.conv1.0-190999kb.merge.csv \
    --bed /home/nwh/software/temp/loMNase_K562/data/loMNase_K562.chr4.sort.bed \
    --out chr4_0_200000kb/model_infer/detected_points.tsv \
    --thr 0.3 \
    --chunk_size 500

Notes
- This script keeps the model architecture and detection logic equivalent
  to scripts/infer_jointcost_focal_minfix_offset.py.
- It pads images to the global max range of the entire --input so the
  model is built once and reused for all chunks.
- For each chunk, the relevant BED slice is computed in memory and can
  also be saved under <out_dir>/bed_chunks for inspection.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# -------------------------- Image utilities --------------------------
def create_image(image_points, size=(1000, 100)):
    """Create a sparse image with 1.0 at given (x, y) integer-rounded points.
    image_points: iterable of (x, y) in original coordinates (relative to window min_x)
    size: (H, W)
    """
    unique_image_points = set(image_points)
    image = np.zeros(size, dtype=np.float32)
    for x, y in unique_image_points:
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < size[0] and 0 <= yi < size[1]:
            image[xi, yi] = 1.0
    return image


def generate_images_from_bed_df(bed_df, windows_df, max_range_global):
    """Build a batch of images using windows_df and a preloaded BED slice.

    Returns (images[N,Hmax,W,1], min_x_values[N], used_windows_df)
    used_windows_df is a subset of windows_df aligned to the images order.
    """
    if windows_df.empty:
        return np.array([]), [], pd.DataFrame(columns=windows_df.columns)

    images, min_x_values, used_positions = [], [], []

    windows_reset = windows_df.reset_index(drop=True)

    # Expect BED-like columns: [0]=chr, [1]=x, [2]=y, ...
    # We'll filter by x in [start_y, end_y] per window.
    for pos, row in windows_reset.iterrows():
        start = row['start_y']
        end = row['end_y']
        if pd.isna(start) or pd.isna(end):
            continue
        subset = bed_df[(bed_df[1] >= start) & (bed_df[1] <= end)]
        if subset.empty:
            continue
        pts = list(zip(subset[1].values, subset[2].values))
        min_x = float(subset[1].min())
        pts_adj = [(x - min_x, y) for x, y in pts]

        H = int(row['range'])
        image = create_image(pts_adj, size=(H, 100))
        pad_bottom = int(max_range_global) - H
        padded = np.pad(image, ((0, pad_bottom), (0, 0)), mode='constant')
        padded = np.expand_dims(np.expand_dims(padded, 0), -1)

        images.append(padded)
        min_x_values.append(min_x)
        used_positions.append(pos)

    if images:
        used_windows = windows_reset.iloc[used_positions].reset_index(drop=True)
        return np.vstack(images), min_x_values, used_windows
    return np.array([]), [], pd.DataFrame(columns=windows_df.columns)


# -------------------------- Model (same as original) --------------------------
def CNN_architectures(max_range, y_range=100, init_filters=16):
    inp = keras.layers.Input(shape=(max_range, y_range, 1), name='input_image')

    def ConvBNReLU(x, filters, k, name):
        x = keras.layers.Conv2D(filters, k, padding='same', use_bias=False, name=f'{name}_conv')(x)
        x = keras.layers.BatchNormalization(name=f'{name}_bn')(x)
        x = keras.layers.ReLU(name=f'{name}_relu')(x)
        return x

    x = ConvBNReLU(inp, init_filters*4, (7, 7), 'block1')
    x = ConvBNReLU(x,   init_filters*2, (5, 5), 'block2')
    x = ConvBNReLU(x,   init_filters*1, (3, 3), 'block3')
    x = ConvBNReLU(x, init_filters*1, (3, 3), 'refine1')
    x = ConvBNReLU(x, init_filters*2, (3, 3), 'refine2')
    x = ConvBNReLU(x, init_filters*4, (3, 3), 'refine3')

    point_presence = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='point_presence')(x)
    off_raw = keras.layers.Conv2D(2, (1, 1), activation='tanh', name='coord_offset_tanh')(x)
    coord_offset = keras.layers.Lambda(lambda t: 0.5 * t, name='coord_offset')(off_raw)
    return keras.models.Model(inputs=inp, outputs=[point_presence, coord_offset])


# -------------------------- Detection utilities --------------------------
def detect_rows(model, images, min_x_values, windows_df, confidence_threshold=0.3, index_offset=0):
    """Run model and return list of detection rows for a chunk.

    The output schema matches the original script.
    'image_index' is offset by index_offset to be globally unique across chunks.
    """
    if images.size == 0:
        return []

    num_samples = len(images)
    H, W = images.shape[1], images.shape[2]

    # Determine output scaling once
    sample_presence = model(tf.zeros((1, H, W, 1)))[0]
    OH, OW = sample_presence.shape[1], sample_presence.shape[2]
    sh, sw = OH / H, OW / W

    rows = []
    for idx in range(num_samples):
        img = images[idx:idx+1]
        img_min_x = min_x_values[idx]

        # windows_df here matches images order (subset of original windows)
        img_start = windows_df.iloc[idx]['start_y']
        img_end = windows_df.iloc[idx]['end_y']

        presence_pred, offset_pred = model(img, training=False)
        pres = presence_pred.numpy().squeeze()
        offs = offset_pred.numpy().squeeze()

        xs, ys = np.where(pres > confidence_threshold)
        if len(xs) == 0:
            # Top-1 fallback to ensure at least one prediction per image
            flat = pres.reshape(-1)
            topk_idx = np.argmax(flat)
            Hm, Wm = pres.shape
            xs = np.array([topk_idx // Wm])
            ys = np.array([topk_idx % Wm])

        for xi, yi in zip(xs, ys):
            off_x, off_y = offs[xi, yi, 0], offs[xi, yi, 1]
            x_rel = ((xi + 0.5) + off_x) / sh
            y_rel = ((yi + 0.5) + off_y) / sw
            x_rel = float(np.clip(x_rel, 0.0, H - 1))
            y_rel = float(np.clip(y_rel, 0.0, W - 1))
            x_abs = round(x_rel + img_min_x, 2)
            y_abs = round(y_rel, 2)

            rows.append([
                index_offset + idx,
                img_start,
                img_end,
                f'({round(x_rel, 2)}, {round(y_rel, 2)})',
                f'({x_abs}, {y_abs})',
                round(float(pres[xi, yi]), 4),
                'no_matched_reference',
            ])

    return rows


def write_detections(rows, output_csv_path):
    header = [
        'image_index', 'image_original_start', 'image_original_end',
        'detected_point_relative', 'detected_point_absolute',
        'detection_confidence', 'matched_reference_point'
    ]
    df = pd.DataFrame(rows, columns=header)

    # Sort for readability (same behavior as original)
    def parse_abs(s):
        if s == 'no_detected_point':
            return (np.inf, np.inf)
        vals = s.strip('()').split(',')
        return (float(vals[0]), float(vals[1]))
    df[['abs_x', 'abs_y']] = pd.DataFrame(df['detected_point_absolute'].apply(parse_abs).tolist(), index=df.index)
    df = df.sort_values(['image_index', 'abs_x', 'abs_y']).drop(columns=['abs_x', 'abs_y'])

    out_dir = os.path.dirname(os.path.abspath(output_csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv_path, sep='\t', index=False, encoding='utf-8')
    return output_csv_path


# -------------------------- Main (chunked) --------------------------
def main():
    parser = argparse.ArgumentParser(description='Chunked inference using pre-trained CNN weights.')
    parser.add_argument('--weights', required=True, help='Path to model weights .h5 (model.save_weights).')
    parser.add_argument('--input', required=True, help='Path to conv1 merge TSV (tab-separated).')
    parser.add_argument('--bed', required=True, help='Path to BED-like reference file (tab-separated).')
    parser.add_argument('--out', required=True, help='Output TSV path to write detections.')
    parser.add_argument('--thr', type=float, default=0.3, help='Confidence threshold for presence map.')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of rows per input chunk.')
    parser.add_argument('--save_bed_chunks', action='store_true', help='If set, save per-chunk BED slices to <out_dir>/bed_chunks/.')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Warning: Input file {args.input} does not exist, skipping inference")
        sys.exit(0)

    # Load windows (conv1 merge). Validate required columns.
    windows = pd.read_csv(args.input, sep='\t')
    required_cols = {'start_y', 'end_y', 'range'}
    if not required_cols.issubset(set(windows.columns)):
        raise ValueError('Input TSV must contain columns: start_y, end_y, range')

    if windows.empty:
        raise RuntimeError('Input is empty; nothing to infer.')

    # Global maximum range across the full input; used to build a single model.
    Hmax = int(windows['range'].max())
    W = 100

    # Build model once and load weights.
    model = CNN_architectures(Hmax, W)
    model.load_weights(args.weights)

    # Read only the columns we need from BED to reduce memory.
    # Expect: chr (ignored), x, y; no header.
    bed_all = pd.read_csv(args.bed, sep='\t', header=None, usecols=[0, 1, 2])

    out_dir = os.path.dirname(os.path.abspath(args.out))
    bed_chunks_dir = os.path.join(out_dir, 'bed_chunks') if out_dir else 'bed_chunks'
    if args.save_bed_chunks:
        os.makedirs(bed_chunks_dir, exist_ok=True)

    # Chunk the input and process.
    rows_out = []
    global_img_idx = 0
    n = len(windows)
    step = max(1, int(args.chunk_size))

    for start_idx in range(0, n, step):
        end_idx = min(start_idx + step, n)
        chunk = windows.iloc[start_idx:end_idx]

        # Compute union range for the chunk to pre-slice BED once.
        chunk_min = float(chunk['start_y'].min())
        chunk_max = float(chunk['end_y'].max())

        bed_chunk = bed_all[(bed_all[1] >= chunk_min) & (bed_all[1] <= chunk_max)]

        # Optionally save the BED slice to disk for this chunk.
        if args.save_bed_chunks:
            bed_name = os.path.basename(args.bed)
            out_name = f"{os.path.splitext(bed_name)[0]}.chunk_{start_idx:06d}-{end_idx-1:06d}.{int(chunk_min)}-{int(chunk_max)}.bed"
            bed_path = os.path.join(bed_chunks_dir, out_name)
            bed_chunk.to_csv(bed_path, sep='\t', header=False, index=False)

        # Build images from the in-memory BED slice and pad to global Hmax.
        images, min_x_vals, chunk_used = generate_images_from_bed_df(bed_chunk, chunk, Hmax)

        if images.size == 0:
            continue

        # Detect and accumulate rows with global index offset.
        det_rows = detect_rows(
            model,
            images,
            min_x_vals,
            chunk_used,
            confidence_threshold=args.thr,
            index_offset=global_img_idx,
        )
        rows_out.extend(det_rows)

        # Advance global index by the number of images we actually inferred.
        global_img_idx += len(images)

    if not rows_out:
        raise RuntimeError('No detections produced. Check coverage between --input and --bed.')

    write_detections(rows_out, args.out)
    print(f'Wrote detections to: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
