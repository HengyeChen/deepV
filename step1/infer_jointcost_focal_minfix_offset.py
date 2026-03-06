#!/usr/bin/env python3
"""
Inference script for the joint-cost + focal + offset CNN.

- Builds the same model architecture as train_jointcost_focal_minfix_offset.py
- Loads pre-trained weights from a .h5 (model.save_weights) file
- Reads a conv1 merge TSV (with columns: start_y, end_y, range, start_x, end_x)
- Generates input images from a BED-like reference file (chr, x, y, ...)
- Runs prediction and writes detected points to a TSV in this directory

Usage example:
  python predict_points_from_h5.py \
    --weights /path/to/final_jointcost_weights.h5 \
    --input  /home/nwh/test/test_CNN_model/input/conv1.100000-101000kb.merge.csv \
    --bed    /home/nwh/test/test_CNN_model/ref_data/loMNase_K562.chr1.100000-110000kb.bed \
    --out    detected_points.tsv \
    --thr    0.3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# -------------------------- Data utils (mirrors training) --------------------------
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


def get_image_points(file_path, start, end, return_min_x=False):
    """Read a BED-like file and collect (x, y) points with x in [start, end].
    Assumes file has no header and columns:
      [0]=chr, [1]=x, [2]=y, [3]=... (others ignored)
    Returns points with x shifted by min_x so window starts at 0.
    """
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except Exception:
        return ([], start) if return_min_x else []
    filtered = data[(data[1] >= start) & (data[1] <= end)]
    if filtered.empty:
        return ([], start) if return_min_x else []
    pts = list(zip(filtered[1], filtered[2]))
    min_x = min(filtered[1])
    adjusted = [(x - min_x, y) for x, y in pts]
    return (adjusted, min_x) if return_min_x else adjusted


def generate_images_and_image_points(file_path, result_df):
    """Create a batch of images using conv1-merge windows and reference points.
    - Pads each image at bottom so all share the same height (=max range)
    - Returns (images[N,Hmax,W,1], min_x_values[N], Hmax, image_points_list[N])
    """
    if result_df.empty:
        return np.array([]), [], 0, []
    max_range = int(result_df['range'].max())

    images, min_x_values, image_points_list = [], [], []
    for _, row in result_df.iterrows():
        start = row['start_y']; end = row['end_y']
        if pd.isna(start) or pd.isna(end):
            continue
        pts, min_x = get_image_points(file_path, start, end, return_min_x=True)
        if not pts:
            continue
        H = int(row['range'])
        image = create_image(pts, size=(H, 100))
        pad_bottom = max_range - H
        padded = np.pad(image, ((0, pad_bottom), (0, 0)), mode='constant')
        padded = np.expand_dims(np.expand_dims(padded, 0), -1)
        images.append(padded)
        min_x_values.append(min_x)
        image_points_list.append(pts)
    if images:
        return np.vstack(images), min_x_values, max_range, image_points_list
    return np.array([]), [], 0, []


# -------------------------- Model (same as training) --------------------------
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


# -------------------------- Detection export (no ground-truth needed) --------------------------
def generate_detection_csv(model, images, min_x_values, conv1_merge_df,
                           output_csv_path, confidence_threshold=0.3,
                           use_topk_fallback=True):
    """Run model on images and write detected points.
    - If no cells exceed threshold and fallback enabled, picks top-k cells where k
      equals the number of points in the reference list (not available here),
      so we fallback to k=1 in that case.
    - Output columns mirror the training script, with matched_reference set to
      'no_matched_reference'.
    """
    num_samples = len(images)
    H, W = images.shape[1], images.shape[2]
    sample_presence = model(tf.zeros((1, H, W, 1)))[0]
    OH, OW = sample_presence.shape[1], sample_presence.shape[2]
    sh, sw = OH / H, OW / W

    header = [
        'image_index', 'image_original_start', 'image_original_end',
        'detected_point_relative', 'detected_point_absolute',
        'detection_confidence', 'matched_reference_point'
    ]
    rows = []

    for img_idx in range(num_samples):
        img = images[img_idx:img_idx+1]
        img_min_x = min_x_values[img_idx]
        img_start = conv1_merge_df.iloc[img_idx]['start_y']
        img_end = conv1_merge_df.iloc[img_idx]['end_y']

        presence_pred, offset_pred = model(img, training=False)
        pres = presence_pred.numpy().squeeze()
        offs = offset_pred.numpy().squeeze()

        xs, ys = np.where(pres > confidence_threshold)
        if len(xs) == 0 and use_topk_fallback:
            flat = pres.reshape(-1)
            k = 1  # no ground truth available; choose at least one top candidate
            topk_idx = np.argpartition(-flat, k-1)[:k]
            Hm, Wm = pres.shape
            xs = topk_idx // Wm
            ys = topk_idx % Wm
        if len(xs) == 0:
            rows.append([img_idx, img_start, img_end, 'no_detected_point', 'no_detected_point', 0.0, 'no_matched_reference'])
            continue

        for xi, yi in zip(xs, ys):
            off_x, off_y = offs[xi, yi, 0], offs[xi, yi, 1]
            x_rel = ((xi + 0.5) + off_x) / sh
            y_rel = ((yi + 0.5) + off_y) / sw
            x_rel = float(np.clip(x_rel, 0.0, H - 1))
            y_rel = float(np.clip(y_rel, 0.0, W - 1))
            x_abs = round(x_rel + img_min_x, 2)
            y_abs = round(y_rel, 2)
            rows.append([
                img_idx, img_start, img_end,
                f'({round(x_rel, 2)}, {round(y_rel, 2)})',
                f'({x_abs}, {y_abs})',
                round(float(pres[xi, yi]), 4),
                'no_matched_reference'
            ])

    df = pd.DataFrame(rows, columns=header)
    # sort by absolute coordinates for readability
    def parse_abs(s):
        if s == 'no_detected_point':
            return (np.inf, np.inf)
        vals = s.strip('()').split(',')
        return (float(vals[0]), float(vals[1]))
    df[['abs_x', 'abs_y']] = pd.DataFrame(df['detected_point_absolute'].apply(parse_abs).tolist(), index=df.index)
    df = df.sort_values(['image_index', 'abs_x', 'abs_y']).drop(columns=['abs_x', 'abs_y'])
    df.to_csv(output_csv_path, sep='\t', index=False, encoding='utf-8')
    return output_csv_path


def main():
    parser = argparse.ArgumentParser(description='Predict points using pre-trained CNN weights.')
    parser.add_argument('--weights', required=True, help='Path to model weights .h5 file (from model.save_weights).')
    parser.add_argument('--input', default='/home/nwh/test/test_CNN_model/input/conv1.100000-101000kb.merge.csv', help='Path to conv1 merge TSV (tab-separated).')
    parser.add_argument('--bed', default='/home/nwh/test/test_CNN_model/ref_data/loMNase_K562.chr1.100000-110000kb.bed', help='Path to BED-like reference file used to build images.')
    parser.add_argument('--out', default='detected_points.tsv', help='Output TSV path to write detections.')
    parser.add_argument('--thr', type=float, default=0.3, help='Confidence threshold for presence map.')
    args = parser.parse_args()

    # Load conv1 merge windows
    conv1_df = pd.read_csv(args.input, sep='\t')
    if not {'start_y', 'end_y', 'range'}.issubset(set(conv1_df.columns)):
        raise ValueError('Input TSV must contain columns: start_y, end_y, range')

    # Build images from BED-like reference
    images, min_x_values, max_range, _ = generate_images_and_image_points(args.bed, conv1_df)
    if images.size == 0:
        raise RuntimeError('No images were generated. Check --input and --bed coverage.')

    # Build model with matching spatial size and load weights
    H, W = images.shape[1], images.shape[2]
    model = CNN_architectures(H, W)
    model.load_weights(args.weights)

    # Predict and export
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generate_detection_csv(model, images, min_x_values, conv1_df, out_path, confidence_threshold=args.thr)
    print(f'Wrote detections to: {out_path}')


if __name__ == '__main__':
    main()

