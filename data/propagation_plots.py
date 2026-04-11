"""
Plot constraint-propagation decay curves from paired baseline/perturbed runs.

Expects an input folder containing:
    metadata.csv                       — columns: test_id, perturbed_frame,
                                         perturbation_magnitude, perturbed_joint
    baseline_<test_id>.csv             — columns: frame, joint_name, x, y, z
    perturbed_<test_id>.csv            — columns: frame, joint_name, x, y, z

Usage:
    python propagation_plots.py <input_dir> [--outdir DIR]

Produces (in outdir, default = <input_dir>/plots):
    propagation_<test_id>.png          — per-test per-frame mean joint distance
    propagation_aggregate.png          — all tests overlaid by relative frame,
                                         with the average decay curve
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pair(input_dir, test_id):
    base_path = os.path.join(input_dir, f'baseline_{test_id}.csv')
    pert_path = os.path.join(input_dir, f'perturbed_{test_id}.csv')
    if not os.path.exists(base_path) or not os.path.exists(pert_path):
        return None
    return pd.read_csv(base_path), pd.read_csv(pert_path)


def mean_distance_per_frame(baseline, perturbed):
    merged = baseline.merge(
        perturbed, on=['frame', 'joint_name'], suffixes=('_b', '_p')
    )
    dx = merged['x_b'] - merged['x_p']
    dy = merged['y_b'] - merged['y_p']
    dz = merged['z_b'] - merged['z_p']
    merged['dist'] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return merged.groupby('frame', as_index=False)['dist'].mean().sort_values('frame')


def plot_single_test(series, meta_row, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series['frame'], series['dist'], color='steelblue', linewidth=1.5,
            label='mean joint distance')

    perturbed_frame = int(meta_row['perturbed_frame'])
    ax.axvline(perturbed_frame, color='red', linestyle='--', linewidth=1.2,
               label=f'perturbed frame = {perturbed_frame}')

    title_bits = [f"test_id = {meta_row['test_id']}"]
    if 'perturbed_joint' in meta_row and pd.notna(meta_row['perturbed_joint']):
        title_bits.append(f"joint = {meta_row['perturbed_joint']}")
    if 'perturbation_magnitude' in meta_row and pd.notna(meta_row['perturbation_magnitude']):
        title_bits.append(f"magnitude = {float(meta_row['perturbation_magnitude']):.4f}")
    ax.set_title('  |  '.join(title_bits))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean joint distance (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregate(all_series, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    combined = []
    for test_id, rel_series in all_series:
        ax.plot(rel_series['rel_frame'], rel_series['dist'],
                alpha=0.35, linewidth=1.0, label=test_id)
        combined.append(rel_series)

    if combined:
        stacked = pd.concat(combined, ignore_index=True)
        mean_curve = stacked.groupby('rel_frame', as_index=False)['dist'].mean()
        mean_curve = mean_curve.sort_values('rel_frame')
        ax.plot(mean_curve['rel_frame'], mean_curve['dist'],
                color='black', linewidth=2.5, label='mean across tests')

    ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='perturbed frame')
    ax.set_xlabel('Frames from perturbed frame')
    ax.set_ylabel('Mean joint distance (m)')
    ax.set_title('Propagation decay curves (aligned on perturbed frame)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='Folder with baseline/perturbed CSVs and metadata.csv')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for PNG files (defaults to <input_dir>/plots)')
    args = parser.parse_args()

    input_dir = args.input_dir
    outdir = args.outdir or os.path.join(input_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)

    meta_path = os.path.join(input_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'metadata.csv not found in {input_dir}')

    meta = pd.read_csv(meta_path)
    required = {'test_id', 'perturbed_frame'}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f'metadata.csv missing columns: {missing}')

    aggregate_series = []

    for _, row in meta.iterrows():
        test_id = str(row['test_id'])
        pair = load_pair(input_dir, test_id)
        if pair is None:
            print(f'[skip] missing CSV pair for test_id={test_id}')
            continue
        baseline, perturbed = pair

        series = mean_distance_per_frame(baseline, perturbed)
        if series.empty:
            print(f'[skip] no shared (frame, joint) rows for test_id={test_id}')
            continue

        single_path = os.path.join(outdir, f'propagation_{test_id}.png')
        plot_single_test(series, row, single_path)
        print(f'Wrote {single_path}')

        rel = series.copy()
        rel['rel_frame'] = rel['frame'] - int(row['perturbed_frame'])
        aggregate_series.append((test_id, rel[['rel_frame', 'dist']]))

    if aggregate_series:
        agg_path = os.path.join(outdir, 'propagation_aggregate.png')
        plot_aggregate(aggregate_series, agg_path)
        print(f'Wrote {agg_path}')
    else:
        print('No test cases plotted; aggregate skipped.')


if __name__ == '__main__':
    main()
