"""
Plot per-joint keyframe error from a `keyframe_error.csv` export.

Usage:
    python keyframe_error_plots.py <path/to/keyframe_error.csv> [--outdir DIR]

Produces:
    keyframe_error_per_joint.png   — bar chart of mean error per joint
    keyframe_error_vs_frame.png    — scatter of error vs frame, colored by joint
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_keyframe_error(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['is_keyframe'] == 1].copy()
    dx = df['generated_x'] - df['keyframe_x']
    dy = df['generated_y'] - df['keyframe_y']
    dz = df['generated_z'] - df['keyframe_z']
    df['error'] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return df


def plot_mean_error_per_joint(df, out_path):
    per_joint = df.groupby('joint_name')['error'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(per_joint.index, per_joint.values, color='steelblue')
    ax.set_xlabel('Joint')
    ax.set_ylabel('Mean keyframe error (m)')
    ax.set_title('Mean keyframe error per joint')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_vs_frame(df, out_path):
    joints = sorted(df['joint_name'].unique())
    cmap = plt.get_cmap('tab20', len(joints))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, joint in enumerate(joints):
        sub = df[df['joint_name'] == joint]
        ax.scatter(sub['frame'], sub['error'], s=20, color=cmap(i), label=joint, alpha=0.8)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Keyframe error (m)')
    ax.set_title('Keyframe error vs frame')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', help='Path to keyframe_error.csv')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for PNG files (defaults to CSV directory)')
    args = parser.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(outdir, exist_ok=True)

    df = compute_keyframe_error(args.csv_path)
    if df.empty:
        print('No keyframed rows found; nothing to plot.')
        return

    bar_path = os.path.join(outdir, 'keyframe_error_per_joint.png')
    scatter_path = os.path.join(outdir, 'keyframe_error_vs_frame.png')

    plot_mean_error_per_joint(df, bar_path)
    plot_error_vs_frame(df, scatter_path)

    print(f'Wrote {bar_path}')
    print(f'Wrote {scatter_path}')


if __name__ == '__main__':
    main()
