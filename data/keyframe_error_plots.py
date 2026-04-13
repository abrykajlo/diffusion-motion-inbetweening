"""
Plot per-joint keyframe error from a `keyframe_error.csv` export.

Usage:
    python keyframe_error_plots.py <run_name> [--outdir DIR]
    python keyframe_error_plots.py --csv <path/to/keyframe_error.csv> [--outdir DIR]

When called with a run name, the script looks in
``blender_inferences/`` for a matching inference run folder (exact name
first, then the highest-numbered ``<name>_N`` variant) and reads
``data/keyframe_error.csv`` from it.

Produces:
    keyframe_error_per_joint.png   — bar chart of mean error per joint
    keyframe_error_vs_frame.png    — scatter of error vs frame, colored by joint
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resolve_run import resolve_run_file


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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('run_name', nargs='?', default=None,
                        help='Inference run name to look up in blender_inferences/')
    parser.add_argument('--csv', default=None, dest='csv_path',
                        help='Explicit path to keyframe_error.csv (bypasses run name lookup)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for PNG files (defaults to CSV directory)')
    args = parser.parse_args()

    if args.csv_path:
        csv_path = args.csv_path
        default_outdir = os.path.dirname(os.path.abspath(csv_path))
    elif args.run_name:
        csv_path = resolve_run_file(args.run_name, 'data/keyframe_error.csv')
        print(f'Resolved: {csv_path}')
        # csv lives at <run_dir>/data/keyframe_error.csv — output to <run_dir>/images
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
        default_outdir = os.path.join(run_dir, 'images')
    else:
        parser.error('provide a run name or --csv path')

    outdir = args.outdir or default_outdir
    os.makedirs(outdir, exist_ok=True)

    df = compute_keyframe_error(csv_path)
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
