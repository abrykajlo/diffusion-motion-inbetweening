"""
Plot foot skating from a `foot_skating.csv` export.

Usage:
    python foot_skating_plots.py <run_name> [--outdir DIR] [--threshold 0.01]
    python foot_skating_plots.py --csv <path/to/foot_skating.csv> [--outdir DIR]

When called with a run name, the script looks in
``blender_inferences/`` for a matching inference run folder (exact name
first, then the highest-numbered ``<name>_N`` variant) and reads
``data/foot_skating.csv`` from it.

Computes per-frame displacement between consecutive frames for each foot.
The foot skating ratio is the proportion of grounded frames where the
displacement exceeds the threshold.

Produces:
    foot_displacement_over_time.png  — line chart with grounded frames highlighted
    foot_skating_ratio.png           — bar chart comparing left vs right ratios
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resolve_run import resolve_run_file


def compute_displacements(df):
    df = df.sort_values('frame').reset_index(drop=True)

    for side in ('left', 'right'):
        dx = df[f'{side}_foot_x'].diff()
        dy = df[f'{side}_foot_y'].diff()
        dz = df[f'{side}_foot_z'].diff()
        df[f'{side}_disp'] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return df


def skating_ratio(df, side, threshold):
    grounded = df[df[f'{side}_foot_grounded'] == 1]
    grounded = grounded.dropna(subset=[f'{side}_disp'])
    if grounded.empty:
        return 0.0, 0, 0
    slipping = (grounded[f'{side}_disp'] > threshold).sum()
    total = len(grounded)
    return slipping / total, int(slipping), int(total)


def plot_displacement_over_time(df, out_path, threshold):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, side, color in ((axes[0], 'left', 'tab:blue'), (axes[1], 'right', 'tab:orange')):
        ax.plot(df['frame'], df[f'{side}_disp'], color=color, label=f'{side} foot displacement')

        grounded_mask = df[f'{side}_foot_grounded'] == 1
        if grounded_mask.any():
            ax.scatter(
                df.loc[grounded_mask, 'frame'],
                df.loc[grounded_mask, f'{side}_disp'],
                s=18, color=color, edgecolor='black', linewidth=0.4,
                label='grounded frame', zorder=3,
            )

        ax.axhline(threshold, color='red', linestyle='--', linewidth=1,
                   label=f'threshold = {threshold}')
        ax.set_ylabel(f'{side} foot disp. (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame')
    fig.suptitle('Foot displacement over time (grounded frames highlighted)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_skating_ratio(left_ratio, right_ratio, out_path, threshold):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['left foot', 'right foot'], [left_ratio, right_ratio],
                  color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('Skating ratio')
    ax.set_ylim(0, max(1.0, left_ratio, right_ratio) * 1.1)
    ax.set_title(f'Foot skating ratio (threshold = {threshold})')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, ratio in zip(bars, (left_ratio, right_ratio)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{ratio:.2%}', ha='center', va='bottom')

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
                        help='Explicit path to foot_skating.csv (bypasses run name lookup)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for PNG files (defaults to <run_dir>/images)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Displacement threshold (m) above which a grounded frame counts as skating')
    args = parser.parse_args()

    if args.csv_path:
        csv_path = args.csv_path
        default_outdir = os.path.dirname(os.path.abspath(csv_path))
    elif args.run_name:
        csv_path = resolve_run_file(args.run_name, 'data/foot_skating.csv')
        print(f'Resolved: {csv_path}')
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
        default_outdir = os.path.join(run_dir, 'images')
    else:
        parser.error('provide a run name or --csv path')

    outdir = args.outdir or default_outdir
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = compute_displacements(df)

    left_ratio, left_slip, left_total = skating_ratio(df, 'left', args.threshold)
    right_ratio, right_slip, right_total = skating_ratio(df, 'right', args.threshold)

    print(f'Left foot:  {left_slip}/{left_total} grounded frames slipping ({left_ratio:.2%})')
    print(f'Right foot: {right_slip}/{right_total} grounded frames slipping ({right_ratio:.2%})')

    disp_path = os.path.join(outdir, 'foot_displacement_over_time.png')
    ratio_path = os.path.join(outdir, 'foot_skating_ratio.png')

    plot_displacement_over_time(df, disp_path, args.threshold)
    plot_skating_ratio(left_ratio, right_ratio, ratio_path, args.threshold)

    print(f'Wrote {disp_path}')
    print(f'Wrote {ratio_path}')


if __name__ == '__main__':
    main()
