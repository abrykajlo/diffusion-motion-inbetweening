"""
Plot inference time trends from an `inference_time_log.csv` export.

Usage:
    python inference_time_plots.py <path/to/inference_time_log.csv> [--outdir DIR]

Produces:
    inference_time_vs_sequence_length.png
    inference_time_vs_num_keyframes.png
    inference_time_vs_num_constrained_joints.png
    inference_time_correlation_heatmap.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Y_COL = 'inference_time_seconds'


def scatter_with_regression(df, x_col, y_col, out_path):
    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.4)

    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color='red', linewidth=1.5,
                label=f'y = {slope:.4f}x + {intercept:.4f}')

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.text(0.02, 0.98, f'$R^2 = {r2:.3f}$  (n={len(x)})',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='lower right', fontsize=9)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, out_path):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        print('Not enough numeric columns for a correlation heatmap.')
        return

    corr = numeric.corr()
    cols = corr.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(cols) + 2),
                                    max(5, 0.7 * len(cols) + 2)))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr.values[i, j]) > 0.5 else 'black',
                    fontsize=9)

    ax.set_title('Correlation heatmap (numeric columns)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', help='Path to inference_time_log.csv')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for PNG files (defaults to CSV directory)')
    args = parser.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if df.empty:
        print('CSV is empty; nothing to plot.')
        return

    scatter_targets = [
        ('sequence_length', 'inference_time_vs_sequence_length.png'),
        ('num_keyframes', 'inference_time_vs_num_keyframes.png'),
        ('num_constrained_joints', 'inference_time_vs_num_constrained_joints.png'),
    ]

    for x_col, filename in scatter_targets:
        if x_col not in df.columns:
            print(f'Column {x_col!r} missing; skipping.')
            continue
        path = os.path.join(outdir, filename)
        scatter_with_regression(df, x_col, Y_COL, path)
        print(f'Wrote {path}')

    heatmap_path = os.path.join(outdir, 'inference_time_correlation_heatmap.png')
    plot_correlation_heatmap(df, heatmap_path)
    print(f'Wrote {heatmap_path}')


if __name__ == '__main__':
    main()
