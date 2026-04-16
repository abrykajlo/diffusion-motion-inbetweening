"""
Plot a constraint-propagation decay curve from two inference runs.

Usage:
    python propagation_plots.py <baseline_run> <perturbed_run> [--outdir DIR]

Looks in ``blender_inferences/`` for each run folder (exact name first, then
the highest-numbered ``<name>_N`` variant). For each run we read:

    <run>/result.npz     — generated joint_positions [n_frames, 22, 3]
    <run>/export.npz     — Blender-side export with constraint_mask and
                           constrained joint_positions [n_frames, 22, 3]

Per-frame mean joint distance between the two runs' generated motions is
plotted as the decay curve. The perturbed (frame, joint) is detected by
comparing the two ``export.npz`` files at every location constrained in
either run and picking the one with the largest position delta.

Produces (in outdir, default = <perturbed_run_dir>/images):
    propagation_<baseline>_vs_<perturbed>.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from resolve_run import resolve_run_dir

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from data_loaders.humanml_utils import HML_JOINT_NAMES  # noqa: E402


def load_result_positions(run_dir):
    path = os.path.join(run_dir, 'result.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'result.npz not found at {path}')
    return np.load(path)['joint_positions']


def mean_distance_per_frame(baseline_pos, perturbed_pos):
    """[n_frames, 22, 3] pair → (frames, mean_distance_per_frame)."""
    if baseline_pos.shape[1:] != perturbed_pos.shape[1:]:
        raise ValueError(
            f'joint/coord shape mismatch: baseline {baseline_pos.shape} '
            f'vs perturbed {perturbed_pos.shape}'
        )
    n = min(baseline_pos.shape[0], perturbed_pos.shape[0])
    if baseline_pos.shape[0] != perturbed_pos.shape[0]:
        print(f'[warn] frame count differs ({baseline_pos.shape[0]} vs '
              f'{perturbed_pos.shape[0]}); truncating to {n}')
    per_joint = np.linalg.norm(baseline_pos[:n] - perturbed_pos[:n], axis=-1)
    return np.arange(n), per_joint.mean(axis=-1)


def detect_perturbation(baseline_run_dir, perturbed_run_dir):
    """Find the constrained location whose position differs most between the two runs.

    Returns (frame_index, joint_name, magnitude) or None if no difference.
    """
    base = np.load(os.path.join(baseline_run_dir, 'export.npz'))
    pert = np.load(os.path.join(perturbed_run_dir, 'export.npz'))

    b_pos = base['joint_positions']
    p_pos = pert['joint_positions']
    b_mask = base['constraint_mask']
    p_mask = pert['constraint_mask']

    if b_pos.shape != p_pos.shape:
        raise ValueError(
            f'export.npz joint_positions shape mismatch: '
            f'baseline {b_pos.shape} vs perturbed {p_pos.shape}'
        )

    either_mask = b_mask | p_mask
    delta = np.linalg.norm(b_pos - p_pos, axis=-1)
    delta = np.where(either_mask, delta, 0.0)

    if not np.any(delta > 0):
        return None

    fi, ji = np.unravel_index(int(np.argmax(delta)), delta.shape)
    magnitude = float(delta[fi, ji])
    joint_name = HML_JOINT_NAMES[ji] if ji < len(HML_JOINT_NAMES) else f'joint_{ji}'
    return int(fi), joint_name, magnitude


def plot_propagation(frames, distances, base_name, pert_name, detection, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frames, distances, color='steelblue', linewidth=1.5,
            label='mean joint distance')

    title_bits = [f'{base_name} vs {pert_name}']
    if detection is not None:
        frame, joint, magnitude = detection
        ax.axvline(frame, color='red', linestyle='--', linewidth=1.2,
                   label=f'perturbed frame = {frame}')
        title_bits.append(f'joint = {joint}')
        title_bits.append(f'magnitude = {magnitude:.4f}')

    ax.set_title('  |  '.join(title_bits))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean joint distance (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('baseline_run', help='Baseline inference run name')
    parser.add_argument('perturbed_run', help='Perturbed inference run name')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (defaults to <perturbed_run_dir>/images)')
    args = parser.parse_args()

    base_dir = resolve_run_dir(args.baseline_run)
    pert_dir = resolve_run_dir(args.perturbed_run)
    print(f'Baseline:  {base_dir}')
    print(f'Perturbed: {pert_dir}')

    outdir = args.outdir or os.path.join(pert_dir, 'images')
    os.makedirs(outdir, exist_ok=True)

    baseline_pos = load_result_positions(base_dir)
    perturbed_pos = load_result_positions(pert_dir)

    frames, distances = mean_distance_per_frame(baseline_pos, perturbed_pos)
    if len(frames) == 0:
        raise RuntimeError('result.npz files have zero frames')

    detection = detect_perturbation(base_dir, pert_dir)
    if detection is None:
        print('[warn] no constrained-position difference found between export.npz files')
    else:
        frame, joint, mag = detection
        print(f'Detected perturbation: frame={frame}, joint={joint}, magnitude={mag:.4f}')

    base_name = os.path.basename(base_dir)
    pert_name = os.path.basename(pert_dir)
    out_path = os.path.join(outdir, f'propagation_{base_name}_vs_{pert_name}.png')
    plot_propagation(frames, distances, base_name, pert_name, detection, out_path)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
