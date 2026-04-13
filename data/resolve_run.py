"""
Resolve inference run paths from a run name.

Looks in ``blender_inferences/`` for a matching folder (exact name first,
then the highest-numbered ``<name>_N`` variant).
"""

import os
import re

# Project root is one level above this script's directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INFERENCES_DIR = os.path.join(_PROJECT_ROOT, 'blender_inferences')


def resolve_run_dir(run_name, inferences_dir=None):
    """Find the inference run directory for *run_name*.

    Search order:
      1. Exact match: ``<run_name>/``
      2. Numbered variants ``<run_name>_N`` — pick the highest N.

    Returns the absolute path to the run directory or raises FileNotFoundError.
    """
    base = inferences_dir or _INFERENCES_DIR

    exact = os.path.join(base, run_name)
    if os.path.isdir(exact):
        return exact

    pattern = re.compile(re.escape(run_name) + r'_(\d+)$')
    best_n = -1
    best_path = None
    for entry in os.listdir(base):
        m = pattern.match(entry)
        if m and int(m.group(1)) > best_n:
            candidate = os.path.join(base, entry)
            if os.path.isdir(candidate):
                best_n = int(m.group(1))
                best_path = candidate

    if best_path is not None:
        return best_path

    raise FileNotFoundError(
        f"No inference run '{run_name}' found in {base}"
    )


def resolve_run_file(run_name, relative_path, inferences_dir=None):
    """Find a file inside an inference run directory.

    *relative_path* is relative to the run directory,
    e.g. ``data/keyframe_error.csv``.

    Returns the absolute file path or raises FileNotFoundError.
    """
    run_dir = resolve_run_dir(run_name, inferences_dir)
    path = os.path.join(run_dir, relative_path)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"File '{relative_path}' not found in run directory {run_dir}"
    )
