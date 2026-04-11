"""
CSV export functions for the CondMDI evaluation pipeline.

Each function is independently callable and writes a single CSV file
according to `evaluation_export_spec.md`. All files are written directly
to `<output_dir>/`; the run's inference name (used as test_id) is encoded
in the filename rather than a subdirectory, since `<output_dir>` already
lives inside the per-run folder.
"""

import csv
import json
import os
from datetime import datetime

import bpy

from .skeleton import HML_JOINT_NAMES
from .constraints import Constraints


FLOAT_PRECISION = 6
INFERENCE_LOG_FILE = "inference_time_log.csv"
PROPAGATION_METADATA_FILE = "propagation_metadata.csv"
FOOT_GROUNDED_KEY = "dmi_foot_grounded"
PROPAGATION_META_KEY = "dmi_propagation_meta"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round(v):
    return round(float(v), FLOAT_PRECISION)


def _frame_range(context):
    """(first_blender_frame, frame_count) for the current scene."""
    scene = context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    return start, max(0, end - start + 1)


def _sample_world_positions(context):
    """Return [(frame_index, {joint_name: (x, y, z)}), ...] in Blender world space.

    frame_index is zero-based relative to scene.frame_start so it matches the
    CSV spec's zero-indexed `frame` column.
    """
    obj = context.active_object
    start, n_frames = _frame_range(context)
    original_frame = context.scene.frame_current

    samples = []
    for fi in range(n_frames):
        blender_frame = start + fi
        context.scene.frame_set(blender_frame)
        context.view_layer.update()

        per_bone = {}
        for ji, name in enumerate(HML_JOINT_NAMES):
            pose_bone = obj.pose.bones.get(name)
            if pose_bone is None:
                continue
            if ji == 0:
                world = obj.matrix_world @ pose_bone.head
            else:
                world = obj.matrix_world @ pose_bone.tail
            per_bone[name] = (world.x, world.y, world.z)
        samples.append((fi, blender_frame, per_bone))

    context.scene.frame_set(original_frame)
    return samples


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _append_csv(path, header, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _load_foot_grounded(scene):
    raw = scene.get(FOOT_GROUNDED_KEY, "{}")
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except json.JSONDecodeError:
        return {}


def _load_propagation_meta(scene):
    raw = scene.get(PROPAGATION_META_KEY, "{}")
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except json.JSONDecodeError:
        return {}


def _constraint_counts(scene, n_frames):
    constraints = Constraints(scene)
    num_keyframes = len(constraints.data)
    num_pairs = sum(len(bones) for bones in constraints.data.values())
    density = (num_keyframes / n_frames) if n_frames else 0.0
    return constraints, num_keyframes, num_pairs, density


# ---------------------------------------------------------------------------
# 1. Joint world positions
# ---------------------------------------------------------------------------

def export_joint_positions(output_dir, context=None):
    context = context or bpy.context
    samples = _sample_world_positions(context)

    rows = []
    for fi, _blender_frame, per_bone in samples:
        for name in HML_JOINT_NAMES:
            pos = per_bone.get(name)
            if pos is None:
                continue
            rows.append([fi, name, _round(pos[0]), _round(pos[1]), _round(pos[2])])

    path = os.path.join(output_dir, "joint_positions.csv")
    _write_csv(path, ['frame', 'joint_name', 'x', 'y', 'z'], rows)
    return path


# ---------------------------------------------------------------------------
# 2. Keyframe error
# ---------------------------------------------------------------------------

def export_keyframe_error_data(output_dir, context=None):
    from .operators import _apply_keyframes  # lazy import to avoid cycles

    context = context or bpy.context
    obj = context.active_object
    props = context.scene.dmi_props

    header = [
        'frame', 'joint_name',
        'generated_x', 'generated_y', 'generated_z',
        'keyframe_x', 'keyframe_y', 'keyframe_z', 'is_keyframe',
    ]
    path = os.path.join(output_dir, "keyframe_error.csv")

    has_constrained = bool(props.keyframes_constrained and props.keyframes_constrained != '{}')
    has_inferred = bool(props.keyframes_inferred and props.keyframes_inferred != '{}')

    if not has_inferred:
        # Nothing to compare against — sample current state as generated with no keyframe targets
        samples = _sample_world_positions(context)
        rows = _build_keyframe_error_rows(samples, {}, Constraints(context.scene))
        _write_csv(path, header, rows)
        return path

    prev_layer = props.active_keyframe_layer
    try:
        _apply_keyframes(obj, json.loads(props.keyframes_inferred))
        generated_samples = _sample_world_positions(context)

        keyframe_lookup = {}
        if has_constrained:
            _apply_keyframes(obj, json.loads(props.keyframes_constrained))
            for _fi, blender_frame, per_bone in _sample_world_positions(context):
                keyframe_lookup[blender_frame] = per_bone
    finally:
        restore_raw = props.keyframes_constrained if prev_layer == 'CONSTRAINED' else props.keyframes_inferred
        if restore_raw and restore_raw != '{}':
            _apply_keyframes(obj, json.loads(restore_raw))
        props.active_keyframe_layer = prev_layer

    constraints = Constraints(context.scene)
    rows = _build_keyframe_error_rows(generated_samples, keyframe_lookup, constraints)
    _write_csv(path, header, rows)
    return path


def _build_keyframe_error_rows(generated_samples, keyframe_lookup, constraints):
    rows = []
    for fi, blender_frame, per_bone in generated_samples:
        frame_keyframes = keyframe_lookup.get(blender_frame, {})
        for name in HML_JOINT_NAMES:
            gen = per_bone.get(name)
            if gen is None:
                continue
            is_kf = 1 if constraints.has(blender_frame, name) else 0
            kf = frame_keyframes.get(name) if is_kf else None
            rows.append([
                fi, name,
                _round(gen[0]), _round(gen[1]), _round(gen[2]),
                _round(kf[0]) if kf else '',
                _round(kf[1]) if kf else '',
                _round(kf[2]) if kf else '',
                is_kf,
            ])
    return rows


# ---------------------------------------------------------------------------
# 3. Foot skating
# ---------------------------------------------------------------------------

def export_foot_skating_data(output_dir, context=None):
    context = context or bpy.context
    samples = _sample_world_positions(context)
    grounded = _load_foot_grounded(context.scene)

    rows = []
    for fi, blender_frame, per_bone in samples:
        lf = per_bone.get('left_foot')
        rf = per_bone.get('right_foot')
        if lf is None or rf is None:
            continue
        flags = grounded.get(str(blender_frame), {})
        rows.append([
            fi,
            _round(lf[0]), _round(lf[1]), _round(lf[2]),
            _round(rf[0]), _round(rf[1]), _round(rf[2]),
            int(bool(flags.get('left', 0))),
            int(bool(flags.get('right', 0))),
        ])

    path = os.path.join(output_dir, "foot_skating.csv")
    _write_csv(
        path,
        [
            'frame',
            'left_foot_x', 'left_foot_y', 'left_foot_z',
            'right_foot_x', 'right_foot_y', 'right_foot_z',
            'left_foot_grounded', 'right_foot_grounded',
        ],
        rows,
    )
    return path


# ---------------------------------------------------------------------------
# 4. Inference time
# ---------------------------------------------------------------------------

def export_inference_time(test_id, output_dir, inference_time_seconds, context=None):
    context = context or bpy.context

    _, n_frames = _frame_range(context)
    _, num_keyframes, num_pairs, density = _constraint_counts(context.scene, n_frames)

    row = [
        test_id,
        datetime.utcnow().isoformat(),
        n_frames,
        num_keyframes,
        num_pairs,
        _round(density),
        _round(inference_time_seconds),
    ]
    path = os.path.join(output_dir, INFERENCE_LOG_FILE)
    _append_csv(
        path,
        [
            'test_id', 'timestamp', 'sequence_length', 'num_keyframes',
            'num_constrained_joints', 'keyframe_density', 'inference_time_seconds',
        ],
        row,
    )
    return path


# ---------------------------------------------------------------------------
# 5a. Propagation run
# ---------------------------------------------------------------------------

def export_propagation_run(run_type, output_dir, context=None):
    if run_type not in ('baseline', 'perturbed'):
        raise ValueError(f"run_type must be 'baseline' or 'perturbed', got {run_type!r}")

    context = context or bpy.context
    samples = _sample_world_positions(context)

    rows = []
    for fi, _blender_frame, per_bone in samples:
        for name in HML_JOINT_NAMES:
            pos = per_bone.get(name)
            if pos is None:
                continue
            rows.append([
                run_type, fi, name,
                _round(pos[0]), _round(pos[1]), _round(pos[2]),
            ])

    path = os.path.join(output_dir, f"propagation_{run_type}.csv")
    _write_csv(
        path,
        ['run_type', 'frame', 'joint_name', 'x', 'y', 'z'],
        rows,
    )
    return path


# ---------------------------------------------------------------------------
# 5b. Propagation metadata
# ---------------------------------------------------------------------------

def export_propagation_metadata(output_dir, context=None):
    context = context or bpy.context
    props = context.scene.dmi_props

    meta = _load_propagation_meta(context.scene)
    _, n_frames = _frame_range(context)
    _, num_keyframes, _num_pairs, density = _constraint_counts(context.scene, n_frames)

    dx = float(meta.get('perturbation_dx', 0.0))
    dy = float(meta.get('perturbation_dy', 0.0))
    dz = float(meta.get('perturbation_dz', 0.0))
    magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5

    row = [
        int(meta.get('perturbed_frame', 0)),
        str(meta.get('perturbed_joint', '')),
        _round(magnitude),
        _round(dx), _round(dy), _round(dz),
        n_frames,
        num_keyframes,
        _round(density),
        int(props.seed),
    ]
    path = os.path.join(output_dir, PROPAGATION_METADATA_FILE)
    _append_csv(
        path,
        [
            'perturbed_frame', 'perturbed_joint',
            'perturbation_magnitude', 'perturbation_dx', 'perturbation_dy', 'perturbation_dz',
            'sequence_length', 'num_keyframes', 'keyframe_density', 'random_seed',
        ],
        row,
    )
    return path


# ---------------------------------------------------------------------------
# 6. Batch
# ---------------------------------------------------------------------------

def export_all_evaluation_data(output_dir, context=None):
    context = context or bpy.context
    return [
        export_joint_positions(output_dir, context),
        export_keyframe_error_data(output_dir, context),
        export_foot_skating_data(output_dir, context),
    ]
