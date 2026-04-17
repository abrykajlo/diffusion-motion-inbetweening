"""
Blender UI: addon preferences, scene properties, and the sidebar panel.
"""

import json
import os

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, PointerProperty, BoolProperty, EnumProperty
from bpy.types import AddonPreferences, PropertyGroup, Panel

from .skeleton import HML_JOINT_NAMES
from .constraints import Constraints


# ---------------------------------------------------------------------------
# Cached dump-step file list (avoids os.listdir on every panel redraw)
# ---------------------------------------------------------------------------

_dump_steps_cache = {}   # inference_dir -> sorted list of (timestep_int, filepath)
_dump_steps_mtime = {}   # inference_dir -> last scan time


def get_dump_steps(inference_dir):
    """Return sorted list of (timestep, filepath) for dump_step_*.npz in inference_dir.

    Cached per directory; rescans at most once per second.
    """
    import time, re, os
    if not inference_dir or not os.path.isdir(inference_dir):
        return []
    now = time.monotonic()
    if inference_dir in _dump_steps_cache and now - _dump_steps_mtime.get(inference_dir, 0) < 1.0:
        return _dump_steps_cache[inference_dir]
    pattern = re.compile(r'^dump_step_(\d+)\.npz$')
    steps = []
    for name in os.listdir(inference_dir):
        m = pattern.match(name)
        if m:
            steps.append((int(m.group(1)), os.path.join(inference_dir, name)))
    steps.sort(key=lambda x: x[0])
    _dump_steps_cache[inference_dir] = steps
    _dump_steps_mtime[inference_dir] = now
    return steps


# ---------------------------------------------------------------------------
# Preferences dotfile persistence
# ---------------------------------------------------------------------------

_PREFS_KEYS = ("python_path", "project_path", "model_path")

def _prefs_path():
    base = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(base, ".dmi_prefs.json")


def save_prefs(prefs):
    data = {k: getattr(prefs, k) for k in _PREFS_KEYS}
    try:
        with open(_prefs_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def load_prefs(prefs):
    try:
        with open(_prefs_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    for k in _PREFS_KEYS:
        if k in data:
            setattr(prefs, k, data[k])


def _on_pref_update(self, context):
    save_prefs(self)


# ---------------------------------------------------------------------------
# Addon preferences (persistent across blend files)
# ---------------------------------------------------------------------------

class DMI_AddonPreferences(AddonPreferences):
    bl_idname = __package__

    python_path: StringProperty(
        name="Python Executable",
        description=(
            "Path to the Python interpreter that has PyTorch and the project "
            "dependencies installed (e.g. /path/to/venv/bin/python)"
        ),
        default="python",
        subtype='FILE_PATH',
        update=_on_pref_update,
    )

    project_path: StringProperty(
        name="Project Root",
        description="Absolute path to the diffusion-motion-inbetweening repository root",
        default="",
        subtype='DIR_PATH',
        update=_on_pref_update,
    )

    model_path: StringProperty(
        name="Model Checkpoint",
        description="Path to the .pt model checkpoint file",
        default="",
        subtype='FILE_PATH',
        update=_on_pref_update,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Inference Settings", icon='SETTINGS')
        layout.prop(self, "python_path")
        layout.prop(self, "project_path")
        layout.prop(self, "model_path")


# ---------------------------------------------------------------------------
# Per-scene properties
# ---------------------------------------------------------------------------

class DMI_Properties(PropertyGroup):
    text_prompt: StringProperty(
        name="Text Prompt",
        description="Text description for motion generation",
        default="a person walks forward",
    )
    frame_count: IntProperty(
        name="Frame Count",
        description="Number of frames to export (at 20 FPS)",
        default=120,
        min=10,
        max=196,
    )
    inference_name: StringProperty(
        name="Name",
        description="Name used for export/result NPZ files (saved to <project>/blender_inferences/)",
        default="inference",
    )
    guidance_param: FloatProperty(
        name="Guidance",
        description=(
            "Classifier-free guidance scale for the text prompt. "
            "1.0 = no guidance; higher values push generation harder toward the prompt"
        ),
        default=2.5,
        min=0.0,
        soft_max=10.0,
    )
    show_constraint_list: BoolProperty(
        name="Show Constrained Bones",
        default=False,
    )
    active_keyframe_layer: EnumProperty(
        name="Active Layer",
        items=[
            ('CONSTRAINED', "Constrained", "Keyframes from when constraints were set"),
            ('INFERRED',    "Inferred",    "Keyframes produced by the last inference run"),
        ],
        default='CONSTRAINED',
    )
    keyframes_constrained: StringProperty(default='{}')
    keyframes_inferred: StringProperty(default='{}')
    dump_diffusion_steps: BoolProperty(
        name="Dump Steps",
        description="Save intermediate diffusion step predictions for debugging",
        default=False,
    )
    diffusion_step_index: IntProperty(
        name="Step",
        description="Index into the sorted list of dump step files currently being viewed",
        default=-1,
        min=-1,
    )
    last_inference_dir: StringProperty(
        name="Last Inference Directory",
        description="Path to the most recent inference run folder",
        default="",
    )
    import_source: EnumProperty(
        name="Source",
        description="File format to import",
        items=[
            ('NPZ', "DMI NPZ", "NPZ file exported and processed by the DMI addon"),
            ('HML3D', "HumanML3D", "Raw .npy file from the HumanML3D dataset (joint positions or 263-dim features)"),
            ('INFERENCE', "Inference Run", "Load an inference run folder (export.npz + result.npz)"),
        ],
        default='NPZ',
    )


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------

class DMI_PT_Panel(Panel):
    bl_label = "Motion Inbetween"
    bl_idname = "DMI_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Motion Inbetween"

    def draw(self, context):
        layout = self.layout
        props = context.scene.dmi_props

        # --- Armature ---
        box = layout.box()
        box.label(text="Armature", icon='ARMATURE_DATA')
        box.operator("dmi.create_armature", icon='ADD')

        # --- Constraints ---
        box = layout.box()
        box.label(text="Constraints", icon='CON_LOCKTRACK')
        row = box.row(align=True)
        row.operator("dmi.toggle_constraint", text="Toggle Selected")
        row.operator("dmi.constrain_all_bones", text="All Bones")
        box.operator("dmi.clear_constraints", text="Clear All", icon='X')

        constraints = Constraints(context.scene)
        frame = context.scene.frame_current
        constrained = [name for name in HML_JOINT_NAMES if constraints.has(frame, name)]
        total = len(constraints)

        if constrained or total > 0:
            summary = f"{len(constrained)} bone(s) at frame {frame}"
            if total > 0:
                summary += f"  |  {total} total"
            row = box.row()
            row.prop(
                props, "show_constraint_list",
                icon='TRIA_DOWN' if props.show_constraint_list else 'TRIA_RIGHT',
                text=summary,
                emboss=False,
            )
            if props.show_constraint_list and constrained:
                sub = box.column(align=True)
                for name in constrained:
                    sub.label(text=f"  {name}", icon='CONSTRAINT_BONE')

        # --- Inference ---
        box = layout.box()
        box.label(text="Inference", icon='PLAY')
        box.prop(props, "inference_name")
        box.prop(props, "guidance_param")
        box.prop(props, "dump_diffusion_steps")
        op_row = box.row(align=True)
        op_row.operator("dmi.run_inference", text="Run Inference", icon='SHADERFX')

        # Show progress while inference is running
        from .operators import DMI_OT_RunInference
        if DMI_OT_RunInference._timer is not None:
            total = DMI_OT_RunInference._progress_total
            current = DMI_OT_RunInference._progress_current
            if total > 0:
                pct = int(100 * current / total)
                filled = pct // 5
                bar = "\u2588" * filled + "\u2591" * (20 - filled)
                box.label(text=f"{bar}  {current} / {total}")
            else:
                box.label(text="Starting inference...", icon='TIME')

        # --- Export ---
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(props, "text_prompt")
        box.prop(props, "frame_count")
        box.operator("dmi.export", icon='FILE_TICK')

        # --- Import ---
        box = layout.box()
        box.label(text="Import", icon='IMPORT')
        box.prop(props, "import_source")
        box.operator("dmi.import_result", icon='FILE_REFRESH')

        # --- Export CSV ---
        box = layout.box()
        box.label(text="Evaluation CSV Export", icon='FILE_TEXT')
        row = box.row()
        row.enabled = bool(props.last_inference_dir)
        row.operator("dmi.export_csv", icon='FILE_TICK')

        # --- Keyframe Layers ---
        box = layout.box()
        box.label(text="Keyframe Layers", icon='DECORATE_KEYFRAME')

        has_constrained = bool(props.keyframes_constrained and props.keyframes_constrained != '{}')
        has_inferred    = bool(props.keyframes_inferred    and props.keyframes_inferred    != '{}')

        row = box.row(align=True)
        sub = row.row(align=True)
        sub.enabled = has_constrained
        op = sub.operator(
            "dmi.apply_keyframe_layer",
            text="Constrained",
            icon='KEYFRAME' if props.active_keyframe_layer == 'CONSTRAINED' else 'KEYFRAME_HLT',
            depress=(props.active_keyframe_layer == 'CONSTRAINED'),
        )
        op.layer = 'CONSTRAINED'

        sub = row.row(align=True)
        sub.enabled = has_inferred
        op = sub.operator(
            "dmi.apply_keyframe_layer",
            text="Inferred",
            icon='KEYFRAME' if props.active_keyframe_layer == 'INFERRED' else 'KEYFRAME_HLT',
            depress=(props.active_keyframe_layer == 'INFERRED'),
        )
        op.layer = 'INFERRED'

        # --- Diffusion step navigator ---
        dump_steps = get_dump_steps(props.last_inference_dir)
        if dump_steps:
            step_box = box.box()
            step_box.label(text="Diffusion Steps", icon='SEQUENCE')
            row = step_box.row(align=True)
            prev_op = row.operator("dmi.browse_diffusion_step", text="", icon='TRIA_LEFT')
            prev_op.direction = 'PREV'

            idx = props.diffusion_step_index
            n = len(dump_steps)
            if 0 <= idx < n:
                t = dump_steps[idx][0]
                step_label = f"Step {idx + 1} / {n}  (t={t})"
            else:
                step_label = f"Result  ({n} steps available)"
            row.label(text=step_label)

            next_op = row.operator("dmi.browse_diffusion_step", text="", icon='TRIA_RIGHT')
            next_op.direction = 'NEXT'

            # Button to jump back to the final result
            if 0 <= idx < n:
                step_box.operator("dmi.browse_diffusion_step_result", text="Back to Result", icon='LOOP_BACK')
