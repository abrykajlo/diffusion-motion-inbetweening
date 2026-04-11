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
    guidance_param: FloatProperty(
        name="Guidance Scale",
        description="Classifier-free guidance scale for inference",
        default=2.5,
        min=1.0,
        max=10.0,
    )
    num_repetitions: IntProperty(
        name="Repetitions",
        description="Number of motion samples to generate",
        default=1,
        min=1,
        max=10,
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed for reproducibility",
        default=10,
    )
    inference_name: StringProperty(
        name="Name",
        description="Name used for export/result NPZ files (saved to <project>/blender_inferences/)",
        default="inference",
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
        box.prop(props, "num_repetitions")
        box.prop(props, "seed")
        op_row = box.row(align=True)
        op_row.operator("dmi.run_inference", text="Run Inference", icon='SHADERFX')
        
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

        box.operator("dmi.snapshot_constraint_keyframes", text="Snapshot as Constrained", icon='PINNED')
