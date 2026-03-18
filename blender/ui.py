"""
Blender UI: addon preferences, scene properties, and the sidebar panel.
"""

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, PointerProperty
from bpy.types import AddonPreferences, PropertyGroup, Panel

from .skeleton import HML_JOINT_NAMES
from .constraints import Constraints


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
    )

    project_path: StringProperty(
        name="Project Root",
        description="Absolute path to the diffusion-motion-inbetweening repository root",
        default="",
        subtype='DIR_PATH',
    )

    model_path: StringProperty(
        name="Model Checkpoint",
        description="Path to the .pt model checkpoint file",
        default="",
        subtype='FILE_PATH',
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
    export_path: StringProperty(
        name="Export Path",
        description="Path to save the NPZ export file",
        default="//dmi_export.npz",
        subtype='FILE_PATH',
    )
    import_path: StringProperty(
        name="Import Path",
        description="Path to the NPZ result file to import",
        default="//dmi_result.npz",
        subtype='FILE_PATH',
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
        if constrained:
            sub = box.column(align=True)
            sub.label(text=f"Frame {frame} constraints:")
            for name in constrained:
                sub.label(text=f"  {name}", icon='CONSTRAINT_BONE')

        total = len(constraints)
        if total > 0:
            box.label(text=f"Total constraint markers: {total}")

        # --- Export ---
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(props, "text_prompt")
        box.prop(props, "frame_count")
        box.prop(props, "export_path")
        box.operator("dmi.export", icon='FILE_TICK')

        # --- Inference ---
        box = layout.box()
        box.label(text="Inference", icon='PLAY')
        box.prop(props, "guidance_param")
        box.prop(props, "num_repetitions")
        box.prop(props, "seed")
        box.prop(props, "import_path")
        op_row = box.row(align=True)
        op_row.operator("dmi.run_inference", text="Run Inference", icon='SHADERFX')

        # --- Import ---
        box = layout.box()
        box.label(text="Import", icon='IMPORT')
        box.prop(props, "import_path")
        box.operator("dmi.import_result", icon='FILE_REFRESH')
