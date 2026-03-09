bl_info = {
    "name": "Diffusion Motion Inbetweening",
    "author": "DMI",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Motion Inbetween",
    "description": "Create SMPL armatures, mark joint constraints per frame, and export/import for diffusion-based motion generation",
    "category": "Animation",
}

import bpy
import json
import os
from bpy.props import StringProperty, IntProperty, PointerProperty
from bpy.types import PropertyGroup, Operator, Panel
from mathutils import Vector

import numpy as np

# ---------------------------------------------------------------------------
# Constants: 22-joint SMPL skeleton used by HumanML3D
# ---------------------------------------------------------------------------

HML_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# Parent index for each joint (-1 = root)
JOINT_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Rest-pose joint positions relative to pelvis (network coords: Y-up, Z-forward)
# Extracted from dataset/000021.npy reference skeleton, first frame
REST_POSITIONS_NETWORK = [
    [0.000000, 0.000000, 0.000000],    # 0  pelvis
    [-0.017997, -0.084810, -0.055795], # 1  left_hip
    [0.003173, -0.091322, 0.061002],   # 2  right_hip
    [-0.049963, 0.121739, 0.001620],   # 3  spine1
    [-0.013664, -0.475937, -0.099533], # 4  left_knee
    [-0.017759, -0.478489, 0.104146],  # 5  right_knee
    [-0.023445, 0.262192, -0.006473],  # 6  spine2
    [-0.088398, -0.899901, -0.061073], # 7  left_ankle
    [-0.072430, -0.899016, 0.066794],  # 8  right_ankle
    [-0.019436, 0.319396, -0.005034],  # 9  spine3
    [0.032593, -0.955685, -0.114060],  # 10 left_foot
    [0.061955, -0.957583, 0.096339],   # 11 right_foot
    [-0.037010, 0.537973, 0.000394],   # 12 neck
    [-0.037392, 0.433182, -0.080020],  # 13 left_collar
    [-0.026214, 0.438880, 0.073897],   # 14 right_collar
    [0.031190, 0.606817, -0.034831],   # 15 head
    [-0.070935, 0.416979, -0.206194],  # 16 left_shoulder
    [-0.022817, 0.436870, 0.196837],   # 17 right_shoulder
    [-0.101886, 0.162039, -0.213797],  # 18 left_elbow
    [-0.065000, 0.180017, 0.235381],   # 19 right_elbow
    [-0.021291, -0.089844, -0.242452], # 20 left_wrist
    [0.023536, -0.069744, 0.285665],   # 21 right_wrist
]


def net_to_blender(pos):
    """Convert network coords (X-right, Y-up, Z-forward) to Blender (X-right, Y-forward, Z-up)."""
    return Vector((pos[0], pos[2], pos[1]))


def blender_to_net(pos):
    """Convert Blender coords (X-right, Y-forward, Z-up) to network (X-right, Y-up, Z-forward)."""
    return [pos[0], pos[2], pos[1]]

def rest_blender():
    rest_bl = [net_to_blender(p) for p in REST_POSITIONS_NETWORK]
    min_z = min(v.z for v in rest_bl)
    rest_offset = Vector((0, 0, -min_z))
    rest_bl = [v + rest_offset for v in rest_bl]
    return rest_bl
# ---------------------------------------------------------------------------
# Properties
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
        description="Path to the NPZ file to import",
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


# ---------------------------------------------------------------------------
# Constraint storage helpers
# ---------------------------------------------------------------------------

CONSTRAINTS_KEY = "dmi_constraints"

class Constraints:
    def __init__(self, scene):
        self.scene = scene
        self.load()
    
    def __len__(self):
        return len(self.data)

    def save(self):
        self.scene[CONSTRAINTS_KEY] = json.dumps(self.data)

    def load(self):
        raw = self.scene.get(CONSTRAINTS_KEY, "{}")
        if isinstance(raw, str):
            self.data = json.loads(raw)
        self.data = {}

    def set(self, frame: int, bone: str):
        if frame not in self.data:
            self.data[frame] = {}
        self.data[frame][bone] = True

    def has(self, frame: int, bone: str):
        return frame in self.data and self.data[frame][bone]
    
    def remove(self, frame: int, bone: str):
        if frame in self.data:
            del self.data[frame][bone]
    
    def clear(self):
        self.data = {}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class DMI_OT_CreateArmature(Operator):
    bl_idname = "dmi.create_armature"
    bl_label = "Create SMPL Armature"
    bl_description = "Create a 22-bone SMPL skeleton armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Compute rest positions in Blender coordinates
        rest_bl = rest_blender()

        # Create armature
        arm_data = bpy.data.armatures.new("DMI_Armature")
        arm_obj = bpy.data.objects.new("DMI_Armature", arm_data)
        context.collection.objects.link(arm_obj)
        context.view_layer.objects.active = arm_obj
        arm_obj.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')

        bones = {}
        for i, name in enumerate(HML_JOINT_NAMES):
            bone = arm_data.edit_bones.new(name)
            head = rest_bl[i]
            # Bone tail: extend a small amount from head in a reasonable direction
            if i == 0:
                # Root bone points upward
                bone.head = head
                bone.tail = head + Vector((0, 0, 0.1))
            else:
                parent_idx = JOINT_PARENTS[i]
                bone.head = rest_bl[parent_idx]
                bone.tail = head
                # Avoid zero-length bones
                if (bone.tail - bone.head).length < 0.001:
                    bone.tail = bone.head + Vector((0, 0, 0.01))
            bones[name] = bone

        # Set parent relationships
        for i, name in enumerate(HML_JOINT_NAMES):
            if JOINT_PARENTS[i] >= 0:
                parent_name = HML_JOINT_NAMES[JOINT_PARENTS[i]]
                bones[name].parent = bones[parent_name]

        bpy.ops.object.mode_set(mode='OBJECT')

        # Set scene FPS to 20
        context.scene.render.fps = 20
        context.scene.render.fps_base = 1.0

        # Initialize empty constraint dict
        constraints = Constraints(context.scene)
        constraints.save()

        self.report({'INFO'}, "Created 22-bone SMPL armature")
        return {'FINISHED'}


class DMI_OT_ToggleConstraint(Operator):
    bl_idname = "dmi.toggle_constraint"
    bl_label = "Toggle Constraint"
    bl_description = "Toggle constraint on selected bones at current frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'WARNING'}, "Select an armature first")
            return {'CANCELLED'}

        frame = context.scene.frame_current
        constraints = Constraints(context.scene)
        toggled = []

        for bone in context.selected_pose_bones or []:
            if bone.name not in HML_JOINT_NAMES:
                continue
            if constraints.has(frame, bone.name):
                constraints.remove(frame, bone.name)
                toggled.append(f"-{bone.name}")
            else:
                constraints.set(frame, bone.name)
                toggled.append(f"+{bone.name}")

        constraints.save()

        if toggled:
            self.report({'INFO'}, f"Frame {frame}: {', '.join(toggled)}")
        else:
            self.report({'WARNING'}, "No valid bones selected")
        return {'FINISHED'}


class DMI_OT_ConstrainAllBones(Operator):
    bl_idname = "dmi.constrain_all_bones"
    bl_label = "Constrain All Bones at Frame"
    bl_description = "Mark all 22 bones as constrained at the current frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        frame = context.scene.frame_current
        constraints = Constraints(context.scene)
        for name in HML_JOINT_NAMES:
            constraints.set(frame, name)
        constraints.save()
        self.report({'INFO'}, f"All bones constrained at frame {frame}")
        return {'FINISHED'}


class DMI_OT_ClearAllConstraints(Operator):
    bl_idname = "dmi.clear_constraints"
    bl_label = "Clear All Constraints"
    bl_description = "Remove all constraint markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        constraints = Constraints(context.scene)
        constraints.clear()
        constraints.save()
        self.report({'INFO'}, "All constraints cleared")
        return {'FINISHED'}


class DMI_OT_Export(Operator):
    bl_idname = "dmi.export"
    bl_label = "Export for Inference"
    bl_description = "Export joint positions and constraint mask to NPZ"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.dmi_props
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        n_frames = props.frame_count
        joint_positions = np.zeros((n_frames, 22, 3), dtype=np.float32)
        constraint_mask = np.zeros((n_frames, 22), dtype=bool)

        constraints = Constraints(context.scene)

        original_frame = context.scene.frame_current

        for fi in range(n_frames):
            frame = fi + 1
            context.scene.frame_set(frame)  # Blender frames are 1-based
            context.view_layer.update()

            for ji, name in enumerate(HML_JOINT_NAMES):
                pose_bone = obj.pose.bones.get(name)
                if pose_bone is None:
                    continue

                if ji == 0:
                    # Root bone: use tail position (which is where the joint is)
                    # Actually for the root, the joint position IS the head of child bones
                    # But pelvis bone head = pelvis position after offset
                    # Use the bone's world-space head position for the pelvis
                    # The pelvis bone head is at the pelvis rest position
                    world_pos = obj.matrix_world @ pose_bone.head
                else:
                    # Non-root bones: the joint position is at the bone's tail
                    world_pos = obj.matrix_world @ pose_bone.tail

                net_pos = blender_to_net(world_pos)
                joint_positions[fi, ji] = net_pos

                # Check constraint
                if constraints.has(frame, name): # frames are 1-based in Blender
                    constraint_mask[fi, ji] = True

        context.scene.frame_set(original_frame)

        # Save NPZ
        export_path = bpy.path.abspath(props.export_path)
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.', exist_ok=True)
        np.savez(
            export_path,
            joint_positions=joint_positions,
            constraint_mask=constraint_mask,
            text_prompt=props.text_prompt,
            fps=20,
        )

        self.report({'INFO'}, f"Exported {n_frames} frames to {export_path}")
        return {'FINISHED'}


class DMI_OT_Import(Operator):
    bl_idname = "dmi.import_result"
    bl_label = "Import Result"
    bl_description = "Load NPZ with joint positions and apply as animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.dmi_props
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        import_path = bpy.path.abspath(props.import_path)
        if not os.path.exists(import_path):
            self.report({'ERROR'}, f"File not found: {import_path}")
            return {'CANCELLED'}

        data = np.load(import_path, allow_pickle=True)
        joint_positions = data['joint_positions']  # [n_frames, 22, 3] in network coords
        n_frames = joint_positions.shape[0]

        # Ensure we're in pose mode
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='POSE')

        # Compute rest-pose data for IK
        rest_bl = rest_blender()

        for fi in range(n_frames):
            frame = fi + 1
            context.scene.frame_set(frame)

            positions_bl = [net_to_blender(joint_positions[fi, ji]) for ji in range(22)]

            arm_matrix_inv = obj.matrix_world.inverted()

            for ji in range(22):
                bone = obj.pose.bones[HML_JOINT_NAMES[ji]]
                target_world = positions_bl[ji]

                if ji == 0:
                    rest_world = obj.matrix_world @ bone.bone.head_local
                    delta_local = arm_matrix_inv @ (target_world - rest_world)
                    bone.location = delta_local
                    bone.keyframe_insert(data_path="location", frame=frame)

            
        bpy.ops.object.mode_set(mode='OBJECT')

        # Update scene frame range
        context.scene.frame_start = 1
        context.scene.frame_end = n_frames

        self.report({'INFO'}, f"Imported {n_frames} frames from {import_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI Panel
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

        # Show constrained bones at current frame
        constraints = Constraints(context.scene)
        frame = context.scene.frame_current
        constrained = [name for name in HML_JOINT_NAMES
                       if constraints.has(frame, name)]
        if constrained:
            sub = box.column(align=True)
            sub.label(text=f"Frame {frame} constraints:")
            for name in constrained:
                sub.label(text=f"  {name}", icon='CONSTRAINT_BONE')

        # Count total constrained frames
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

        # --- Import ---
        box = layout.box()
        box.label(text="Import", icon='IMPORT')
        box.prop(props, "import_path")
        box.operator("dmi.import_result", icon='FILE_REFRESH')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    DMI_Properties,
    DMI_OT_CreateArmature,
    DMI_OT_ToggleConstraint,
    DMI_OT_ConstrainAllBones,
    DMI_OT_ClearAllConstraints,
    DMI_OT_Export,
    DMI_OT_Import,
    DMI_PT_Panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dmi_props = PointerProperty(type=DMI_Properties)


def unregister():
    del bpy.types.Scene.dmi_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
