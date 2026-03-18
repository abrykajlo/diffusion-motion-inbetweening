"""
Blender operators for the DMI addon:
  - Create SMPL armature
  - Toggle / constrain / clear constraints
  - Export joint positions to NPZ
  - Run inference (launches blender_inference.py as a subprocess)
  - Import NPZ result as animation
"""

import os
import queue
import subprocess
import threading

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from .skeleton import (
    HML_JOINT_NAMES, JOINT_PARENTS,
    net_to_blender, blender_to_net, rest_blender,
)
from .constraints import Constraints


# ---------------------------------------------------------------------------
# Create armature
# ---------------------------------------------------------------------------

class DMI_OT_CreateArmature(Operator):
    bl_idname = "dmi.create_armature"
    bl_label = "Create SMPL Armature"
    bl_description = "Create a 22-bone SMPL skeleton armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rest_bl = rest_blender()

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
            if i == 0:
                bone.head = head
                bone.tail = head + Vector((0, 0, 0.1))
            else:
                parent_idx = JOINT_PARENTS[i]
                bone.head = rest_bl[parent_idx]
                bone.tail = head
                if (bone.tail - bone.head).length < 0.001:
                    bone.tail = bone.head + Vector((0, 0, 0.01))
            bones[name] = bone

        for i, name in enumerate(HML_JOINT_NAMES):
            if JOINT_PARENTS[i] >= 0:
                parent_name = HML_JOINT_NAMES[JOINT_PARENTS[i]]
                bones[name].parent = bones[parent_name]

        bpy.ops.object.mode_set(mode='OBJECT')

        context.scene.render.fps = 20
        context.scene.render.fps_base = 1.0

        constraints = Constraints(context.scene)
        constraints.save()

        self.report({'INFO'}, "Created 22-bone SMPL armature")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Constraint operators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

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
            context.scene.frame_set(frame)
            context.view_layer.update()

            for ji, name in enumerate(HML_JOINT_NAMES):
                pose_bone = obj.pose.bones.get(name)
                if pose_bone is None:
                    continue

                if ji == 0:
                    world_pos = obj.matrix_world @ pose_bone.head
                else:
                    world_pos = obj.matrix_world @ pose_bone.tail

                joint_positions[fi, ji] = blender_to_net(world_pos)

                if constraints.has(frame, name):
                    constraint_mask[fi, ji] = True

        context.scene.frame_set(original_frame)

        export_path = bpy.path.abspath(props.export_path)
        os.makedirs(
            os.path.dirname(export_path) if os.path.dirname(export_path) else '.',
            exist_ok=True,
        )
        np.savez(
            export_path,
            joint_positions=joint_positions,
            constraint_mask=constraint_mask,
            text_prompt=props.text_prompt,
            fps=20,
        )

        self.report({'INFO'}, f"Exported {n_frames} frames to {export_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

class DMI_OT_RunInference(Operator):
    """Export the current armature animation, run diffusion inference as a
    subprocess, then automatically import the result."""
    bl_idname = "dmi.run_inference"
    bl_label = "Run Inference"
    bl_description = (
        "Export the current animation, run the diffusion model, and import the result. "
        "Configure Python / project / model paths in Addon Preferences."
    )
    bl_options = {'REGISTER'}

    # ---------- modal timer state ----------
    _timer = None
    _thread = None
    _proc = None
    _output_lines = []
    _return_code = None
    _progress_queue = None   # queue.Queue of (pct: int, msg: str) tuples

    def _get_prefs(self, context):
        return context.preferences.addons[__package__].preferences

    def execute(self, context):
        prefs = self._get_prefs(context)

        # Validate preferences
        if not prefs.model_path:
            self.report({'ERROR'}, "Set Model Checkpoint in Addon Preferences first")
            return {'CANCELLED'}

        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        # 1. Run export first
        result = bpy.ops.dmi.export()
        if result != {'FINISHED'}:
            self.report({'ERROR'}, "Export failed – see console for details")
            return {'CANCELLED'}

        props = context.scene.dmi_props
        export_path = bpy.path.abspath(props.export_path)
        import_path = bpy.path.abspath(props.import_path)

        if not os.path.exists(export_path):
            self.report({'ERROR'}, f"Export file not found: {export_path}")
            return {'CANCELLED'}

        # 2. Build subprocess command
        python_exe = bpy.path.abspath(prefs.python_path) if prefs.python_path != "python" else "python"
        cmd = [
            python_exe,
            "-m", "sample.blender_inference",
            "--model_path", bpy.path.abspath(prefs.model_path),
            "--blender_input", export_path,
            "--output", import_path,
            "--text_prompt", props.text_prompt,
            "--guidance_param", str(props.guidance_param),
            "--num_repetitions", str(props.num_repetitions),
            "--seed", str(props.seed),
        ]

        self.report({'INFO'}, f"Starting inference subprocess…")

        # 3. Launch subprocess in a background thread so Blender stays responsive
        DMI_OT_RunInference._output_lines = []
        DMI_OT_RunInference._return_code = None
        DMI_OT_RunInference._progress_queue = queue.Queue()

        def run():
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=prefs.project_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                DMI_OT_RunInference._proc = proc
                for line in proc.stdout:
                    line = line.rstrip()
                    DMI_OT_RunInference._output_lines.append(line)
                    if line.startswith("PROGRESS:"):
                        parts = line.split(":", 2)
                        if len(parts) == 3:
                            try:
                                pct = int(parts[1])
                                msg = parts[2]
                                DMI_OT_RunInference._progress_queue.put((pct, msg))
                            except ValueError:
                                pass
                    else:
                        print("[DMI inference]", line)
                proc.wait()
                DMI_OT_RunInference._return_code = proc.returncode
            except Exception as exc:
                DMI_OT_RunInference._output_lines.append(f"ERROR: {exc}")
                DMI_OT_RunInference._return_code = -1

        DMI_OT_RunInference._thread = threading.Thread(target=run, daemon=True)
        DMI_OT_RunInference._thread.start()

        # 4. Register a modal timer to poll the thread
        wm = context.window_manager
        wm.progress_begin(0, 100)
        DMI_OT_RunInference._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Drain progress queue and update Blender's progress bar
        wm = context.window_manager
        q = DMI_OT_RunInference._progress_queue
        if q is not None:
            last_msg = None
            while not q.empty():
                try:
                    pct, msg = q.get_nowait()
                    wm.progress_update(pct)
                    last_msg = msg
                except queue.Empty:
                    break
            if last_msg is not None:
                self.report({'INFO'}, last_msg)

        # Still running?
        if DMI_OT_RunInference._return_code is None:
            return {'RUNNING_MODAL'}

        # Done – clean up timer and progress bar
        wm.event_timer_remove(DMI_OT_RunInference._timer)
        wm.progress_end()
        DMI_OT_RunInference._timer = None

        if DMI_OT_RunInference._return_code != 0:
            last_lines = "\n".join(DMI_OT_RunInference._output_lines[-5:])
            self.report({'ERROR'}, f"Inference failed (exit {DMI_OT_RunInference._return_code}):\n{last_lines}")
            return {'CANCELLED'}

        # 5. Auto-import result
        result = bpy.ops.dmi.import_result()
        if result == {'FINISHED'}:
            self.report({'INFO'}, "Inference complete and result imported.")
        else:
            self.report({'WARNING'}, "Inference complete but import failed – check Import Path.")

        return {'FINISHED'}

    def cancel(self, context):
        wm = context.window_manager
        if DMI_OT_RunInference._timer:
            wm.event_timer_remove(DMI_OT_RunInference._timer)
            wm.progress_end()
        if DMI_OT_RunInference._proc:
            DMI_OT_RunInference._proc.terminate()


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------

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

        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='POSE')

        rest_bl = rest_blender()

        # Pre-compute rest-pose bone directions (parent→child)
        rest_bone_dirs = {}
        for i, name in enumerate(HML_JOINT_NAMES):
            if JOINT_PARENTS[i] >= 0:
                parent_pos = rest_bl[JOINT_PARENTS[i]]
                child_pos = rest_bl[i]
                rest_bone_dirs[name] = (child_pos - parent_pos).normalized()

        for pb in obj.pose.bones:
            pb.rotation_mode = 'QUATERNION'

        for fi in range(n_frames):
            frame = fi + 1
            context.scene.frame_set(frame)

            positions_bl = [net_to_blender(joint_positions[fi, ji]) for ji in range(22)]
            arm_matrix_inv = obj.matrix_world.inverted()

            # Root (pelvis) location
            pelvis_bone = obj.pose.bones.get('pelvis')
            if pelvis_bone:
                local_pos = arm_matrix_inv @ positions_bl[0]
                rest_head = rest_bl[0]
                pelvis_bone.location = local_pos - rest_head
                pelvis_bone.keyframe_insert(data_path="location", frame=frame)

            # Bone rotations (process parents before children)
            parent_world_rots = {}
            for i, name in enumerate(HML_JOINT_NAMES):
                if JOINT_PARENTS[i] < 0:
                    bone_rest_rot = obj.pose.bones[name].bone.matrix_local.to_quaternion()
                    parent_world_rots[i] = bone_rest_rot
                    continue

                pose_bone = obj.pose.bones.get(name)
                if pose_bone is None:
                    continue

                parent_idx = JOINT_PARENTS[i]
                target_dir = (positions_bl[i] - positions_bl[parent_idx]).normalized()
                rest_dir = rest_bone_dirs.get(name)

                if rest_dir is None or target_dir.length < 0.001:
                    bone_rest_rot = pose_bone.bone.matrix_local.to_quaternion()
                    parent_world_rots[i] = bone_rest_rot
                    continue

                rotation = rest_dir.rotation_difference(target_dir)

                bone_rest_rot = pose_bone.bone.matrix_local.to_quaternion()
                parent_posed_rot = parent_world_rots.get(parent_idx, bone_rest_rot)
                parent_rest_rot = (
                    pose_bone.parent.bone.matrix_local.to_quaternion()
                    if pose_bone.parent else bone_rest_rot
                )

                local_rest_rot = parent_rest_rot.inverted() @ bone_rest_rot
                accumulated = parent_posed_rot @ local_rest_rot
                local_rot = accumulated.inverted() @ rotation @ bone_rest_rot
                pose_bone.rotation_quaternion = local_rot
                parent_world_rots[i] = accumulated @ local_rot

                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        bpy.ops.object.mode_set(mode='OBJECT')

        context.scene.frame_start = 1
        context.scene.frame_end = n_frames

        self.report({'INFO'}, f"Imported {n_frames} frames from {import_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Public list of all operator classes
# ---------------------------------------------------------------------------

classes = (
    DMI_OT_CreateArmature,
    DMI_OT_ToggleConstraint,
    DMI_OT_ConstrainAllBones,
    DMI_OT_ClearAllConstraints,
    DMI_OT_Export,
    DMI_OT_RunInference,
    DMI_OT_Import,
)
