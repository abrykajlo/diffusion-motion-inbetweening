"""
Blender operators for the DMI addon:
  - Create SMPL armature
  - Toggle / constrain / clear constraints
  - Export joint positions to NPZ
  - Run inference (launches blender_inference.py as a subprocess)
  - Import NPZ result as animation
"""

import os
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

from bpy_extras.io_utils import ExportHelper, ImportHelper


# ---------------------------------------------------------------------------
# Keyframe snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_keyframes(obj):
    """Return a JSON-serialisable snapshot of all DMI bone keyframes on obj."""
    action = obj.animation_data.action if (obj and obj.animation_data) else None
    data = {}
    if not action:
        return data
    for name in HML_JOINT_NAMES:
        bone_data = {}
        for fcurve in action.fcurves:
            if f'pose.bones["{name}"]' not in fcurve.data_path:
                continue
            dp = fcurve.data_path.rsplit('.', 1)[-1]
            if dp not in bone_data:
                bone_data[dp] = []
            for kp in fcurve.keyframe_points:
                bone_data[dp].append([int(round(kp.co.x)), fcurve.array_index, kp.co.y])
        if bone_data:
            data[name] = bone_data
    return data


def _apply_keyframes(obj, data):
    """Replace all DMI bone keyframes on obj with the given snapshot."""
    import json as _json
    if not obj.animation_data:
        obj.animation_data_create()
    action = obj.animation_data.action
    if not action:
        action = bpy.data.actions.new("DMI_Action")
        obj.animation_data.action = action

    # Remove existing DMI fcurves
    for name in HML_JOINT_NAMES:
        for fc in [fc for fc in action.fcurves if f'pose.bones["{name}"]' in fc.data_path]:
            action.fcurves.remove(fc)

    # Re-create from snapshot
    for name, bone_data in data.items():
        for dp, triples in bone_data.items():
            full_dp = f'pose.bones["{name}"].{dp}'
            by_channel = {}
            for frame, ch, val in triples:
                by_channel.setdefault(ch, []).append((frame, val))
            for ch, points in sorted(by_channel.items()):
                fc = action.fcurves.new(data_path=full_dp, index=ch)
                fc.keyframe_points.add(len(points))
                for i, (frame, val) in enumerate(sorted(points)):
                    fc.keyframe_points[i].co = (frame, val)
                    fc.keyframe_points[i].interpolation = 'LINEAR'
                fc.update()


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

        # Pelvis location is set in armature space, not bone-local space
        arm_data.bones['pelvis'].use_local_location = False

        context.scene.render.fps = 20
        context.scene.render.fps_base = 1.0

        constraints = Constraints(context.scene)
        constraints.save()

        # Clear any cached HML3D data from a previous import
        if HML3D_CACHE_KEY in context.scene:
            del context.scene[HML3D_CACHE_KEY]

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
        obj = context.active_object
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
        obj = context.active_object
        constraints = Constraints(context.scene)

        action = obj.animation_data.action if (obj and obj.type == 'ARMATURE' and obj.animation_data) else None
        removed_kf = 0

        if action:
            for frame_str, bones in constraints.data.items():
                frame = int(frame_str)
                for name in bones:
                    for fcurve in action.fcurves:
                        if f'pose.bones["{name}"]' not in fcurve.data_path:
                            continue
                        for kp in list(fcurve.keyframe_points):
                            if int(round(kp.co.x)) == frame:
                                fcurve.keyframe_points.remove(kp)
                                removed_kf += 1

        constraints.clear()
        constraints.save()
        self.report({'INFO'}, f"All constraints cleared (removed {removed_kf} keyframe(s))")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_npz(filepath, context):
    """Export joint positions and constraint mask to an NPZ file at filepath."""
    props = context.scene.dmi_props
    obj = context.active_object

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

    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else '.',
        exist_ok=True,
    )
    save_kwargs = dict(
        joint_positions=joint_positions,
        constraint_mask=constraint_mask,
        text_prompt=props.text_prompt,
        fps=20,
    )

    # Include cached HML3D network-coord positions if available
    cache_path = context.scene.get(HML3D_CACHE_KEY, "")
    if cache_path and os.path.exists(cache_path):
        hml3d_positions = np.load(cache_path, allow_pickle=True)
        save_kwargs['hml3d_joint_positions'] = hml3d_positions

    np.savez(filepath, **save_kwargs)


class DMI_OT_Export(Operator, ExportHelper):
    bl_idname = "dmi.export"
    bl_label = "Export for Inference"
    bl_description = "Export joint positions and constraint mask to NPZ"
    bl_options = {'REGISTER'}

    filename_ext = ".npz"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        filepath = bpy.path.abspath(self.filepath)
        export_npz(filepath, context)
        
        props = context.scene.dmi_props
        self.report({'INFO'}, f"Exported {props.frame_count} frames to {filepath}")
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

        props = context.scene.dmi_props
        inferences_dir = os.path.join(bpy.path.abspath(prefs.project_path), "blender_inferences")
        os.makedirs(inferences_dir, exist_ok=True)
        name = props.inference_name or "inference"

        # Create a unique sub-folder for this inference run
        run_dir = os.path.join(inferences_dir, name)
        if os.path.exists(run_dir):
            n = 1
            while os.path.exists(os.path.join(inferences_dir, f"{name}_{n}")):
                n += 1
            run_dir = os.path.join(inferences_dir, f"{name}_{n}")
        os.makedirs(run_dir)

        export_path = os.path.join(run_dir, "export.npz")
        import_path = os.path.join(run_dir, "result.npz")

        # 1. Run export first
        try:
            export_npz(export_path, context)
        except Exception as exc:
            self.report({'ERROR'}, f"Export failed: {exc}")
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
        DMI_OT_RunInference._import_path = import_path
        DMI_OT_RunInference._output_lines = []
        DMI_OT_RunInference._return_code = None

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
        DMI_OT_RunInference._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Still running?
        if DMI_OT_RunInference._return_code is None:
            return {'RUNNING_MODAL'}

        # Done – clean up timer
        wm = context.window_manager
        wm.event_timer_remove(DMI_OT_RunInference._timer)
        DMI_OT_RunInference._timer = None

        if DMI_OT_RunInference._return_code != 0:
            last_lines = "\n".join(DMI_OT_RunInference._output_lines[-5:])
            self.report({'ERROR'}, f"Inference failed (exit {DMI_OT_RunInference._return_code}):\n{last_lines}")
            return {'CANCELLED'}

        # 5. Auto-import result
        import_path = DMI_OT_RunInference._import_path
        try:
            import_npz(import_path, context)
            self.report({'INFO'}, "Inference complete and result imported.")
        except Exception as exc:
            self.report({'WARNING'}, f"Inference complete but import failed: {exc}")

        return {'FINISHED'}

    def cancel(self, context):
        wm = context.window_manager
        if DMI_OT_RunInference._timer:
            wm.event_timer_remove(DMI_OT_RunInference._timer)
        if DMI_OT_RunInference._proc:
            DMI_OT_RunInference._proc.terminate()


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------

def _apply_joint_positions(joint_positions, context):
    """Apply a (n_frames, 22, 3) array of network-coord joint positions as animation.

    Snapshots current keyframes as the constrained layer, writes the new animation,
    then snapshots the result as the inferred layer. Returns n_frames.
    """
    import json

    props = context.scene.dmi_props
    obj = context.active_object
    n_frames = joint_positions.shape[0]

    props.keyframes_constrained = json.dumps(_snapshot_keyframes(obj))

    context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE')

    rest_bl = rest_blender()

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

        pelvis_bone = obj.pose.bones.get('pelvis')
        if pelvis_bone:
            local_pos = arm_matrix_inv @ positions_bl[0]
            rest_head = rest_bl[0]
            pelvis_bone.location = local_pos - rest_head
            pelvis_bone.keyframe_insert(data_path="location", frame=frame)

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

    props.keyframes_inferred = json.dumps(_snapshot_keyframes(obj))
    props.active_keyframe_layer = 'INFERRED'

    return n_frames


def import_npz(filepath, context):
    """Load joint positions from a DMI NPZ file and apply as animation."""
    data = np.load(filepath, allow_pickle=True)
    joint_positions = data['joint_positions']  # [n_frames, 22, 3] network coords
    n_frames = _apply_joint_positions(joint_positions, context)

    if 'constraint_mask' in data:
        constraint_mask = data['constraint_mask']  # [n_frames, 22] bool
        constraints = Constraints(context.scene)
        constraints.clear()
        for fi in range(constraint_mask.shape[0]):
            frame = fi + 1
            for ji in range(constraint_mask.shape[1]):
                if constraint_mask[fi, ji]:
                    constraints.set(frame, HML_JOINT_NAMES[ji])
        constraints.save()

    return n_frames


# ---------------------------------------------------------------------------
# HumanML3D numpy helpers
# ---------------------------------------------------------------------------

def _qrot_np(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) [w, x, y, z]."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def _qinv_np(q):
    """Invert unit quaternions by negating the xyz components."""
    inv = q.copy()
    inv[..., 1:] *= -1.0
    return inv


def _recover_from_ric_np(data, joints_num=22):
    """Numpy reimplementation of recover_from_ric.

    data: (seq_len, 263) HumanML3D feature vectors
    Returns: (seq_len, joints_num, 3) joint positions in network coords (Y-up, Z-forward)
    """
    # --- root rotation (Y-axis) ---
    rot_vel = data[:, 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)

    r_rot_quat = np.zeros((len(data), 4), dtype=np.float32)
    r_rot_quat[:, 0] = np.cos(r_rot_ang)  # w
    r_rot_quat[:, 2] = np.sin(r_rot_ang)  # y (rotation around Y axis)

    # --- root position ---
    r_pos = np.zeros((len(data), 3), dtype=np.float32)
    r_pos[1:, [0, 2]] = data[:-1, 1:3]
    r_pos = _qrot_np(_qinv_np(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = data[:, 3]

    # --- RIC joint positions (non-root) ---
    positions = data[:, 4:(joints_num - 1) * 3 + 4].reshape(len(data), joints_num - 1, 3)

    # Apply root Y-axis rotation to local joint positions
    r_rot_exp = np.tile(r_rot_quat[:, np.newaxis, :], (1, joints_num - 1, 1))
    positions = _qrot_np(_qinv_np(r_rot_exp), positions)

    # Add root XZ offset
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]

    # Prepend root joint
    return np.concatenate([r_pos[:, np.newaxis, :], positions], axis=1).astype(np.float32)


HML3D_CACHE_KEY = "dmi_hml3d_cache_path"


def import_humanml3d_npy(filepath, context):
    """Load a HumanML3D .npy file and apply as animation.

    Accepts either:
      - (seq_len, 22, 3)  raw joint positions in network coords
      - (seq_len, 263)    HumanML3D feature vectors (recover_from_ric is applied)

    Caches the original network-coord joint positions so they can be
    re-exported for inference without Blender round-trip errors.
    """
    raw = np.load(filepath, allow_pickle=True)

    if raw.ndim == 3 and raw.shape[1] == 22 and raw.shape[2] == 3:
        joint_positions = raw.astype(np.float32)
    elif raw.ndim == 2 and raw.shape[1] == 263:
        joint_positions = _recover_from_ric_np(raw)
    else:
        raise ValueError(
            f"Unrecognised HumanML3D array shape {raw.shape}. "
            "Expected (seq_len, 22, 3) or (seq_len, 263)."
        )

    # Cache the network-coord positions for later export
    cache_dir = os.path.join(os.path.dirname(filepath), ".dmi_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, os.path.basename(filepath))
    np.save(cache_path, joint_positions)
    context.scene[HML3D_CACHE_KEY] = cache_path

    return _apply_joint_positions(joint_positions, context)



def import_inference_run(run_dir, context):
    """Load an inference run folder containing export.npz and result.npz.

    Imports the export (constrained keyframes + constraints) and the result
    (inferred keyframes), restoring both keyframe layers.
    """
    export_path = os.path.join(run_dir, "export.npz")
    result_path = os.path.join(run_dir, "result.npz")

    if not os.path.exists(export_path):
        raise FileNotFoundError(f"export.npz not found in {run_dir}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"result.npz not found in {run_dir}")

    props = context.scene.dmi_props
    obj = context.active_object

    # --- Load the export (constrained pose + constraints) ---
    export_data = np.load(export_path, allow_pickle=True)
    export_positions = export_data['joint_positions']
    _apply_joint_positions(export_positions, context)

    if 'constraint_mask' in export_data:
        constraint_mask = export_data['constraint_mask']
        constraints = Constraints(context.scene)
        constraints.clear()
        for fi in range(constraint_mask.shape[0]):
            frame = fi + 1
            for ji in range(constraint_mask.shape[1]):
                if constraint_mask[fi, ji]:
                    constraints.set(frame, HML_JOINT_NAMES[ji])
        constraints.save()

    if 'text_prompt' in export_data:
        props.text_prompt = str(export_data['text_prompt'])

    # Cache the network-coord positions for re-export
    if 'hml3d_joint_positions' in export_data:
        cache_dir = os.path.join(run_dir, ".dmi_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "hml3d_positions.npy")
        np.save(cache_path, export_data['hml3d_joint_positions'])
        context.scene[HML3D_CACHE_KEY] = cache_path

    # --- Load the result (inferred animation) ---
    result_data = np.load(result_path, allow_pickle=True)
    result_positions = result_data['joint_positions']
    n_frames = _apply_joint_positions(result_positions, context)

    return n_frames


class DMI_OT_Import(Operator, ImportHelper):
    bl_idname = "dmi.import_result"
    bl_label = "Import Result"
    bl_description = "Load an NPZ result, a HumanML3D .npy file, or an inference run folder"
    bl_options = {'REGISTER', 'UNDO'}

    # Default; overridden in invoke based on import_source
    filename_ext = ".npz"

    def invoke(self, context, event):
        props = context.scene.dmi_props
        if props.import_source == 'HML3D':
            self.filename_ext = ".npy"
        else:
            self.filename_ext = ".npz"

        prefs = context.preferences.addons[__package__].preferences
        if props.import_source == 'INFERENCE':
            inferences_dir = os.path.join(
                bpy.path.abspath(prefs.project_path), "blender_inferences"
            )
            if os.path.isdir(inferences_dir):
                self.filepath = inferences_dir + os.sep
        elif props.import_source == 'HML3D':
            hml3d_dir = os.path.join(
                bpy.path.abspath(prefs.project_path), "dataset", "HumanML3D"
            )
            if os.path.isdir(hml3d_dir):
                self.filepath = hml3d_dir + os.sep
        else:
            inferences_dir = os.path.join(
                bpy.path.abspath(prefs.project_path), "blender_inferences"
            )
            if os.path.isdir(inferences_dir):
                self.filepath = inferences_dir + os.sep

        return super().invoke(context, event)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        filepath = bpy.path.abspath(self.filepath)
        props = context.scene.dmi_props

        try:
            if props.import_source == 'INFERENCE':
                # User may select either file in the folder or the folder itself;
                # resolve to the containing directory.
                if os.path.isfile(filepath):
                    run_dir = os.path.dirname(filepath)
                else:
                    run_dir = filepath
                n_frames = import_inference_run(run_dir, context)
                self.report({'INFO'}, f"Imported inference run ({n_frames} frames) from {run_dir}")
            elif props.import_source == 'HML3D':
                if not os.path.exists(filepath):
                    self.report({'ERROR'}, f"File not found: {filepath}")
                    return {'CANCELLED'}
                n_frames = import_humanml3d_npy(filepath, context)
                self.report({'INFO'}, f"Imported {n_frames} frames from {filepath}")
            else:
                if not os.path.exists(filepath):
                    self.report({'ERROR'}, f"File not found: {filepath}")
                    return {'CANCELLED'}
                n_frames = import_npz(filepath, context)
                self.report({'INFO'}, f"Imported {n_frames} frames from {filepath}")
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Keyframe layer switcher
# ---------------------------------------------------------------------------

class DMI_OT_ApplyKeyframeLayer(Operator):
    bl_idname = "dmi.apply_keyframe_layer"
    bl_label = "Switch Keyframe Layer"
    bl_description = "Replace the armature's keyframes with the stored constrained or inferred set"
    bl_options = {'REGISTER', 'UNDO'}

    layer: bpy.props.EnumProperty(
        items=[
            ('CONSTRAINED', "Constrained", "Restore the keyframes captured when constraints were set"),
            ('INFERRED',    "Inferred",    "Restore the keyframes produced by the last inference run"),
        ],
        name="Layer",
    )

    def execute(self, context):
        import json

        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        props = context.scene.dmi_props
        raw = props.keyframes_constrained if self.layer == 'CONSTRAINED' else props.keyframes_inferred

        if not raw or raw == '{}':
            self.report({'WARNING'}, f"No stored {self.layer.lower()} keyframes found")
            return {'CANCELLED'}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Stored keyframe data is corrupted")
            return {'CANCELLED'}

        _apply_keyframes(obj, data)
        props.active_keyframe_layer = self.layer
        self.report({'INFO'}, f"Switched to {self.layer.lower()} keyframes")
        return {'FINISHED'}


class DMI_OT_SnapshotConstraintKeyframes(Operator):
    bl_idname = "dmi.snapshot_constraint_keyframes"
    bl_label = "Snapshot as Constrained"
    bl_description = "Save the armature's current keyframes as the constrained layer"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import json

        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the DMI armature first")
            return {'CANCELLED'}

        context.scene.dmi_props.keyframes_constrained = json.dumps(_snapshot_keyframes(obj))
        context.scene.dmi_props.active_keyframe_layer = 'CONSTRAINED'
        self.report({'INFO'}, "Constrained keyframes snapshot saved")
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
    DMI_OT_ApplyKeyframeLayer,
    DMI_OT_SnapshotConstraintKeyframes,
)
