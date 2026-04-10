"""
Inference bridge between Blender addon and diffusion motion inbetweening model.

Loads an NPZ file exported from the Blender addon (joint positions + constraint mask),
runs diffusion sampling conditioned on the constraints, and saves output joint positions
for re-import into Blender.

Usage:
    python -m sample.blender_inference \
        --model_path save/my_model/model000500000.pt \
        --blender_input dmi_export.npz \
        --output dmi_result.npz \
        --text_prompt "a person walks forward" \
        --guidance_param 2.5
"""

import os
import argparse
import json

import numpy as np
import torch

from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, DatasetConfig
from data_loaders.humanml.scripts.motion_process import (
    extract_features,
    recover_from_ric,
    recover_root_rot_pos,
    uniform_skeleton,
)
from data_loaders.humanml.utils.paramUtil import (
    t2m_raw_offsets,
    t2m_kinematic_chain,
)
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import qbetween_np, qrot_np, qinv_np
from data_loaders.tensors import collate
from utils.editing_util import joint_to_full_mask


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Blender-to-diffusion inference bridge")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--blender_input", type=str, required=True,
                        help="Path to Blender-exported NPZ file")
    parser.add_argument("--output", type=str, default="dmi_result.npz",
                        help="Path to save output NPZ for Blender import")
    parser.add_argument("--guidance_param", type=float, default=2.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--num_repetitions", type=int, default=1,
                        help="Number of motion samples to generate")
    parser.add_argument("--text_prompt", type=str, default=None,
                        help="Override text prompt from the NPZ file")
    return parser.parse_args()


def load_model_args(model_path):
    """Load training args from args.json next to the checkpoint."""
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'args.json not found at {args_path}'
    with open(args_path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def interpolate_positions(joint_positions, constraint_mask):
    """
    Linearly interpolate joint positions between constrained keyframes.
    Needed because extract_features() requires positions at every frame
    for velocity and foot contact computation.

    joint_positions: [n_frames, 22, 3]
    constraint_mask: [n_frames, 22] bool

    Returns: [n_frames, 22, 3] with interpolated positions
    """
    n_frames, n_joints, _ = joint_positions.shape
    result = joint_positions.copy()

    for j in range(n_joints):
        # Find constrained frames for this joint
        constrained_frames = np.where(constraint_mask[:, j])[0]

        if len(constrained_frames) == 0:
            # No constraints on this joint: leave as zeros (model will generate freely)
            continue

        if len(constrained_frames) == 1:
            # Single constraint: fill all frames with that position
            result[:, j] = joint_positions[constrained_frames[0], j]
            continue

        # Interpolate between consecutive constrained frames
        for ci in range(len(constrained_frames) - 1):
            f_start = constrained_frames[ci]
            f_end = constrained_frames[ci + 1]
            pos_start = joint_positions[f_start, j]
            pos_end = joint_positions[f_end, j]

            for f in range(f_start, f_end + 1):
                alpha = (f - f_start) / max(f_end - f_start, 1)
                result[f, j] = pos_start * (1 - alpha) + pos_end * alpha

        # Extrapolate before first constrained frame
        result[:constrained_frames[0], j] = joint_positions[constrained_frames[0], j]
        # Extrapolate after last constrained frame
        result[constrained_frames[-1]:, j] = joint_positions[constrained_frames[-1], j]

    return result


def positions_to_features(joint_positions, mean_abs, std_abs):
    """
    Convert [n_frames, 22, 3] joint positions to normalized 263-feature representation.
    Follows the full pipeline in motion_process.py:process_file() so that the
    features match the training data distribution.

    Returns:
        sample_abs: torch tensor [1, 263, 1, n_frames]
        root_quat_init: [4] numpy quaternion used to pre-rotate to face +Z
        root_pos_init_xz: [3] numpy offset subtracted to center root at XZ origin
    """
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    fid_r, fid_l = [8, 11], [7, 10]

    # Normalize skeleton proportions to reference skeleton
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    example_data = np.load(os.path.join('dataset', '000021.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_data[0]))

    positions = torch.from_numpy(joint_positions).float()
    positions = uniform_skeleton(positions.numpy(), tgt_offsets, n_raw_offsets, kinematic_chain)

    # Put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # --- Replicate process_file() preprocessing ---
    # Center root XZ at origin
    root_pos_init = positions[0]
    root_pos_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pos_init_xz

    # Rotate all positions so the character initially faces +Z.
    # Training data is always pre-rotated this way; without it the absolute
    # position features end up in the wrong coordinate frame and the model
    # generates motion along the wrong axis.
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    # Broadcast to all frames and joints
    root_quat_init_exp = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init_exp, positions)

    # extract_features returns [n_frames-1, 263] (relative representation)
    sample_rel = extract_features(
        positions.copy(),
        0.002,
        n_raw_offsets, kinematic_chain,
        face_joint_indx, fid_r, fid_l,
    )

    # Duplicate last frame to restore frame count
    sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
    sample_rel = torch.cat([sample_rel, sample_rel[:, -1:, :].clone()], dim=1)
    # sample_rel shape: [1, n_frames, 263]

    # Convert root from relative to absolute (following motion_to_abs_data)
    r_rot_quat, r_pos, rot_ang = recover_root_rot_pos(
        sample_rel[None], abs_3d=False, return_rot_ang=True
    )
    sample_abs = sample_rel[None].clone()
    sample_abs[..., 0] = rot_ang           # absolute rotation angle
    sample_abs[..., [1, 2]] = r_pos[..., [0, 2]]  # absolute XZ position

    # Z-normalize with absolute stats
    sample_abs = (sample_abs - mean_abs) / std_abs

    # Reshape to [1, 263, 1, n_frames]
    sample_abs = sample_abs.squeeze(0)  # [1, n_frames, 263]
    sample_abs = sample_abs.permute(0, 2, 1).unsqueeze(2)  # [1, 263, 1, n_frames]

    return sample_abs, root_quat_init.flatten(), root_pos_init_xz


def create_obs_mask(constraint_mask, max_frames, feature_mode='pos_rot_vel'):
    """
    Convert [n_frames, 22] bool constraint mask to [1, 263, 1, max_frames] feature mask.
    """
    n_frames = constraint_mask.shape[0]

    # Ensure the root joint is constrained at every frame where any joint is
    # constrained. Non-root joint positions are stored as root-invariant (RIC)
    # features. If the root rotation/position is not also constrained the
    # diffusion model is free to change it, causing the RIC positions to be
    # interpreted with the wrong root transform and placing joints on the
    # wrong axis.
    constraint_mask = constraint_mask.copy()
    any_constrained = constraint_mask.any(axis=1)  # [n_frames]
    constraint_mask[any_constrained, 0] = True

    # Build joint mask [1, 22, 1, max_frames]
    joint_mask = torch.zeros(1, 22, 1, max_frames, dtype=torch.bool)
    effective_frames = min(n_frames, max_frames)
    joint_mask[0, :, 0, :effective_frames] = torch.from_numpy(
        constraint_mask[:effective_frames].T
    )

    # Convert to feature mask [1, 263, 1, max_frames]
    feature_mask = joint_to_full_mask(joint_mask, mode=feature_mode)
    return feature_mask


def features_to_positions(sample, inv_transform_fn, abs_3d=True,
                          root_quat_init=None, root_pos_init_xz=None):
    """
    Convert model output [bs, 263, 1, n_frames] to joint positions [bs, n_frames, 22, 3].

    If root_quat_init / root_pos_init_xz are provided the inverse of the
    pre-rotation applied by positions_to_features is undone so the output
    positions are back in the original world frame.
    """
    # [bs, 263, 1, n_frames] -> [bs, 1, n_frames, 263]
    sample = sample.cpu().permute(0, 2, 3, 1)

    # Denormalize
    sample = inv_transform_fn(sample).float()

    # Recover joint positions
    positions = recover_from_ric(sample, joints_num=22, abs_3d=abs_3d)
    # positions shape: [bs, 1, n_frames, 22, 3]

    positions = positions.squeeze(1).numpy()  # [bs, n_frames, 22, 3]

    # Undo the pre-rotation that was applied to match training data
    if root_quat_init is not None:
        inv_quat = qinv_np(root_quat_init[np.newaxis])  # [1, 4]
        for i in range(positions.shape[0]):
            inv_quat_exp = np.ones(positions[i].shape[:-1] + (4,)) * inv_quat
            positions[i] = qrot_np(inv_quat_exp, positions[i])

    if root_pos_init_xz is not None:
        positions = positions + root_pos_init_xz

    return positions


# ---------------------------------------------------------------------------
# Dataset loading (minimal, for normalization stats)
# ---------------------------------------------------------------------------

def load_dataset(model_args, max_frames):
    """Load dataset to get normalization statistics."""
    conf = DatasetConfig(
        name=model_args.get('dataset', 'humanml'),
        batch_size=1,
        num_frames=max_frames,
        split='test',
        hml_mode='train',
        use_abs3d=model_args.get('abs_3d', True),
        traject_only=model_args.get('traj_only', False),
        use_random_projection=model_args.get('use_random_proj', False),
        random_projection_scale=model_args.get('random_proj_scale', None),
        augment_type='none',
        std_scale_shift=tuple(model_args.get('std_scale_shift', (1.0, 0.0))),
        drop_redundant=model_args.get('drop_redundant', False),
    )
    data = get_dataset_loader(conf, shuffle=False, num_workers=0, drop_last=False)
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    fixseed(args.seed)

    print(f"Loading Blender export from: {args.blender_input}")
    blender_data = np.load(args.blender_input, allow_pickle=True)
    joint_positions = blender_data['joint_positions']     # [n_frames, 22, 3]
    constraint_mask = blender_data['constraint_mask']     # [n_frames, 22]
    text_prompt = str(blender_data['text_prompt']) if 'text_prompt' in blender_data else ''
    if args.text_prompt is not None:
        text_prompt = args.text_prompt

    n_input_frames = joint_positions.shape[0]
    print(f"  Frames: {n_input_frames}, Constrained markers: {constraint_mask.sum()}")
    print(f"  Text: '{text_prompt}'")

    # Load model training args
    model_args = load_model_args(args.model_path)
    assert model_args.get('dataset') == 'humanml', "Only humanml dataset is supported"
    assert model_args.get('abs_3d', False), "Only abs_3d models are supported"
    assert model_args.get('keyframe_conditioned', False), "Model must be keyframe-conditioned"

    max_frames = 196
    n_frames = min(n_input_frames, max_frames)

    # Setup device
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load dataset for normalization stats
    print("Loading dataset for normalization...")
    data = load_dataset(model_args, max_frames)
    t2m_dataset = data.dataset.t2m_dataset

    # Use cached HML3D positions if available (avoids Blender round-trip errors)
    if 'hml3d_joint_positions' in blender_data:
        print("Using cached HML3D joint positions (debug mode)")
        hml3d_positions = blender_data['hml3d_joint_positions'][:n_frames]
        # Override cached positions at constrained frames/joints with the
        # Blender-exported positions.  The user may have moved keyframes to
        # new locations that the cache doesn't reflect.
        mask = constraint_mask[:n_frames]
        hml3d_positions[mask] = joint_positions[:n_frames][mask]
    else:
        hml3d_positions = joint_positions[:n_frames]

    # Interpolate positions between constrained keyframes
    print("Interpolating positions between keyframes...")
    interp_positions = interpolate_positions(
        hml3d_positions, constraint_mask[:n_frames]
    )

    # Check if any constraints exist
    has_constraints = constraint_mask[:n_frames].any()
    if not has_constraints:
        print("WARNING: No constraints found. Running unconditional generation.")

    # Convert positions to 263 features
    print("Converting positions to feature representation...")
    mean_abs = torch.from_numpy(t2m_dataset.mean).float()
    std_abs = torch.from_numpy(t2m_dataset.std).float()

    obs_x0, root_quat_init, root_pos_init_xz = positions_to_features(
        interp_positions, mean_abs, std_abs
    )
    # Pad to max_frames if needed
    if obs_x0.shape[-1] < max_frames:
        padding = torch.zeros(1, 263, 1, max_frames - obs_x0.shape[-1])
        obs_x0 = torch.cat([obs_x0, padding], dim=-1)

    # Create observation mask
    obs_mask = create_obs_mask(
        constraint_mask[:n_frames], max_frames,
        feature_mode=model_args.get('feature_mode', 'pos_rot_vel'),
    )

    # Build model_kwargs using collate
    collate_args = [{
        'inp': torch.zeros(max_frames),
        'tokens': None,
        'lengths': n_frames,
        'text': text_prompt,
    }]
    _, model_kwargs = collate(collate_args)

    # Move to device
    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(device)

    # Set conditioning
    model_kwargs['obs_x0'] = obs_x0.to(device)
    model_kwargs['obs_mask'] = obs_mask.to(device)

    # Setup imputation (following conditional_synthesis.py:171-177)
    zero_kf_loss = model_args.get('zero_keyframe_loss', False)
    if zero_kf_loss:
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = 0
        model_kwargs['y']['replacement_distribution'] = 'conditional'
        model_kwargs['y']['inpainted_motion'] = obs_x0.to(device)
        model_kwargs['y']['inpainting_mask'] = obs_mask.to(device)
        model_kwargs['y']['reconstruction_guidance'] = False
    else:
        # Still use imputation for best results
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = 0
        model_kwargs['y']['replacement_distribution'] = 'conditional'
        model_kwargs['y']['inpainted_motion'] = obs_x0.to(device)
        model_kwargs['y']['inpainting_mask'] = obs_mask.to(device)

    model_kwargs['y']['diffusion_steps'] = model_args.get('diffusion_steps', 1000)

    # Create model and diffusion
    print("Creating model and diffusion...")

    # Build a namespace-like object from model_args for create_model_and_diffusion
    class Args:
        pass

    margs = Args()
    for k, v in model_args.items():
        setattr(margs, k, v)
    # Override with CLI args where needed
    margs.guidance_param = args.guidance_param
    margs.batch_size = 1

    model, diffusion = create_model_and_diffusion(margs, data)

    print(f"Loading checkpoint from: {args.model_path}")
    load_saved_model(model, args.model_path)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)

    model.to(device)
    model.eval()

    # Run sampling
    all_positions = []

    for rep_i in range(args.num_repetitions):
        print(f"Sampling repetition {rep_i + 1}/{args.num_repetitions}...")

        if args.guidance_param != 1:
            model_kwargs['y']['text_scale'] = (
                torch.ones(1, device=device) * args.guidance_param
            )

        sample = diffusion.p_sample_loop(
            model,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )  # [1, 263, 1, max_frames]

        # Convert to joint positions
        positions = features_to_positions(
            sample,
            inv_transform_fn=t2m_dataset.inv_transform,
            abs_3d=model_args.get('abs_3d', True),
            root_quat_init=root_quat_init,
            root_pos_init_xz=root_pos_init_xz,
        )  # [1, n_frames, 22, 3] -- but full max_frames

        # Trim to actual motion length
        positions = positions[:, :n_frames, :, :]
        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)  # [num_reps, n_frames, 22, 3]

    # Save output
    print(f"Saving {all_positions.shape[0]} sample(s) to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Save the first repetition as the primary result (for Blender import)
    np.savez(
        args.output,
        joint_positions=all_positions[0],  # [n_frames, 22, 3]
        fps=20,
    )

    # If multiple repetitions, save all
    if args.num_repetitions > 1:
        all_path = args.output.replace('.npz', '_all.npz')
        np.savez(
            all_path,
            joint_positions=all_positions,  # [num_reps, n_frames, 22, 3]
            fps=20,
        )
        print(f"All repetitions saved to: {all_path}")

    print("Done!")


if __name__ == "__main__":
    main()
