"""
Skeleton constants and coordinate conversion utilities for the
22-joint SMPL skeleton used by HumanML3D.
"""

from mathutils import Vector

# ---------------------------------------------------------------------------
# Joint names and hierarchy
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


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def net_to_blender(pos):
    """Convert network coords (X-right, Y-up, Z-forward) to Blender (X-right, Y-forward, Z-up)."""
    return Vector((pos[0], pos[2], pos[1]))


def blender_to_net(pos):
    """Convert Blender coords (X-right, Y-forward, Z-up) to network (X-right, Y-up, Z-forward)."""
    return [pos[0], pos[2], pos[1]]


def rest_blender():
    """Return rest-pose joint positions in Blender coordinates, placed on the floor (min Z = 0)."""
    rest_bl = [net_to_blender(p) for p in REST_POSITIONS_NETWORK]
    min_z = min(v.z for v in rest_bl)
    rest_offset = Vector((0, 0, -min_z))
    return [v + rest_offset for v in rest_bl]
