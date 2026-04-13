import roma
import torch
from torch.amp import autocast

from gaussianformer.utils.quaternion import quaternion_multiply


@torch.no_grad()
@autocast("cuda", enabled=False)
def transform_gaussians_to_cam_coord(c2w: torch.Tensor, gaussians: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform Gaussians to a camera coordinate system.
    Assumes gaussians tensor layout: [..., pos(3), scale(3), rot(4), ...rest]
    Assumes rigid transform (no scaling in c2w matrix).

    Args:
        c2w: (batch_size, 4, 4) tensor of camera to world matrices.
        gaussians: (batch_size, n_gaussians, dim) tensor of gaussians.

    Returns:
        gaussians_cam: (batch_size, n_gaussians, dim) tensor of gaussians in a camera coordinate system.
        c2w_cam: (batch_size, 4, 4) tensor of camera-to-world matrices in a camera coordinate system, should always be identity.
    """
    device = gaussians.device
    dtype = gaussians.dtype
    bs, n_gaussians, dim = gaussians.shape

    # Extract components
    positions = gaussians[..., :3]
    scales = gaussians[..., 3:6]
    rotations = gaussians[..., 6:10]  # Assuming quaternion (w, x, y, z) or (x, y, z, w) - roma expects (w,x,y,z)
    rest = gaussians[..., 10:]

    # Invert camera-to-world to get world-to-camera transform
    T = roma.Rigid.from_homogeneous(c2w)
    T_inv = T.inverse()

    # Transform positions
    positions_cam = T_inv[:, None].apply(positions)

    # --- Corrected Rotation Transformation ---

    # Get the world-to-camera rotation as a quaternion from the inverse transform's rotation matrix.
    # T_inv.linear gives the 3x3 rotation matrix component.
    # roma returns (x,y,z,w); reorder to (w,x,y,z) for quaternion_multiply
    w2c_rotations = roma.rotmat_to_unitquat(T_inv.linear)[..., [3, 0, 1, 2]]

    # Ensure the per-Gaussian rotations are normalized to be unit quaternions.
    rotations_normalized = rotations / torch.linalg.norm(rotations, dim=-1, keepdim=True)

    # Compose the world-to-camera rotation with the original Gaussian rotation.
    # The key is to use `roma.unitquat_mul` for quaternion multiplication, not the '@' operator.
    # We expand the camera rotation to match the number of gaussians for broadcasting.
    # Shape of w2c_rotations[:, None]: [bs, 1, 4]
    # Shape of rotations_normalized:   [bs, n_gaussians, 4]
    # The multiplication will broadcast correctly.
    rotations_cam = quaternion_multiply(w2c_rotations[:, None], rotations_normalized)

    # Scales are assumed to be invariant under rigid transformation
    scales_cam = scales

    # Reassemble the gaussians tensor
    gaussians_cam = torch.cat([positions_cam, scales_cam, rotations_cam, rest], dim=-1)

    # The new camera-to-world matrix is the identity matrix
    c2w_cam = torch.eye(4, device=device, dtype=dtype).repeat(c2w.shape[0], 1, 1)

    return gaussians_cam, c2w_cam
