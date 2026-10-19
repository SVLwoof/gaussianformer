import roma
import torch
import torch.nn.functional as F
from torch.amp import autocast


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Composition of rotations given as wxyz quaternions, returned with a non-negative real part."""
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ab = torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), -1)
    return torch.where(ab[..., 0:1] < 0, -ab, ab)


@torch.no_grad()
@autocast("cuda", enabled=False)
def transform_gaussians_to_cam_coord(c2w: torch.Tensor, gaussians: torch.Tensor) -> torch.Tensor:
    """Gaussians [B, N, 14] (pos, scale, quat wxyz, ...) -> the same Gaussians in the camera frame of
    the rigid camera-to-world matrices c2w [B, 4, 4]."""
    w2c = roma.Rigid.from_homogeneous(c2w).inverse()
    pos = w2c[:, None].apply(gaussians[..., :3])
    w2c_quat = roma.rotmat_to_unitquat(w2c.linear)[..., [3, 0, 1, 2]]  # roma is xyzw
    rot = F.normalize(gaussians[..., 6:10], dim=-1)  # padded all-zero rows stay zero instead of NaN
    rot = quaternion_multiply(w2c_quat[:, None], rot)
    return torch.cat([pos, gaussians[..., 3:6], rot, gaussians[..., 10:]], dim=-1)
