# Copyright (c) 2026-2027 zh
"""
内容：
    提供 Isaac Gym 深度相机的可视化辅助函数，绘制 D435 外壳、视场和相机姿态。
"""

import math
import torch
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply, quat_mul


# D435 外壳尺寸，光轴沿局部 +X。
D435_DEPTH_M = 0.025
D435_LENGTH_M = 0.090
D435_HEIGHT_M = 0.025


def _camera_frustum_lines(depth_cfg):
    """生成相机局部坐标系下的视锥线段。"""
    height, width = depth_cfg.resized
    far = float(depth_cfg.far_clip)
    tan_h = math.tan(math.radians(float(depth_cfg.horizontal_fov)) / 2.0)
    tan_v = tan_h * float(height) / float(width)
    corners = np.asarray([[far, -far * tan_h, -far * tan_v], [far, far * tan_h, -far * tan_v], [far, far * tan_h, far * tan_v], [far, -far * tan_h, far * tan_v]], dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    lines = [[origin, corner] for corner in corners]
    lines += [[corners[index], corners[(index + 1) % 4]] for index in range(4)]
    return np.asarray(lines, dtype=np.float32)


def fix_depth_camera_pose(depth_cfg):
    """固定相机姿态。"""
    depth_cfg.position = [float(value) for value in depth_cfg.position]
    depth_cfg.x_angle = [sum(depth_cfg.x_angle) / 2.0] * 2
    depth_cfg.y_angle = [sum(depth_cfg.y_angle) / 2.0] * 2
    depth_cfg.z_angle = [sum(depth_cfg.z_angle) / 2.0] * 2


def draw_d435_camera_boxes(env, color=(0.0, 1.0, 1.0), fov_color=(1.0, 1.0, 0.0)):
    """绘制 D435 外壳和 FOV 视锥。"""
    # 检查相机和 viewer。
    if env.viewer is None or not env.cfg.depth.use_camera:
        return

    # 有效相机数量。
    camera_count = min(len(env.depth_index), len(env.camera_local_positions), len(env.camera_local_quats))
    if camera_count == 0:
        return

    # 创建相机线框。
    env.gym.clear_lines(env.viewer)
    box_geometry = gymutil.WireframeBoxGeometry(D435_DEPTH_M, D435_LENGTH_M, D435_HEIGHT_M, None, color=color)
    frustum_lines = _camera_frustum_lines(env.cfg.depth)
    frustum_vertices = np.empty((len(frustum_lines), 2), dtype=gymapi.Vec3.dtype)
    frustum_vertices["x"], frustum_vertices["y"], frustum_vertices["z"] = frustum_lines[..., 0], frustum_lines[..., 1], frustum_lines[..., 2]
    frustum_colors = np.empty(len(frustum_lines), dtype=gymapi.Vec3.dtype)
    frustum_colors.fill(fov_color)

    # 相机外参 Tensor。
    camera_env_ids = torch.as_tensor(env.depth_index[:camera_count], device=env.device, dtype=torch.long)
    local_positions = torch.as_tensor(env.camera_local_positions[:camera_count], device=env.device, dtype=env.root_states.dtype)
    local_quats = torch.as_tensor(env.camera_local_quats[:camera_count], device=env.device, dtype=env.root_states.dtype)

    # 转换到世界坐标系。
    root_positions = env.root_states[camera_env_ids, :3]
    root_quats = env.root_states[camera_env_ids, 3:7]
    world_positions = root_positions + quat_apply(root_quats, local_positions)
    world_quats = quat_mul(root_quats, local_quats)

    # 转为 CPU 数组。
    world_positions = world_positions.detach().cpu().numpy()
    world_quats = world_quats.detach().cpu().numpy()
    camera_env_ids = camera_env_ids.detach().cpu().tolist()

    # 绘制各相机线框。
    for camera_slot, env_id in enumerate(camera_env_ids):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*world_positions[camera_slot])
        pose.r = gymapi.Quat(*world_quats[camera_slot])
        gymutil.draw_lines(box_geometry, env.gym, env.viewer, env.envs[env_id], pose)

        # 将局部视锥转换到世界坐标系。
        env.gym.add_lines(env.viewer, env.envs[env_id], len(frustum_lines), pose.transform_points(frustum_vertices), frustum_colors)
