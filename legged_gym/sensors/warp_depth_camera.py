# Copyright (c) 2026-2027 zh
"""
内容：
    基于 NVIDIA Warp 和地形网格实现深度相机仿真，生成深度图并维护相机位姿。
"""

import numpy as np
import torch
import warp as wp
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul
from legged_gym.sensors.warp.warp_cam import WarpCam


class WMPDepthCameraConfig:
    sensor_type = "camera"
    num_sensors = 1 # 相机数量
    calculate_depth = True # 输出轴向深度，而非射线距离
    return_pointcloud = False # 选择深度图或点云 kernel
    pointcloud_in_world_frame = False
    segmentation_camera = False # 选择普通或语义分割 kernel

    def __init__(self, depth_cfg):
        self.height = depth_cfg.resized[0]
        self.width = depth_cfg.resized[1]
        self.horizontal_fov_deg = depth_cfg.horizontal_fov
        self.max_range = depth_cfg.far_clip
        self.min_range = depth_cfg.near_clip


class WarpDepthCamera:
    def __init__(self, terrain, depth_cfg, local_positions, local_quats, device):
        # 初始化 NVIDIA Warp 运行环境
        wp.init()

        # 保存深度相机配置
        self.depth_cfg = depth_cfg

        # 深度相机数量
        self.num_cameras = len(local_positions)

        # 将项目配置转换为 WarpCam 所需的配置格式
        self.cfg = WMPDepthCameraConfig(depth_cfg)

        # 复制地形顶点，并移除地形边界偏移，使其回到仿真世界坐标系
        vertices = np.asarray(terrain.vertices, dtype=np.float32).copy()
        vertices[:, 0] -= terrain.cfg.border_size
        vertices[:, 1] -= terrain.cfg.border_size

        # 将三角面索引展开为 Warp Mesh 所需的一维数组
        triangles = np.asarray(terrain.triangles, dtype=np.int32).reshape(-1)

        # 在目标设备上保存地形顶点和三角面索引
        self.mesh_points = torch.tensor(vertices, device=device, dtype=torch.float32)
        self.mesh_indices = torch.tensor(triangles, device=device, dtype=torch.int32)

        # 使用地形顶点和三角面创建用于射线检测的 Warp Mesh
        self.wp_mesh = wp.Mesh(points=wp.from_torch(self.mesh_points, dtype=wp.vec3), indices=wp.from_torch(self.mesh_indices, dtype=wp.int32))

        # 所有相机共用同一个地形 Mesh
        mesh_ids = [self.wp_mesh.id] * self.num_cameras

        # 将每个相机对应的 Mesh ID 转换成 Warp 数组
        self.mesh_ids_array = wp.array(mesh_ids, dtype=wp.uint64)

        # Warp 深度相机
        self.camera = WarpCam(num_envs=self.num_cameras, config=self.cfg, mesh_ids_array=self.mesh_ids_array, device=device)

        # 深度图缓冲区 (N, 1, H, W)
        self.pixels = torch.zeros((self.num_cameras, 1, self.cfg.height, self.cfg.width), device=device, requires_grad=False)

        # 相机世界坐标 (N, 1, xyz)
        self.camera_positions = torch.zeros((self.num_cameras, 1, 3), device=device, requires_grad=False)

        # 相机世界四元数 (N, 1, xyzw)
        self.camera_orientations = torch.zeros((self.num_cameras, 1, 4), device=device, requires_grad=False)
        self.camera_orientations[..., 3] = 1.0

        # 相机相对机身的位置 (N, xyz)
        self.local_positions = torch.tensor(np.asarray(local_positions, dtype=np.float32), device=device, requires_grad=False)

        # 相机相对机身的四元数 (N, xyzw)
        self.local_quats = torch.tensor(np.asarray(local_quats, dtype=np.float32), device=device, requires_grad=False)

        # Warp 相机坐标系转换角 (xyz)
        frame_euler = torch.tensor([-torch.pi / 2, 0.0, -torch.pi / 2], device=device, dtype=torch.float32)

        # 每个相机的固定坐标系转换四元数
        self.frame_quat = quat_from_euler_xyz(frame_euler[0].view(1), frame_euler[1].view(1), frame_euler[2].view(1)).repeat(self.num_cameras, 1)

        self.camera.set_pose_tensor(self.camera_positions, self.camera_orientations)
        self.camera.set_image_tensors(self.pixels, segmentation_pixels=None)

    def render(self, root_positions, root_quats):
        self.camera_positions[:, 0, :] = quat_apply(root_quats, self.local_positions) + root_positions
        self.camera_orientations[:, 0, :] = quat_mul(root_quats, quat_mul(self.local_quats, self.frame_quat))
        depth = self.camera.capture().squeeze(1)
        depth = torch.clip(depth, self.depth_cfg.near_clip, self.depth_cfg.far_clip)
        return (depth - self.depth_cfg.near_clip) / (self.depth_cfg.far_clip - self.depth_cfg.near_clip) - 0.5
