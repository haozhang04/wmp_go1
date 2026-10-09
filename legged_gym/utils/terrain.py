# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import random
import numpy as np
from isaacgym import terrain_utils
from scipy.ndimage import binary_dilation
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import trimesh


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.added_trimesh = None
        if cfg.curriculum:
            self.curriculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.evaluated_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_threshold)
            if self.added_trimesh is not None:
                self.vertices, self.triangles = trimesh.combine_trimeshes((self.vertices, self.triangles), self.added_trimesh)

            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=np.ones((3, 1)))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def evaluated_terrain(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def curriculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain", width=self.width_per_env_pixels, length=self.width_per_env_pixels, vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty, i=0, j=0):
        terrain = terrain_utils.SubTerrain("terrain", width=self.width_per_env_pixels, length=self.width_per_env_pixels, vertical_scale=self.cfg.vertical_scale, horizontal_scale=self.cfg.horizontal_scale)
        amplitude = 0.1 + 0.2 * difficulty
        slope = difficulty * 0.50
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.21
        stepping_stones_size = 0.4 - (0.4 - 0.25) * difficulty
        stone_distance = 0.1 + (0.15 - 0.1) * difficulty
        stepping_stones_height = 0.05 + (0.15 - 0.05) * difficulty
        tilt_width = 0.32 - 0.04 * difficulty
        stairs_step_width = 0.30 + random.random() * 0.04
        normalized_difficulty = min(difficulty * self.cfg.num_rows / max(self.cfg.num_rows - 1, 1), 1.0)
        gap_size = 0.10 + (0.90 - 0.10) * normalized_difficulty
        climb_depth = 0.15 + (0.60 - 0.15) * normalized_difficulty
        stairs_cliff_step_height = 0.05 + (0.26 - 0.05) * normalized_difficulty
        slope_cliff_angle_deg = 10.0 + (35.0 - 10.0) * normalized_difficulty

        if choice < self.proportions[0]:
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[1]:
            if choice < (self.proportions[0] + self.proportions[1]) / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=stairs_step_width, step_height=step_height, platform_size=3.)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            gap_terrain(terrain, gap_size=gap_size)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[6]:  # continuous_gap
            continuous_gap_terrain(terrain, difficulty)

        elif choice < self.proportions[7]:
            bream_terrain(terrain, stone_size=0.9 - 0.4 * difficulty, stone_distance=0.1 + 0.4 * difficulty, max_height=step_height, platform_size=2.0)

        elif choice < self.proportions[8]:
            # stairs_cliff_terrain(terrain, step_height=stairs_cliff_step_height, step_width=0.3, num_steps=10, platform_length=0.2)
            stairs_cliff_terrain(terrain, step_height=stairs_cliff_step_height, step_width=random.choice((0.2, 0.3, 0.4)), num_steps=10, platform_length=0.2)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.04, max_height=0.04, step=0.005, downsampled_scale=0.2)

        elif choice < self.proportions[9]:
            slope_cliff_terrain(terrain, slope_angle_deg=slope_cliff_angle_deg)

        elif choice < self.proportions[10]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=stepping_stones_height, platform_size=2.)

        elif choice < self.proportions[11]:
            bridge_size = 0.6 - (0.6 - 0.26) * difficulty
            stepping_one_bridge_terrain(terrain, stone_size=bridge_size, depth=1.0, platform_size=2.0)

        elif choice < self.proportions[12]:
            climb_terrain(terrain, depth=climb_depth)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)

        elif choice < self.proportions[13]:
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            box_z = 1
            box_x = 0.4 + 0.4 * np.random.random()
            tilt_front_left_trimesh = trimesh.box_trimesh(np.array([box_x, (self.env_width - tilt_width) / 2, box_z], dtype=np.float32), np.array([env_origin_x + self.cfg.border_size + 2 + box_x / 2, env_origin_y + self.cfg.border_size - tilt_width / 2 - (self.env_width - tilt_width) / 4, box_z / 2], dtype=np.float32))
            if self.added_trimesh is None:
                self.added_trimesh = tilt_front_left_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, tilt_front_left_trimesh)

            tilt_front_right_trimesh = trimesh.box_trimesh(np.array([box_x, (self.env_width - tilt_width) / 2, box_z], dtype=np.float32), np.array([env_origin_x + self.cfg.border_size + 2 + box_x / 2, env_origin_y + self.cfg.border_size + tilt_width / 2 + (self.env_width - tilt_width) / 4, box_z / 2], dtype=np.float32))
            if self.added_trimesh is None:
                self.added_trimesh = tilt_front_right_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, tilt_front_right_trimesh)

            tilt_back_left_trimesh = trimesh.box_trimesh(np.array([box_x, (self.env_width - tilt_width) / 2, box_z], dtype=np.float32), np.array([env_origin_x + self.cfg.border_size - 2 - box_x / 2, env_origin_y + self.cfg.border_size - tilt_width / 2 - (self.env_width - tilt_width) / 4, box_z / 2], dtype=np.float32))
            if self.added_trimesh is None:
                self.added_trimesh = tilt_back_left_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, tilt_back_left_trimesh)

            tilt_back_right_trimesh = trimesh.box_trimesh(np.array([box_x, (self.env_width - tilt_width) / 2, box_z], dtype=np.float32), np.array([env_origin_x + self.cfg.border_size - 2 - box_x / 2, env_origin_y + self.cfg.border_size + tilt_width / 2 + (self.env_width - tilt_width) / 4, box_z / 2], dtype=np.float32))
            if self.added_trimesh is None:
                self.added_trimesh = tilt_back_right_trimesh
            else:
                self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, tilt_back_right_trimesh)

        elif choice < self.proportions[14]:
            # Inverted-U stool without a backrest. Its clear height follows the
            # crawl curriculum, while all three boards share a random 4-10 cm
            # thickness. Forward length varies from 0.4-0.8 m and lateral width
            # varies from 1.5-2.0 m.
            stool_clear_height = 0.42 - 0.15 * difficulty
            board_thickness = np.random.uniform(0.04, 0.10)
            stool_length = np.random.uniform(0.40, 0.80)
            stool_width = np.random.uniform(1.50, 2.00)

            # Keep only a stool-width corridor and turn both sides into cliffs,
            # preventing the robot from bypassing the stool laterally.
            cliff_depth = int(-1.0 / terrain.vertical_scale)
            corridor_width = max(1, int(round(stool_width / terrain.horizontal_scale)))
            corridor_y1 = max(0, terrain.width // 2 - corridor_width // 2)
            corridor_y2 = min(terrain.width, corridor_y1 + corridor_width)
            terrain.height_field_raw[:, :] = cliff_depth
            terrain.height_field_raw[:, corridor_y1:corridor_y2] = 0

            # Spawn near the beginning of the 8 m tile, leaving the remaining
            # length for three consecutive stools in the +X direction.
            terrain.env_origin_x_offset = -3.50
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            stool_boxes = (
                # seat
                ((stool_length, stool_width, board_thickness), (0.0, 0.0, stool_clear_height + board_thickness / 2)),
                # left and right side panels
                ((stool_length, board_thickness, stool_clear_height), (0.0, -(stool_width - board_thickness) / 2, stool_clear_height / 2)),
                ((stool_length, board_thickness, stool_clear_height), (0.0, (stool_width - board_thickness) / 2, stool_clear_height / 2)),
            )
            # Three stools are arranged after the spawn area. Their edge-to-edge
            # gaps remain exactly 2 m for every sampled stool length.
            first_stool_x = -2.30
            stool_spacing = stool_length + 2.00
            for x_offset in (first_stool_x, first_stool_x + stool_spacing, first_stool_x + 2 * stool_spacing):
                stool_center = np.array([env_origin_x + self.cfg.border_size + x_offset, env_origin_y + self.cfg.border_size, 0.0], dtype=np.float32)
                for size, offset in stool_boxes:
                    box_mesh = trimesh.box_trimesh(np.asarray(size, dtype=np.float32), stool_center + np.asarray(offset, dtype=np.float32))
                    if self.added_trimesh is None:
                        self.added_trimesh = box_mesh
                    else:
                        self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, box_mesh)
        elif choice < self.proportions[15]:
            crawl_height = 0.42 - 0.15 * difficulty
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            terrain.env_origin_x_offset = 1.0 - self.env_length / 2.0
            bar_thicknesses = np.random.uniform(0.2, 0.5, size=2)
            bar_start_positions = (4.0, self.env_length - bar_thicknesses[1])
            bar_height = 1.0
            for bar_thickness, bar_start_position in zip(bar_thicknesses, bar_start_positions):
                bar_center_x = env_origin_x + self.cfg.border_size - self.env_length / 2.0 + bar_start_position + bar_thickness / 2.0
                bar_size = np.array([bar_thickness, self.env_width, bar_height], dtype=np.float32)
                bar_center = np.array([bar_center_x, env_origin_y + self.cfg.border_size, crawl_height + bar_height / 2.0], dtype=np.float32)
                bar_mesh = trimesh.box_trimesh(bar_size, bar_center)
                if self.added_trimesh is None:
                    self.added_trimesh = bar_mesh
                else:
                    self.added_trimesh = trimesh.combine_trimeshes(self.added_trimesh, bar_mesh)
        else:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        return terrain


    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length + getattr(terrain, "env_origin_x_offset", 0.0)
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = getattr(terrain, "env_origin_z", None)
        if env_origin_z is None:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size):
    gap_size = int(gap_size / terrain.horizontal_scale)
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = int(center_x - 1 / terrain.horizontal_scale)
    x2 = int(center_x + 2 / terrain.horizontal_scale)
    x3 = x1 - gap_size
    x4 = x2 + gap_size
    width = 1 + 1.0 * np.random.random()
    half_width = width / 2
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    x5 = gap_size

    terrain.height_field_raw[:, :] = -1000
    terrain.height_field_raw[x5:x3, y1:y2] = 0
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    terrain.height_field_raw[x4:, y1:y2] = 0

def stairs_cliff_terrain(terrain, step_height, step_width=0.3, num_steps=10, platform_length=0.2, channel_width=2.4, start_after_origin=0.9, origin_x_offset=-3.1):
    step_height = int(round(step_height / terrain.vertical_scale))
    step_width = max(1, int(round(step_width / terrain.horizontal_scale)))
    platform_length = max(1, int(round(platform_length / terrain.horizontal_scale)))
    channel_width = max(1, int(round(channel_width / terrain.horizontal_scale)))
    start_after_origin = int(round(start_after_origin / terrain.horizontal_scale))
    origin_x_offset_px = int(round(origin_x_offset / terrain.horizontal_scale))
    terrain.env_origin_x_offset = origin_x_offset

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    y1 = max(0, center_y - channel_width // 2)
    y2 = min(terrain.width, center_y + channel_width // 2)
    origin_x = center_x + origin_x_offset_px
    x = min(terrain.length, origin_x + start_after_origin)
    remaining_length = terrain.length - x
    max_num_steps = max(1, (remaining_length - platform_length) // (2 * step_width))
    num_steps = min(num_steps, max_num_steps)

    terrain.height_field_raw[:, :] = -1000
    terrain.height_field_raw[:, y1:y2] = 0

    for step_idx in range(num_steps):
        x_next = min(terrain.length, x + step_width)
        terrain.height_field_raw[x:x_next, y1:y2] = (step_idx + 1) * step_height
        x = x_next

    x_next = min(terrain.length, x + platform_length)
    terrain.height_field_raw[x:x_next, y1:y2] = num_steps * step_height
    x = x_next

    for step_idx in range(num_steps):
        x_next = min(terrain.length, x + step_width)
        terrain.height_field_raw[x:x_next, y1:y2] = (num_steps - step_idx - 1) * step_height
        x = x_next

    terrain.env_origin_z = 0.


def slope_cliff_terrain(terrain, slope_angle_deg, max_slope_length=2.8, platform_length=0.6, channel_width=2.4, start_after_origin=0.9, origin_x_offset=-3.1):
    slope_angle_deg = max(slope_angle_deg, 0.0)
    slope = np.tan(np.deg2rad(slope_angle_deg))
    platform_length = max(1, int(round(platform_length / terrain.horizontal_scale)))
    channel_width = max(1, int(round(channel_width / terrain.horizontal_scale)))
    start_after_origin = int(round(start_after_origin / terrain.horizontal_scale))
    origin_x_offset_px = int(round(origin_x_offset / terrain.horizontal_scale))
    terrain.env_origin_x_offset = origin_x_offset

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    y1 = max(0, center_y - channel_width // 2)
    y2 = min(terrain.width, center_y + channel_width // 2)
    origin_x = center_x + origin_x_offset_px
    x = min(terrain.length, origin_x + start_after_origin)
    remaining_length = terrain.length - x
    available_slope_length = max(1, (remaining_length - platform_length) // 2)
    slope_length = max(1, min(available_slope_length, int(round(max_slope_length / terrain.horizontal_scale))))
    platform_height = slope * slope_length * terrain.horizontal_scale

    terrain.height_field_raw[:, :] = -1000
    terrain.height_field_raw[:, y1:y2] = 0

    ramp_height_m = np.linspace(0.0, platform_height, slope_length)
    ramp_height = np.round(ramp_height_m / terrain.vertical_scale).astype(np.int16)
    top_height = int(ramp_height[-1])

    x_next = min(terrain.length, x + slope_length)
    terrain.height_field_raw[x:x_next, y1:y2] = ramp_height[:x_next - x, None]
    x = x_next

    x_next = min(terrain.length, x + platform_length)
    terrain.height_field_raw[x:x_next, y1:y2] = top_height
    x = x_next

    x_next = min(terrain.length, x + slope_length)
    down_height = ramp_height[:x_next - x][::-1]
    terrain.height_field_raw[x:x_next, y1:y2] = down_height[:, None]

    terrain.env_origin_z = 0.


def bream_terrain(terrain, stone_size, stone_distance, max_height, platform_size=2.0, depth=1.0):
    bream_length = max(1, int(stone_size / terrain.horizontal_scale))
    stone_distance = max(1, int(stone_distance / terrain.horizontal_scale))
    platform_size = max(1, int(platform_size / terrain.horizontal_scale))
    max_height = max(1, int(round(max_height / terrain.vertical_scale)))
    height_step = max(1, max_height // 4)
    height_range = np.arange(0, max_height + 1, step=height_step, dtype=np.int16)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = center_x - platform_size // 2
    x2 = center_x + platform_size // 2
    y1 = center_y - platform_size // 2
    y2 = center_y + platform_size // 2

    min_bream_width = max(1, int(0.5 / terrain.horizontal_scale))
    max_bream_width = max(min_bream_width + 1, int(1.0 / terrain.horizontal_scale))

    start_x_front = x1 - 1
    while start_x_front >= 0:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row_y = int(center_y - bream_width / 2)
        stop_x_front = max(0, start_x_front - bream_length)
        terrain.height_field_raw[stop_x_front:start_x_front, row_y:row_y + bream_width] = np.random.choice(height_range)
        start_x_front -= bream_length + stone_distance

    start_x_back = x2 + 1
    while start_x_back < terrain.length:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row_y = int(center_y - bream_width / 2)
        stop_x_back = min(terrain.length, start_x_back + bream_length)
        terrain.height_field_raw[start_x_back:stop_x_back, row_y:row_y + bream_width] = np.random.choice(height_range)
        start_x_back += bream_length + stone_distance

    terrain.height_field_raw[x1:x2, y1:y2] = 0
    terrain.env_origin_z = 0.


def stepping_one_bridge_terrain(terrain, stone_size, depth=1.0, platform_size=2.0):
    stone_size = max(1, int(stone_size / terrain.horizontal_scale))
    platform_size = max(1, int(platform_size / terrain.horizontal_scale))
    platform_y = (terrain.width - platform_size) // 2
    row_y = int(platform_y + platform_size / 2 - stone_size / 2)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    height_range = np.arange(0, 10, step=2, dtype=np.int16)
    start_x = 0
    while start_x < terrain.length:
        stop_x = min(terrain.length, start_x + stone_size)
        terrain.height_field_raw[start_x:stop_x, row_y:row_y + stone_size] = np.random.choice(height_range)
        start_x += stone_size

    center_x = terrain.length // 2
    x1 = center_x - platform_size // 2
    x2 = center_x + platform_size // 2
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
    terrain.env_origin_z = 0.


def continuous_gap_terrain(terrain, difficulty=0.0):
    scale, vertical_scale = terrain.horizontal_scale, terrain.vertical_scale
    gap = max(1, round((0.1 + 0.4 * np.clip(difficulty / 0.9, 0.0, 1.0)) / scale))
    first = max(1, round(1.8 / scale))
    min_platform, max_platform = (max(1, round(length / scale)) for length in (0.3, 0.4))
    remaining, widths = terrain.length - first - gap, []
    while remaining >= min_platform + gap:
        width = np.random.randint(min_platform, min(max_platform, remaining - gap) + 1)
        widths.append(width)
        remaining -= width + gap
    if not widths:
        raise ValueError("Continuous gap does not fit in terrain tile")

    pit = round(-0.3 / vertical_scale)
    terrain.height_field_raw.fill(0)
    x = first + remaining
    for width in widths:
        terrain.height_field_raw[x:x + gap, :] = pit
        x += gap
        terrain.height_field_raw[x:x + width, :] = round(np.random.uniform(0.0, 0.15) / vertical_scale)
        x += width
    terrain.height_field_raw[x:, :] = pit
    terrain.env_origin_x_offset = 0.9 - terrain.length * scale / 2
    terrain.env_origin_z = 0.0


def climb_terrain(terrain, depth):
    depth = int(round(depth / terrain.vertical_scale))
    obstacle_lengths = [0.9 + 0.2 * np.random.random() for _ in range(2)]
    obstacle_lengths_px = [int(round(length / terrain.horizontal_scale)) for length in obstacle_lengths]
    obstacle_starts_px = [int(round(3.3 / terrain.horizontal_scale)), terrain.length - obstacle_lengths_px[1]]
    terrain.env_origin_x_offset = 1.0 - terrain.length * terrain.horizontal_scale / 2.0
    terrain.env_origin_z = 0.0
    for start, length in zip(obstacle_starts_px, obstacle_lengths_px):
        terrain.height_field_raw[start:start + length, :] = depth


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows - 1, :num_cols - 1] += (
                    hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (
                    hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0
