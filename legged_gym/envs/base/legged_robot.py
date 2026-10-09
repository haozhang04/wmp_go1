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

import math
import random

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.terrain_names import TERRAIN_NAMES
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from rsl_rl.datasets.motion_loader import AMPLoader
import cv2

COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        # get terrain type idx
        self.wave_start_idx = 0
        self.wave_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:1]) - 1e-9)
        self.rough_slope_start_idx = self.wave_end_idx
        self.rough_slope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:2]) - 1e-9)
        self.stairs_up_start_idx = self.rough_slope_end_idx
        self.stairs_up_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:3]) - 1e-9)
        self.stairs_down_start_idx = self.stairs_up_end_idx
        self.stairs_down_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:4]) - 1e-9)
        self.discrete_start_idx = self.stairs_down_end_idx
        self.discrete_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:5]) - 1e-9)
        self.gap_start_idx = self.discrete_end_idx
        self.gap_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:6]) - 1e-9)
        self.continuous_gap_start_idx = self.gap_end_idx
        self.continuous_gap_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:7]) - 1e-9)
        self.bream_start_idx = self.continuous_gap_end_idx
        self.bream_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:8]) - 1e-9)
        self.stairs_cliff_start_idx = self.bream_end_idx
        self.stairs_cliff_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:9]) - 1e-9)
        self.slope_cliff_start_idx = self.stairs_cliff_end_idx
        self.slope_cliff_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:10]) - 1e-9)
        self.stepping_stones_start_idx = self.slope_cliff_end_idx
        self.stepping_stones_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:11]) - 1e-9)
        self.stepping_one_bridge_start_idx = self.stepping_stones_end_idx
        self.stepping_one_bridge_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:12]) - 1e-9)
        self.climb_start_idx = self.stepping_one_bridge_end_idx
        self.climb_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:13]) - 1e-9)
        self.tilt_start_idx = self.climb_end_idx
        self.tilt_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:14]) - 1e-9)
        self.stool_start_idx = self.tilt_end_idx
        self.stool_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:15]) - 1e-9)
        self.crawl_start_idx = self.stool_end_idx
        self.crawl_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:16]) - 1e-9)
        self.rough_flat_start_idx = self.crawl_end_idx
        self.rough_flat_end_idx = self.cfg.env.num_envs

        # Each terrain group maintains independent command curriculum limits.
        self.omni_command_x = 0.5
        self.omni_command_y = 0.5
        self.rough_flat_command_x = 0.5
        self.rough_flat_command_y = 0.5
        self.stairs_discrete_command_x = 0.5
        self.stairs_discrete_command_y = 0.5
        self.forward_command_x = 0.5
        self.slope_cliff_command_x = 0.5

        self.sim_params = sim_params
        self.height_samples = None
        # for debug
        self.debug_viz = True
        self.lookat_id = 0
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[0], self.cfg.depth.resized[1]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.command_curriculum_reward_sums = {name: torch.zeros((), device=self.device) for name in ("omni", "rough_flat", "stairs_discrete", "forward", "slope_cliff")}
        self.command_curriculum_episode_counts = {"omni": 0, "rough_flat": 0, "stairs_discrete": 0, "forward": 0, "slope_cliff": 0}
        self.command_curriculum_group_sizes = {
            "omni": self.rough_slope_end_idx,
            "rough_flat": self.rough_flat_end_idx - self.rough_flat_start_idx,
            "stairs_discrete": self.discrete_end_idx - self.stairs_up_start_idx,
            "forward": self.rough_flat_start_idx - self.discrete_end_idx - (self.slope_cliff_end_idx - self.slope_cliff_start_idx),
            "slope_cliff": self.slope_cliff_end_idx - self.slope_cliff_start_idx,
        }
        self.init_done = True

        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.latency_range = [int((self.cfg.domain_rand.latency_range[0] + 1e-8) / self.sim_params.dt), int((self.cfg.domain_rand.latency_range[1] - 1e-8) / self.sim_params.dt) + 1]

        if self.cfg.rewards.reward_curriculum:
            self.reward_curriculum_coef = [schedule[2] for schedule in self.cfg.rewards.reward_curriculum_schedule]

        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.global_counter += 1
        self.total_env_steps_counter += 1

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        #for action latency
        rng = self.latency_range
        action_latency = random.randint(rng[0], rng[1])

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if (self.cfg.domain_rand.randomize_action_latency and _ < action_latency):
                self.torques = self._compute_torques(self.last_actions).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if(self.cfg.domain_rand.randomize_motor_strength):
                rng = self.cfg.domain_rand.motor_strength_range
                self.torques = self.torques * torch_rand_float(rng[0], rng[1], self.torques.shape, device=self.device)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        if getattr(self.cfg.depth, "backend", "isaac") == "warp":
            depth_images = self.warp_depth_camera.render(self.root_states[self.depth_index, :3], self.root_states[self.depth_index, 3:7])
            init_flag = self.episode_length_buf[self.depth_index] <= 1
            for i, depth_image in enumerate(depth_images):
                if init_flag[i]:
                    self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                else:
                    self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.unsqueeze(0)], dim=0)
            return
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        start_time = time()
        self.gym.start_access_image_tensors(self.sim)
        for i in range(len(self.depth_index)):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[self.depth_index[i]], self.cam_handles[i], gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf[self.depth_index] <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
        self.gym.end_access_image_tensors(self.sim)
        print('acquiring depth image time:', time()-start_time)


    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Filter foot contacts across two consecutive simulation steps because
        # PhysX contact reporting on triangle meshes can be intermittent.
        contact = self.sensor_forces[:, :, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # the original code call _post_physics_step_callback before compute reward, which seems unreasonable. e.g., the
        # current action follows the current commands, while _post_physics_step_callback may resample command, resulting a low reward.
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)

        self.update_depth_buffer()

        # after reset idx, the base_lin_vel, base_ang_vel, projected_gravity, height has changed, so should be re-computed
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # self._post_physics_step_callback()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 接触终止
        self.contact_terminate = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,dim=1)
        
        # 速度违反
        vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        self.vel_violate = ((vel_error > 1.5) & (self.commands[:, 0] < 0.0)) | ((vel_error < -1.5) & (self.commands[:, 0] > 0.0))
        self.vel_violate *= (self.terrain_levels > 3)

        # 跌落
        self.fall = (self.root_states[:, 9] < -3.0) | (self.projected_gravity[:, 2] > -0.0)

        # 超时终止
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf = self.contact_terminate.clone()
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.vel_violate
        self.reset_buf |= self.fall

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)

        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0
        self.latency_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.contact_filt[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["reset_contact"] = torch.mean(self.contact_terminate[env_ids].float())
        self.extras["episode"]["reset_vel_violation"] = torch.mean(self.vel_violate[env_ids].float())
        self.extras["episode"]["reset_fall"] = torch.mean(self.fall[env_ids].float())
        self.extras["episode"]["reset_timeout"] = torch.mean(self.time_out_buf[env_ids].float())
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            for terrain_name in TERRAIN_NAMES:
                start_idx = getattr(self, f"{terrain_name}_start_idx")
                end_idx = getattr(self, f"{terrain_name}_end_idx")
                if start_idx < end_idx:
                    self.extras["episode"][f"tl_{terrain_name}"] = torch.mean(self.terrain_levels[start_idx:end_idx].float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["cmd_omni_x"] = self.omni_command_x
            self.extras["episode"]["cmd_omni_y"] = self.omni_command_y
            self.extras["episode"]["cmd_rough_flat_x"] = self.rough_flat_command_x
            self.extras["episode"]["cmd_rough_flat_y"] = self.rough_flat_command_y
            self.extras["episode"]["cmd_stairs_discrete_x"] = self.stairs_discrete_command_x
            self.extras["episode"]["cmd_stairs_discrete_y"] = self.stairs_discrete_command_y
            self.extras["episode"]["cmd_forward_x"] = self.forward_command_x
            self.extras["episode"]["cmd_slope_cliff_x"] = self.slope_cliff_command_x
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # reward curriculum
            if self.cfg.rewards.reward_curriculum:
                for j in range(len(self.cfg.rewards.reward_curriculum_term)):
                    if(name == self.cfg.rewards.reward_curriculum_term[j]):
                        rew *= self.reward_curriculum_coef[j]
                # print('reward:', name, ' coef:', self.reward_curriculum_coef)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def compute_observations(self):
        """ Computes observations
        """
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        if (self.cfg.env.privileged_obs):
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

            if self.cfg.domain_rand.randomize_friction:
                self.privileged_obs_buf= torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)

            if self.cfg.domain_rand.randomize_restitution:
                self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_base_mass):
                self.privileged_obs_buf = torch.cat((self.randomized_added_masses ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_com_pos):
                self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_gains):
                self.privileged_obs_buf = torch.cat(((self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains ,self.privileged_obs_buf), dim=-1)
                self.privileged_obs_buf = torch.cat(((self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains, self.privileged_obs_buf), dim=-1)

            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1)
            contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
            self.privileged_obs_buf = torch.cat((contact_flag, self.privileged_obs_buf), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        elif self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)

    def get_full_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        pos = self.root_states[:, :3]
        if (self.cfg.terrain.measure_heights):
            pos[:, 2:3] = pos[:, 2:3] - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        rot = self.root_states[:, 3:7]
        foot_vel = torch.zeros_like(foot_pos)
        return torch.cat((pos, rot, joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, foot_vel), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera and self.cfg.depth.backend == "isaac":
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            # if env_id==0:
            #     # prepare friction randomization
            #     friction_range = self.cfg.domain_rand.friction_range
            #     num_buckets = 64
            #     bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            #     friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
            #     self.friction_coeffs = friction_buckets[bucket_ids]
            #
            # for s in range(len(props)):
            #     props[s].friction = self.friction_coeffs[env_id]
            rng = self.cfg.domain_rand.friction_range
            self.randomized_frictions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].friction = self.randomized_frictions[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            rng = self.cfg.domain_rand.restitution_range
            self.randomized_restitutions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].restitution = self.randomized_restitutions[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            self.randomized_added_masses[env_id] = added_mass
            props[0].mass += added_mass

        # randomize com position
        if self.cfg.domain_rand.randomize_com_pos:
            rng = self.cfg.domain_rand.com_x_pos_range
            com_x_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,0] = com_x_pos
            rng = self.cfg.domain_rand.com_y_pos_range
            com_y_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,1] = com_y_pos
            rng = self.cfg.domain_rand.com_z_pos_range
            com_z_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,2] = com_z_pos
            props[0].com +=  gymapi.Vec3(com_x_pos,com_y_pos,com_z_pos)

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_env_ids = (~self.direct_yaw_mask).nonzero(as_tuple=False).flatten()
            self.commands[heading_env_ids, 2] = torch.clip(2.0 * wrap_to_pi(self.commands[heading_env_ids, 3] - heading[heading_env_ids]), -math.pi, math.pi)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.measured_forward_heights = self._get_forward_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.direct_yaw_mask[env_ids] = False
        if not self.cfg.commands.curriculum:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
            return

        omni_env_ids = env_ids[env_ids < self.rough_slope_end_idx]
        if len(omni_env_ids) > 0:
            self.commands[omni_env_ids, 0] = torch_rand_float(-self.omni_command_x, self.omni_command_x, (len(omni_env_ids), 1), device=self.device).squeeze(1)
            self.commands[omni_env_ids, 1] = torch_rand_float(-self.omni_command_y, self.omni_command_y, (len(omni_env_ids), 1), device=self.device).squeeze(1)

        rough_flat_env_ids = env_ids[(env_ids >= self.rough_flat_start_idx) & (env_ids < self.rough_flat_end_idx)]
        if len(rough_flat_env_ids) > 0:
            self.commands[rough_flat_env_ids, 0] = torch_rand_float(-self.rough_flat_command_x, self.rough_flat_command_x, (len(rough_flat_env_ids), 1), device=self.device).squeeze(1)
            self.commands[rough_flat_env_ids, 1] = torch_rand_float(-self.rough_flat_command_y, self.rough_flat_command_y, (len(rough_flat_env_ids), 1), device=self.device).squeeze(1)

        stairs_discrete_env_ids = env_ids[(env_ids >= self.stairs_up_start_idx) & (env_ids < self.discrete_end_idx)]
        if len(stairs_discrete_env_ids) > 0:
            self.commands[stairs_discrete_env_ids, 0] = torch_rand_float(-self.stairs_discrete_command_x, self.stairs_discrete_command_x, (len(stairs_discrete_env_ids), 1), device=self.device).squeeze(1)
            self.commands[stairs_discrete_env_ids, 1] = torch_rand_float(-self.stairs_discrete_command_y, self.stairs_discrete_command_y, (len(stairs_discrete_env_ids), 1), device=self.device).squeeze(1)

        all_direction_env_ids = env_ids[(env_ids < self.discrete_end_idx) | (env_ids >= self.rough_flat_start_idx)]
        if len(all_direction_env_ids) > 0:
            if self.cfg.commands.heading_command:
                self.commands[all_direction_env_ids, 3] = torch_rand_float(-math.pi, math.pi, (len(all_direction_env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[all_direction_env_ids, 2] = torch_rand_float(-math.pi, math.pi, (len(all_direction_env_ids), 1), device=self.device).squeeze(1)

        forward_env_ids = env_ids[(env_ids >= self.discrete_end_idx) & (env_ids < self.rough_flat_start_idx)]
        if len(forward_env_ids) > 0:
            self.commands[forward_env_ids, 0] = torch_rand_float(0.0, self.forward_command_x, (len(forward_env_ids), 1), device=self.device).squeeze(1)
            self.commands[forward_env_ids, 1] = 0.0
            if self.cfg.commands.heading_command:
                self.commands[forward_env_ids, 3] = 0.0
            else:
                self.commands[forward_env_ids, 2] = 0.0

        slope_cliff_env_ids = env_ids[(env_ids >= self.slope_cliff_start_idx) & (env_ids < self.slope_cliff_end_idx)]
        if len(slope_cliff_env_ids) > 0:
            self.commands[slope_cliff_env_ids, 0] = torch_rand_float(0.0, self.slope_cliff_command_x, (len(slope_cliff_env_ids), 1), device=self.device).squeeze(1)

        if len(all_direction_env_ids) > 0:
            rand_prob = torch.rand(len(all_direction_env_ids), device=self.device)
            limit_env_ids = all_direction_env_ids[rand_prob < self.cfg.commands.limit_vel_prob]
            zero_env_ids = all_direction_env_ids[(rand_prob >= self.cfg.commands.limit_vel_prob) & (rand_prob < self.cfg.commands.limit_vel_prob + self.cfg.commands.zero_command_prob)]

            if len(limit_env_ids) > 0:
                # Uniformly sample the 27 combinations of min/zero/max commands.
                choice = torch.randint(0, 3, (len(limit_env_ids), 3), device=self.device)
                limit_values = torch.tensor(self.cfg.commands.limit_vel, device=self.device)
                direct_yaw_values = torch.tensor(self.cfg.commands.direct_yaw_values, device=self.device)
                x_limit = torch.where(limit_env_ids < self.rough_slope_end_idx, self.omni_command_x, torch.where(limit_env_ids < self.discrete_end_idx, self.stairs_discrete_command_x, self.rough_flat_command_x))
                y_limit = torch.where(limit_env_ids < self.rough_slope_end_idx, self.omni_command_y, torch.where(limit_env_ids < self.discrete_end_idx, self.stairs_discrete_command_y, self.rough_flat_command_y))
                self.commands[limit_env_ids, 0] = limit_values[choice[:, 0]] * x_limit
                self.commands[limit_env_ids, 1] = limit_values[choice[:, 1]] * y_limit
                self.commands[limit_env_ids, 2] = direct_yaw_values[choice[:, 2]]
                self.direct_yaw_mask[limit_env_ids] = True

            self.commands[zero_env_ids, :2] = 0.0

        # Forward-only terrains retain the original low-speed filtering.
        self.commands[forward_env_ids, :2] *= (torch.norm(self.commands[forward_env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # Keep forward direction.
        self.commands[self.gap_start_idx:self.gap_end_idx, 3] = 0
        self.commands[self.continuous_gap_start_idx:self.continuous_gap_end_idx, 3] = 0
        self.commands[self.bream_start_idx:self.bream_end_idx, 3] = 0
        self.commands[self.stairs_cliff_start_idx:self.stairs_cliff_end_idx, 3] = 0
        self.commands[self.slope_cliff_start_idx:self.slope_cliff_end_idx, 3] = 0
        self.commands[self.stepping_stones_start_idx:self.stepping_stones_end_idx, 3] = 0
        self.commands[self.stepping_one_bridge_start_idx:self.stepping_one_bridge_end_idx, 3] = 0
        self.commands[self.climb_start_idx:self.climb_end_idx, 3] = 0
        self.commands[self.tilt_start_idx:self.tilt_end_idx, 3] = 0
        self.commands[self.crawl_start_idx:self.crawl_end_idx, 3] = 0
        self.commands[self.stool_start_idx:self.stool_end_idx, 3] = 0

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if self.custom_origins:
            # 根据地形类型限制机器人初始位置的随机范围
            tilt_env_ids = env_ids[(env_ids >= self.tilt_start_idx) & (env_ids < self.tilt_end_idx)]
            gap_env_ids = env_ids[(env_ids >= self.gap_start_idx) & (env_ids < self.gap_end_idx)]
            continuous_gap_env_ids = env_ids[(env_ids >= self.continuous_gap_start_idx) & (env_ids < self.continuous_gap_end_idx)]
            bream_env_ids = env_ids[(env_ids >= self.bream_start_idx) & (env_ids < self.bream_end_idx)]
            stairs_cliff_env_ids = env_ids[(env_ids >= self.stairs_cliff_start_idx) & (env_ids < self.stairs_cliff_end_idx)]
            slope_cliff_env_ids = env_ids[(env_ids >= self.slope_cliff_start_idx) & (env_ids < self.slope_cliff_end_idx)]
            stepping_stones_env_ids = env_ids[(env_ids >= self.stepping_stones_start_idx) & (env_ids < self.stepping_stones_end_idx)]
            stepping_one_bridge_env_ids = env_ids[(env_ids >= self.stepping_one_bridge_start_idx) & (env_ids < self.stepping_one_bridge_end_idx)]
            climb_env_ids = env_ids[(env_ids >= self.climb_start_idx) & (env_ids < self.climb_end_idx)]
            crawl_env_ids = env_ids[(env_ids >= self.crawl_start_idx) & (env_ids < self.crawl_end_idx)]
            stool_env_ids = env_ids[(env_ids >= self.stool_start_idx) & (env_ids < self.stool_end_idx)]
            x_only_env_ids = torch.concatenate((gap_env_ids, continuous_gap_env_ids, bream_env_ids, stepping_one_bridge_env_ids, tilt_env_ids))
            xy_random_env_ids = torch.concatenate((stairs_cliff_env_ids, slope_cliff_env_ids, stepping_stones_env_ids, climb_env_ids, crawl_env_ids))

            self.root_states[x_only_env_ids] = self.base_init_state
            self.root_states[x_only_env_ids, :3] += self.env_origins[x_only_env_ids]
            self.root_states[x_only_env_ids, :1] += torch_rand_float(-0.5, 0.5, (len(x_only_env_ids), 1), device=self.device)
            self.root_states[x_only_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(x_only_env_ids), 1), device=self.device)

            self.root_states[xy_random_env_ids] = self.base_init_state
            self.root_states[xy_random_env_ids, :3] += self.env_origins[xy_random_env_ids]
            self.root_states[xy_random_env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(xy_random_env_ids), 2), device=self.device)

            self.root_states[stool_env_ids] = self.base_init_state
            self.root_states[stool_env_ids, :3] += self.env_origins[stool_env_ids]
            self.root_states[stool_env_ids, :1] += torch_rand_float(-0.25, 0.25, (len(stool_env_ids), 1), device=self.device)

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _accumulate_command_tracking_score(self, group_name, env_ids):
        if len(env_ids) == 0:
            return None
        self.command_curriculum_reward_sums[group_name] += torch.sum(self.episode_sums["tracking_lin_vel"][env_ids])
        self.command_curriculum_episode_counts[group_name] += len(env_ids)
        if self.command_curriculum_episode_counts[group_name] < self.command_curriculum_group_sizes[group_name]:
            return None
        score = self.command_curriculum_reward_sums[group_name] / (self.command_curriculum_episode_counts[group_name] * self.max_episode_length * self.reward_scales["tracking_lin_vel"])
        self.command_curriculum_reward_sums[group_name].zero_()
        self.command_curriculum_episode_counts[group_name] = 0
        return score

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # 排除初始化时没有实际运行步数的 reset。
        env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(env_ids) == 0:
            return

        omni_env_ids = env_ids[env_ids < self.rough_slope_end_idx]
        rough_flat_env_ids = env_ids[(env_ids >= self.rough_flat_start_idx) & (env_ids < self.rough_flat_end_idx)]
        stairs_discrete_env_ids = env_ids[(env_ids >= self.stairs_up_start_idx) & (env_ids < self.discrete_end_idx)]
        slope_cliff_env_ids = env_ids[(env_ids >= self.slope_cliff_start_idx) & (env_ids < self.slope_cliff_end_idx)]
        forward_env_ids = env_ids[(env_ids >= self.discrete_end_idx) & (env_ids < self.rough_flat_start_idx) & ~((env_ids >= self.slope_cliff_start_idx) & (env_ids < self.slope_cliff_end_idx))]

        # 各组分数超过 0.8 时独立提升速度上限。
        score = self._accumulate_command_tracking_score("omni", omni_env_ids)
        if score is not None and score > 0.8:
            self.omni_command_x = min(self.omni_command_x + 0.2, 1.5)
            self.omni_command_y = min(self.omni_command_y + 0.2, 1.0)

        score = self._accumulate_command_tracking_score("rough_flat", rough_flat_env_ids)
        if score is not None and score > 0.8:
            self.rough_flat_command_x = min(self.rough_flat_command_x + 0.2, 2.0)
            self.rough_flat_command_y = min(self.rough_flat_command_y + 0.2, 1.0)

        score = self._accumulate_command_tracking_score("stairs_discrete", stairs_discrete_env_ids)
        if score is not None and score > 0.8:
            self.stairs_discrete_command_x = min(self.stairs_discrete_command_x + 0.2, 1.0)
            self.stairs_discrete_command_y = min(self.stairs_discrete_command_y + 0.2, 0.8)

        score = self._accumulate_command_tracking_score("forward", forward_env_ids)
        if score is not None and score > 0.8:
            self.forward_command_x = min(self.forward_command_x + 0.2, 1.0)

        score = self._accumulate_command_tracking_score("slope_cliff", slope_cliff_env_ids)
        if score is not None and score > 0.8:
            self.slope_cliff_command_x = min(self.slope_cliff_command_x + 0.2, 1.5)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_start_dim = self.privileged_dim - 3 # last 3-dim is the linear vel
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[noise_start_dim:noise_start_dim+3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[noise_start_dim+3:noise_start_dim+6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[noise_start_dim+6:noise_start_dim+9] = noise_scales.gravity * noise_level
        noise_vec[noise_start_dim+9:noise_start_dim+12] = 0. # commands
        noise_vec[noise_start_dim+12:noise_start_dim+24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[noise_start_dim+24:noise_start_dim+36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[noise_start_dim+36:noise_start_dim+48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[noise_start_dim+48:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.rigid_body_lin_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[...,7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.contact_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vel_violate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros_like(self.last_actions)
        # for latency
        self.latency_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.direct_yaw_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.forward_height_points = self._init_forward_height_points()
        self.measured_heights = 0
        self.measured_forward_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # stand pose joint targets
        self.stand_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.stand_joint_angles[name]
            self.stand_dof_pos[i] = angle
        self.stand_dof_pos = self.stand_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
            if getattr(self.cfg.depth, "backend", "isaac") == "warp":
                from legged_gym.sensors.warp_depth_camera import WarpDepthCamera

                self.warp_depth_camera = WarpDepthCamera(
                    terrain=self.terrain,
                    depth_cfg=self.cfg.depth,
                    local_positions=self.camera_local_positions,
                    local_quats=self.camera_local_quats,
                    device=self.device,
                )

    def compute_randomized_gains(self, num_envs):
        p_mult = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (num_envs, self.num_actions), device=self.device)
        d_mult = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (num_envs, self.num_actions), device=self.device)
        return p_mult * self.p_gains, d_mult * self.d_gains

    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position) + np.random.uniform(-config.position_noise, config.position_noise, 3)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])
            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle), np.radians(camera_y_angle), np.radians(camera_z_angle))
            self.camera_local_positions.append(camera_position)
            self.camera_local_quats.append([
                local_transform.r.x,
                local_transform.r.y,
                local_transform.r.z,
                local_transform.r.w,
            ])
            if getattr(self.cfg.depth, "backend", "isaac") != "isaac":
                return

            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # use the sensor to acquire contact force, may be more accurate
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.camera_local_positions = []
        self.camera_local_quats = []
        #for domain randomization
        self.randomized_frictions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_restitutions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_com_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        if(self.cfg.depth.use_camera):
            # The added meshes for tilt, stool, and crawl are absent from the forward height map and require simulated cameras.
            # Allocate 80% of the remaining cameras to gap through climb and 20% to other terrains.
            self.cfg.depth.camera_num_envs = min(self.cfg.depth.camera_num_envs, self.num_envs)
            # 必须有相机的环境ID
            mandatory_env_ids = np.arange(self.tilt_start_idx, self.crawl_end_idx, dtype=np.int64)
            if len(mandatory_env_ids) > self.cfg.depth.camera_num_envs:
                raise ValueError("camera_num_envs must cover all tilt/stool/crawl environments: "f"need {len(mandatory_env_ids)}, got {self.cfg.depth.camera_num_envs}")

            # 优先有相机的环境ID
            priority_env_ids = np.arange(self.gap_start_idx, self.climb_end_idx, dtype=np.int64)

            # 其他环境ID
            other_env_ids = np.concatenate((np.arange(0, self.gap_start_idx, dtype=np.int64), np.arange(self.crawl_end_idx, self.num_envs, dtype=np.int64)))

            remaining_camera_count = (self.cfg.depth.camera_num_envs - len(mandatory_env_ids))

            priority_camera_count = min(int(round(0.8 * remaining_camera_count)), len(priority_env_ids))

            other_camera_count = remaining_camera_count - priority_camera_count
            if other_camera_count > len(other_env_ids):
                priority_camera_count += other_camera_count - len(other_env_ids)
                other_camera_count = len(other_env_ids)

            priority_depth_index = np.random.choice(priority_env_ids, priority_camera_count, replace=False)
            other_depth_index = np.random.choice(other_env_ids, other_camera_count, replace=False)
            self.depth_index = np.sort(np.concatenate((mandatory_env_ids, priority_depth_index, other_depth_index))).astype(np.int64)

            # Do not use tilt, stool, or crawl images to supervise the depth predictor.
            self.depth_index_without_tilt_stool_crawl = self.depth_index[(self.depth_index < self.tilt_start_idx) | (self.depth_index >= self.crawl_end_idx)]
            self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int64)
            for i in range(len(self.depth_index)):
                self.depth_index_inverse[self.depth_index[i]] = i

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            if(self.cfg.depth.use_camera and i in self.depth_index):
                self.attach_camera(i, env_handle, anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_forward_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_forward_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_forward_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_forward_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_forward_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_forward_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_forward_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_forward_height_points), self.forward_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_forward_height_points), self.forward_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_forward_map(self):
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_forward_heights, -1, 1) * self.obs_scales.height_measurements

    #------------ reward functions----------------
    #------------ task reward----------------
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_lin_vel(self):
    #     # clipping tracking reward
    #     lin_vel = self.base_lin_vel[:, :2].clone()
    #     lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
    #     lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
    #     clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - clip_lin_vel), dim=1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        # 仅对指定障碍地形裁剪速度，其余地形使用原始速度误差。
        lin_vel = self.base_lin_vel[:, :2].clone()
        for start_idx, end_idx in (
            (self.gap_start_idx, self.gap_end_idx),
            (self.bream_start_idx, self.bream_end_idx),
            (self.continuous_gap_start_idx, self.continuous_gap_end_idx),
            (self.climb_start_idx, self.climb_end_idx),
        ):
            commands = self.commands[start_idx:end_idx, :2]
            lin_vel_upper_bound = torch.where(commands < 0, 1e5, commands + self.cfg.rewards.lin_vel_clip)
            lin_vel_lower_bound = torch.where(commands > 0, -1e5, commands - self.cfg.rewards.lin_vel_clip)
            lin_vel[start_idx:end_idx] = torch.clip(lin_vel[start_idx:end_idx], lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - lin_vel), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    #------------ style reward----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:,[0, 3, 6, 9]])), dim=-1)

    def _reward_has_contact(self):
        contact_filt = 1.*self.contact_filt
        return(torch.norm(self.commands[:, :2], dim=1) < 0.1)*torch.sum(contact_filt,dim=-1)/4 

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_stand_pos(self):
        # Penalize dof positions that are far away from the stand position
        return torch.sum(torch.abs(self.dof_pos - self.stand_dof_pos), dim=1)

    #------------ safe reward----------------
    def _reward_power(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_power_distribution(self):
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_distribution(self):
        # Penalize torques
        return torch.var(torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        return torch.sum(torch.square(self.last_last_actions - 2*self.last_actions + self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    #------------ feet reward----------------
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).round().long()
        feet_pos_xy[..., 0].clamp_(0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1].clamp_(0, self.x_edge_mask.shape[1] - 1)

        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = torch.sum(self.feet_at_edge, dim=-1).float()

        edge_reward = torch.zeros_like(rew)
        edge_reward[self.gap_start_idx:self.gap_end_idx] = rew[self.gap_start_idx:self.gap_end_idx]
        edge_reward[self.climb_start_idx:self.climb_end_idx] = rew[self.climb_start_idx:self.climb_end_idx]
        return edge_reward

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 4*torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    #------------ other reward----------------
    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[self.gap_start_idx:self.crawl_end_idx, 1], forward[self.gap_start_idx:self.crawl_end_idx, 0])
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[self.gap_start_idx:self.crawl_end_idx] = cheat
        return cheat_penalty

    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)

    def _reward_stool_base_height(self):
        """Penalize a base height above 0.45 m only in stool environments."""
        penalty = torch.zeros(self.num_envs, device=self.device)
        stool_slice = slice(self.stool_start_idx, self.stool_end_idx)
        relative_base_height = (self.root_states[stool_slice, 2] - self.env_origins[stool_slice, 2])
        penalty[stool_slice] = (relative_base_height > 0.45).float()
        return penalty
