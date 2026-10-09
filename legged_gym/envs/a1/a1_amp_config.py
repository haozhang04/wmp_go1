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

import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('datasets/mocap_motions/*')
# MOTION_FILES = glob.glob('datasets/go2_motion/*')
MOTION_FILES = [
    # 'datasets/mocap_motions_collected/bound/forward_0.6mps.txt',
    # 'datasets/mocap_motions_collected/bound/forward_1.0mps.txt',
    # 'datasets/mocap_motions_collected/hop/forward_0.6mps.txt',
    'datasets/mocap_motions/trot1.txt',
    'datasets/mocap_motions/trot2.txt',
    'datasets/go2_motion/go2_backward_0.4mps.txt',
    'datasets/go2_motion/go2_backward_0.7mps.txt',
    'datasets/go2_motion/go2_left_0.5mps.txt',
    'datasets/go2_motion/go2_right_0.5mps.txt',
    'datasets/go2_motion/go2_stance_0.0mps.txt',
    'datasets/go2_motion/go2_turn_left_0.5radps.txt',
    'datasets/go2_motion/go2_turn_right_0.5radps.txt',
]


class A1AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4100 # 1500 2048 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 33 # proprioception
        action_dim = 12
        privileged_dim = 24 + 26 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        forward_height_dim = 525 # for depth image prediction
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        reference_state_initialization = False
        amp_motion_files = MOTION_FILES

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        terrain_proportions = [
            0.0,   # wave
            0.1,   # rough_slope
            0.1,   # stairs_up
            0.1,   # stairs_down
            0.05,  # discrete
            0.1,   # gap
            0.05,  # continuous_gap
            0.05,  # bream
            0.1,   # stairs_cliff
            0.05,  # slope_cliff
            0.0,   # stepping_stones
            0.05,  # stepping_one_bridge
            0.1,   # climb
            0.0,   # tilt (camera required)
            0.05,  # stool (camera required)
            0.05,  # crawl (camera required)
            0.05,  # rough_flat
        ]
        # trimesh only:
        slope_threshold = 0.75  # Slopes above this threshold are corrected to vertical surfaces.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

        stand_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.55,     # [rad]
            'RL_thigh_joint': 0.55,   # [rad]
            'FR_thigh_joint': 0.55,     # [rad]
            'RR_thigh_joint': 0.55,   # [rad]

            'FL_calf_joint': -1.38,   # [rad]
            'RL_calf_joint': -1.38,    # [rad]
            'FR_calf_joint': -1.38,  # [rad]
            'RR_calf_joint': -1.38,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}  # [N*m*s/rad]
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class depth:
        use_camera = True
        backend = "warp"  # "isaac" or "warp"
        camera_num_envs = 1100 # 300 512 1024

        position = [0.28, 0, 0.12]  # front camera
        position_noise = 0.01  # XYZ position randomization: +/- 1cm
        y_angle = [15, 25]  # positive pitch down
        z_angle = [-2, 2]  # yaw randomization [deg]
        x_angle = [-2, 2]  # roll randomization [deg]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.25
        foot_height_target = 0.15
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        lin_vel_clip = 0.1

        class scales(LeggedRobotCfg.rewards.scales):
            # task
            # termination = -200
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            # safety
            torques = -0.0001
            dof_acc = -2.5e-7
            action_rate = -0.03
            smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -10.0
            dof_vel_limits = -10.0

            # style
            lin_vel_z = -1.0
            # orientation = -0.0
            # base_height = -0.0
            # dof_error = -0.04
            # stand_pos = -0.04
            # has_contact = 0.5
            # hip_pos = -0.1

            # feet
            feet_air_time = 1.0
            feet_stumble = -1.0
            feet_edge = -1.0

            # other
            cheat = -1.0
            stuck = -1.0
            stool_base_height = -1.0


    class commands:
        curriculum = True
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        limit_vel_prob = 0.0
        zero_command_prob = 0.05
        limit_vel = [-1.0, 0.0, 1.0]
        direct_yaw_values = [-1.5, 0.0, 1.5]

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]


class A1AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'a1_wmp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 100000  # number of policy updates
        save_interval = 1000

        amp_reward_coef = 0.5 * 0.02  # set to 0 means not use amp reward
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 1
        training_iters = 50
        batch_size = 1024
        loss_scale = 100
