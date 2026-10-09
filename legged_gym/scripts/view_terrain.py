# Copyright (c) 2026-2027 zh
"""
内容：
    创建完整训练地形网格，加载 WMP 策略并在 viewer 中显示机器人运行效果。

用法：
    CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -v
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import isaacgym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.sensors.camera_visualization import draw_d435_camera_boxes, fix_depth_camera_pose
from legged_gym.utils import get_args, resolve_load_path, task_registry
from legged_gym.utils.export_deployment_policy import export_deployment_policy

EXPORT_POLICY = False


# ---------- configuration ----------
@torch.no_grad()
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 5000
    env_cfg.env.episode_length_s = 500
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]
    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.depth.use_camera = True
    env_cfg.depth.camera_num_envs = min(env_cfg.depth.camera_num_envs, env_cfg.env.num_envs)
    fix_depth_camera_pose(env_cfg.depth)

    # ---------- environment and policy initialization ----------
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    obs = env.get_observations()
    print(f"view terrain grid: {env_cfg.terrain.num_rows} x {env_cfg.terrain.num_cols}")
    print(f"view terrain_proportions: {env_cfg.terrain.terrain_proportions}")
    policy = ppo_runner.get_inference_policy(device=env.device)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    load_path = resolve_load_path(log_root, args=args, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    checkpoint_name = os.path.splitext(os.path.basename(load_path))[0]
    checkpoint_iteration = checkpoint_name[len('model_'):] if checkpoint_name.startswith('model_') else checkpoint_name
    
    # ---------- policy export ----------
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        policy_name = os.path.basename(os.path.dirname(load_path)) + f'_{checkpoint_iteration}.pt'
        exported_path = export_deployment_policy(
            actor_critic=ppo_runner.alg.actor_critic,
            world_model=ppo_runner._world_model,
            path=path,
            filename=policy_name,
            update_interval=env.cfg.depth.update_interval,
        )
        print('Exported end-to-end policy to:', exported_path)

    # ---------- history and world model initialization ----------
    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs - env.privileged_dim - env.height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6], obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_action_history = torch.zeros(size=(env.num_envs, env.cfg.depth.update_interval, env.num_actions), device=env.device)
    # 与训练一致，首次有效深度到达前使用零特征。
    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    # ---------- inference and visualization ----------
    infos = {"depth": None}
    for i in range(int(env.max_episode_length)):
        if infos["depth"] is not None:
            wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)
            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            wm_is_first[:] = 0

        actions = policy(obs.detach(), trajectory_history.flatten(1).detach(), wm_feature.detach())

        draw_d435_camera_boxes(env)
        obs, _, _, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # ---------- world model input ----------
        wm_action_history = torch.concat((wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }
        wm_obs["image"] = torch.zeros((env.num_envs,) + env.cfg.depth.resized + (1,), device=world_model.device)

        reset_env_ids = reset_env_ids.cpu().numpy()
        if len(reset_env_ids) > 0:
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6], obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
