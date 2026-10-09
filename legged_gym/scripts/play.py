# Copyright (c) 2026-2027 zh

"""
内容：
    加载 WMP 策略，在指定地形上运行单环境推理，录制深度视频并导出部署策略。
    地形名称见 legged_gym/terrain_names.py。
用法：
    1. 无 --path，自动选择最新运行目录和 checkpoint：
    CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --headless --terrain stairs_cliff

    2. 传入 checkpoint 文件：
    CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --headless --terrain stairs_cliff \
    --path /path/to/model_10000.pt

    3. 传入运行目录，自动选择最新 checkpoint：
    CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --headless --terrain stairs_cliff \
    --path /path/to/Sep02_19-51-11

    4. 传入实验根目录，自动选择最新运行目录和 checkpoint：
    CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --headless --terrain stairs_cliff \
    --path /path/to/a1_wmp_example
"""

import os
import cv2
import sys
import atexit
import functools
import shutil
import subprocess
import isaacgym  # 初始化 Isaac Gym 运行时
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.terrain_names import TERRAIN_NAMES
from legged_gym.sensors.camera_visualization import draw_d435_camera_boxes, fix_depth_camera_pose
from legged_gym.utils import get_args, resolve_load_path, task_registry
from legged_gym.utils.export_deployment_policy import export_deployment_policy


EXPORT_POLICY = True
RECORD_FRAMES = True
MOVE_CAMERA = True


# ---------- video ----------
def get_camera_tensor(env):
    """获取离屏相机的 GPU 图像张量。"""
    camera_handle = getattr(env, "camera_sensor", None)
    if camera_handle is None or camera_handle < 0:
        raise RuntimeError("Isaac Gym camera sensor was not initialized")
    image_tensor = env.gym.get_camera_image_gpu_tensor(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
    if image_tensor is None:
        raise RuntimeError("Failed to acquire the Isaac Gym camera image tensor")
    return gymtorch.wrap_tensor(image_tensor)

def capture_headless_frame(env, camera_tensor):
    """按原 viewer 相同的相对位置渲染离屏 RGB 帧。"""
    root_position = env.root_states[0, :3].detach().cpu().numpy()
    camera_position = root_position + np.array([0.0, 2.0, 0.0])
    env.gym.set_camera_location(
        env.camera_sensor,
        env.envs[0],
        gymapi.Vec3(*(float(value) for value in camera_position)),
        gymapi.Vec3(*(float(value) for value in root_position)),
    )
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)
    try:
        return camera_tensor[:, :, :3].detach().cpu().numpy().copy()
    finally:
        env.gym.end_access_image_tensors(env.sim)

def finalize_video(recording_dir, video_path, video_fps, recording_state):
    if not os.path.isdir(recording_dir):
        return

    frame_count = recording_state["frame_count"]
    if frame_count == 0:
        shutil.rmtree(recording_dir)
        print('No video was saved because no frames were captured.')
        return

    subprocess.run(
        [
            'ffmpeg', '-y', '-framerate', str(video_fps),
            '-i', os.path.join(recording_dir, '%06d.png'),
            '-framerate', str(video_fps),
            '-i', os.path.join(recording_dir, 'depth_%06d.png'),
            '-filter_complex',
            '[1:v]scale=320:320:flags=neighbor[depth];'
            '[0:v][depth]overlay=W-w-20:20',
            '-frames:v', str(frame_count),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path,
        ],
        check=True,
    )
    shutil.rmtree(recording_dir)
    print('Saved video to:', video_path)


@torch.no_grad()
def play(args):
    # ---------- configuration ----------
    if args.terrain not in TERRAIN_NAMES:
        raise ValueError(f"Unsupported terrain {args.terrain!r}; expected one of {', '.join(TERRAIN_NAMES)}")
    record_frames = RECORD_FRAMES
    move_camera = MOVE_CAMERA and not args.headless
    if record_frames and shutil.which('ffmpeg') is None:
        raise RuntimeError('Video recording requires ffmpeg on PATH.')
    if record_frames and args.headless and (not args.sim_device.startswith('cuda') or not args.rl_device.startswith('cuda')):
        raise RuntimeError('Server video recording requires CUDA sim and RL devices.')

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 15
    env_cfg.terrain.num_cols = 1
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

    env_cfg.terrain.terrain_proportions = [float(name == args.terrain) for name in TERRAIN_NAMES]
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.depth.use_camera = True
    env_cfg.env.enable_camera_sensors = record_frames and args.headless
    if env_cfg.env.enable_camera_sensors:
        env_cfg.env.camera_width = 1280
        env_cfg.env.camera_height = 720
        env_cfg.env.camera_horizontal_fov = 75.0
    fix_depth_camera_pose(env_cfg.depth)

    # ---------- environment and policy initialization ----------
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.terrain_levels[:] = 8
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    obs = env.get_observations()
    policy = ppo_runner.get_inference_policy(device=env.device)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    load_path = resolve_load_path(log_root, args=args, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    loaded_run_dir = os.path.dirname(load_path)
    checkpoint_name = os.path.splitext(os.path.basename(load_path))[0]
    checkpoint_iteration = checkpoint_name[len('model_'):] if checkpoint_name.startswith('model_') else checkpoint_name

    if record_frames:
        videos_root = os.path.join(loaded_run_dir, f'videos_{checkpoint_iteration}')
        recording_dir = os.path.join(videos_root, f'.{args.terrain}_frames')
        video_path = os.path.join(videos_root, f'{args.terrain}.mp4')
        if os.path.isdir(recording_dir):
            shutil.rmtree(recording_dir)
        os.makedirs(recording_dir, exist_ok=True)
        print('Recording video to:', video_path)
        recording_state = {"frame_count": 0}
        video_fps = max(1, round(1.0 / (2.0 * env.dt)))
        finalize_callback = functools.partial(finalize_video, recording_dir, video_path, video_fps, recording_state)
        atexit.register(finalize_callback)

    # 导出完整的有状态部署策略
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

    if record_frames:
        depth_slot = int(env.depth_index_inverse[0])
        latest_depth_image = torch.zeros(env.cfg.depth.resized, dtype=torch.uint8).numpy()
        if args.headless:
            camera_tensor = get_camera_tensor(env)

    # ---------- inference and recording ----------
    infos = {"depth": None}
    for i in range(int(env.max_episode_length)):
        if infos["depth"] is not None:
            wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)
            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            wm_is_first[:] = 0

        actions = policy(obs.detach(), trajectory_history.flatten(1).detach(), wm_feature.detach())

        try:
            obs, _, _, dones, infos, reset_env_ids, _ = env.step(actions.detach())
        except SystemExit:
            print('Viewer closed; finalizing recorded video.')
            break

        if record_frames and infos["depth"] is not None:
            latest_depth_image = ((infos["depth"][depth_slot] + 0.5).clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()

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

        if move_camera:
            lookat = env.root_states[0, :3]
            camera_position = lookat.detach().cpu().numpy() + [0, 2, 0]
            env.set_camera(camera_position, lookat)
        if not args.headless:
            draw_d435_camera_boxes(env)
        if record_frames and i % 2:
            frame_index = recording_state["frame_count"]
            filename = os.path.join(recording_dir, f"{frame_index:06d}.png")
            if args.headless:
                rgb_frame = capture_headless_frame(env, camera_tensor)
                if not cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f'Failed to save RGB frame: {filename}')
            else:
                try:
                    env.render(sync_frame_time=False)
                except SystemExit:
                    print('Viewer closed; finalizing recorded video.')
                    break
                env.gym.write_viewer_image_to_file(env.viewer, filename)
            depth_filename = os.path.join(recording_dir, f"depth_{frame_index:06d}.png")
            if not cv2.imwrite(depth_filename, latest_depth_image):
                raise RuntimeError(f'Failed to save depth frame: {depth_filename}')
            recording_state["frame_count"] += 1

    # video cleanup
    if record_frames:
        atexit.unregister(finalize_callback)
        finalize_callback()

# ---------- main ----------
if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
    if args.headless and RECORD_FRAMES:
        # Isaac Gym 离屏相机在解释器退出阶段可能触发原生资源二次释放。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
