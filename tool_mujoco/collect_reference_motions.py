#!/usr/bin/env python3
#  (c) 2026-2027 zh
"""Collect flat, hopping, and bounding Go1 reference motions."""

# 前置依赖：
# 1. 克隆 LeggedSkillDeploy：
#    git clone https://github.com/haozhang04/LeggedSkillDeploy.git
# 2. 按照 LeggedSkillDeploy 的 README 安装依赖。
# 3. 将下方 LEGGEDSKILL_ROOT 修改为实际的 LeggedSkillDeploy 目录。
#
# 用法：
# 1. 使用配置好的策略批量采集参考动作：
#    python tool_mujoco/collect_reference_motions.py
# 2. 只采集指定动作类型或指令：
#    python tool_mujoco/collect_reference_motions.py --motion-type hop --motion-name forward
# 3. 覆盖前进速度：
#    python tool_mujoco/collect_reference_motions.py --motion-name forward --forward-speed 1.0
#
# 说明：
# - 脚本会在内存中创建无障碍纯平地 MuJoCo 环境
# - Flat、Hop、Bound 分别使用脚本顶部指定的策略模型
# - 每条轨迹开始前都会重置环境，并固定使用指定策略模型
# - 输出目录是 datasets/mocap_motions_collected 下的 flat、hop 和 bound
# - 每类生成前后、左右、顺时针和逆时针旋转共 6 个文件
# - 输出格式与项目 AMPLoader 使用的 61 维动作格式一致

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf


ROOT_DIR = Path(__file__).resolve().parents[1]
LEGGEDSKILL_ROOT = Path("/home/zh/moe_actor_mutil_critic/deploy/leggedskill")
ROBOT_XML = LEGGEDSKILL_ROOT / "robot_description/mjcf/go1/go1.xml"
OUTPUT_ROOT = ROOT_DIR / "datasets/mocap_motions_collected"

CONTROL_DT = 0.002
FRAME_DURATION = 0.02
WARMUP_SECONDS = 2.0
RECORD_SECONDS = 10.0

FLAT_POLICY_DIR = "issacgym/go1/moe"
FLAT_MODEL_NAME = "moe_best.pt"
HOP_POLICY_DIR = "issacgym/go1/go1"
HOP_MODEL_NAME = "Mar24_22-05-56_hop.pt"
BOUND_POLICY_DIR = "issacgym/go1/go1"
BOUND_MODEL_NAME = "Mar25_19-19-16_bound.pt"

# x velocity, y velocity, yaw velocity (m/s, m/s, rad/s)
COMMANDS = {
    "forward": (0.6, 0.0, 0.0),
    "backward": (-0.6, 0.0, 0.0),
    "left": (0.0, 0.4, 0.0),
    "right": (0.0, -0.4, 0.0),
    "turn_ccw": (0.0, 0.0, 0.8),
    "turn_cw": (0.0, 0.0, -0.8),
}

LEG_ORDER = ("FL", "FR", "RL", "RR")  # Isaac Gym/AMP ordering.
FOOT_OFFSET = np.array([0.0, 0.0, -0.213], dtype=np.float64)


def import_leggedskill():
    if not LEGGEDSKILL_ROOT.is_dir():
        raise FileNotFoundError(f"LeggedSkill repository not found: {LEGGEDSKILL_ROOT}")
    sys.path.insert(0, str(LEGGEDSKILL_ROOT))
    from src.interface.IOMuJoCo import IOMuJoCo
    from src.scripts.rl_data import STATE
    from src.scripts.rl_deploy import RLDeploy

    return IOMuJoCo, RLDeploy, STATE


def load_model():
    # Build a temporary obstacle-free world in memory. The external file named
    # empty_world.xml contains platforms, stairs, and slopes, so it must not be
    # used when collecting flat-ground reference motions.
    terrain_mjcf = mjcf.RootElement(model="flat_world")
    terrain_mjcf.option.timestep = CONTROL_DT
    terrain_mjcf.worldbody.add(
        "geom",
        name="flat_ground",
        type="plane",
        size=(0.0, 0.0, 0.05),
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.2, 0.3, 0.4, 1.0),
    )
    robot_mjcf = mjcf.from_path(str(ROBOT_XML))

    for joint in robot_mjcf.find_all("joint"):
        if joint.tag == "freejoint" or getattr(joint, "type", None) == "free":
            joint.remove()

    attachment_frame = terrain_mjcf.worldbody.attach(robot_mjcf)
    attachment_frame.add("freejoint", name="root")

    physics = mjcf.Physics.from_mjcf_model(terrain_mjcf)
    model, data = physics.model.ptr, physics.data.ptr
    model.opt.timestep = CONTROL_DT
    return physics, model, data, robot_mjcf.model


def find_model_index(deploy, model_name):
    matches = [index for index, (_, _, label) in enumerate(deploy.model_entries) if Path(label).name == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one model named {model_name!r}, found {len(matches)} in {deploy.model_names}")
    return matches[0]


def reset_environment(model, data, deploy):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    data.qpos[3] = 1.0

    actuator_qpos_addresses = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    mapping = deploy.params.joint_mapping
    defaults = deploy.params.default_dof_pos[0].detach().cpu().numpy()
    for policy_index, simulation_index in enumerate(mapping):
        data.qpos[actuator_qpos_addresses[simulation_index]] = defaults[policy_index]

    mujoco.mj_forward(model, data)
    deploy.inference_counter = 0
    deploy.policy.rl_time = 0.0
    return actuator_qpos_addresses


def body_id(model, robot_prefix, name):
    full_name = f"{robot_prefix}/{name}"
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_name)
    if index < 0:
        raise ValueError(f"Body not found in MuJoCo model: {full_name}")
    return index


def collect_raw_sample(model, data, base_id, foot_body_ids, joint_qpos_addresses, joint_mapping):
    root_pos = data.xpos[base_id].copy()
    root_quat_wxyz = data.xquat[base_id].copy()
    root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
    root_rotation = data.xmat[base_id].reshape(3, 3).copy()

    joint_pos_sim = np.array([data.qpos[address] for address in joint_qpos_addresses])
    joint_vel_sim = np.array([
        data.qvel[int(model.jnt_dofadr[int(model.actuator_trnid[index, 0])])]
        for index in range(model.nu)
    ])
    joint_pos = joint_pos_sim[joint_mapping]
    joint_vel = joint_vel_sim[joint_mapping]

    toe_pos_local = []
    for foot_id in foot_body_ids:
        foot_rotation = data.xmat[foot_id].reshape(3, 3)
        toe_world = data.xpos[foot_id] + foot_rotation @ FOOT_OFFSET
        toe_pos_local.extend(root_rotation.T @ (toe_world - root_pos))

    base_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, base_velocity, 0
    )
    angular_vel_local = root_rotation.T @ base_velocity[:3]
    linear_vel_local = root_rotation.T @ base_velocity[3:]

    return {
        "root_pos": root_pos,
        "root_quat": root_quat_xyzw,
        "joint_pos": joint_pos,
        "toe_pos": np.asarray(toe_pos_local),
        "linear_vel": linear_vel_local,
        "angular_vel": angular_vel_local,
        "joint_vel": joint_vel,
    }


def build_amp_frames(samples):
    toe_positions = np.stack([sample["toe_pos"] for sample in samples])
    toe_velocities = np.gradient(toe_positions, FRAME_DURATION, axis=0, edge_order=1)
    frames = []
    for sample, toe_velocity in zip(samples, toe_velocities):
        frame = np.concatenate((
            sample["root_pos"],
            sample["root_quat"],
            sample["joint_pos"],
            sample["toe_pos"],
            sample["linear_vel"],
            sample["angular_vel"],
            sample["joint_vel"],
            toe_velocity,
        ))
        if frame.shape != (61,):
            raise RuntimeError(f"Expected a 61-value AMP frame, got {frame.shape}")
        frames.append(frame.tolist())
    return frames


def save_motion(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    motion = {
        "LoopMode": "Wrap",
        "FrameDuration": FRAME_DURATION,
        "MotionWeight": 1.0,
        "Frames": frames,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(motion, file, separators=(",", ":"))
        file.write("\n")


def collect_one(policy_dir, model_name, command, output_path, interfaces):
    IOMuJoCo, RLDeploy, STATE = interfaces
    physics, model, data, robot_prefix = load_model()
    deploy = RLDeploy([policy_dir])
    model_index = find_model_index(deploy, model_name)
    deploy.robot_state.control.model_flag = model_index
    deploy.change_model(model_index)
    io = IOMuJoCo(model, data, deploy)
    joint_qpos_addresses = reset_environment(model, data, deploy)
    deploy.running_state = STATE.STATE_RL_RUNNING

    control = deploy.robot_state.control
    control.x, control.y, control.yaw = command
    base_id = body_id(model, robot_prefix, "base_link")
    foot_body_ids = [body_id(model, robot_prefix, f"{leg}_calf") for leg in LEG_ORDER]
    joint_mapping = np.asarray(deploy.params.joint_mapping, dtype=np.int64)
    inference_interval = round(FRAME_DURATION / CONTROL_DT)
    warmup_steps = round(WARMUP_SECONDS / CONTROL_DT)
    record_steps = round(RECORD_SECONDS / CONTROL_DT)
    samples = []

    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        for step in range(warmup_steps + record_steps):
            io.recv()
            deploy.step(inference_interval)
            if deploy.last_model_flag != model_index:
                raise RuntimeError(f"Policy switched away from {model_name}: active index={deploy.last_model_flag}")
            io.send()
            mujoco.mj_step(model, data)

            if step >= warmup_steps and (step - warmup_steps) % inference_interval == 0:
                samples.append(collect_raw_sample(
                    model,
                    data,
                    base_id,
                    foot_body_ids,
                    joint_qpos_addresses,
                    joint_mapping,
                ))

    frames = build_amp_frames(samples)
    save_motion(output_path, frames)
    del physics
    print(f"Saved {len(frames):4d} frames: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-type",
        choices=("all", "flat", "hop", "bound"),
        default="all",
        help="collect only one policy family (default: all)",
    )
    parser.add_argument(
        "--motion-name",
        choices=tuple(COMMANDS),
        help="collect only one command (default: all commands)",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        help="override the forward command speed in m/s",
    )
    args = parser.parse_args()
    interfaces = import_leggedskill()
    jobs = (
        ("flat", FLAT_POLICY_DIR, FLAT_MODEL_NAME),
        ("hop", HOP_POLICY_DIR, HOP_MODEL_NAME),
        ("bound", BOUND_POLICY_DIR, BOUND_MODEL_NAME),
    )
    for folder, policy_dir, model_name in jobs:
        if args.motion_type != "all" and folder != args.motion_type:
            continue
        for motion_name, command in COMMANDS.items():
            if args.motion_name is not None and motion_name != args.motion_name:
                continue
            if motion_name == "forward" and args.forward_speed is not None:
                command = (args.forward_speed, command[1], command[2])
            if motion_name.startswith("turn_"):
                output_name = f"{motion_name}_{abs(command[2]):.1f}radps.txt"
            else:
                linear_speed = max(abs(command[0]), abs(command[1]))
                output_name = f"{motion_name}_{linear_speed:.1f}mps.txt"
            collect_one(
                policy_dir=policy_dir,
                model_name=model_name,
                command=command,
                output_path=OUTPUT_ROOT / folder / output_name,
                interfaces=interfaces,
            )


if __name__ == "__main__":
    main()
