#!/usr/bin/env python3
#  (c) 2026-2027 zh
"""Play selected AMP motion files with MuJoCo."""

# 用法：
# 1. 在 MOTION_PATHS 中填写需要播放的一个或多个动作文件
# 2. 直接启动 MuJoCo 播放器：
#    python tool_mujoco/visualize_mocap_mujoco.py
#
# 说明：
# - 支持原始动作和新采集动作的 txt 文件
# - 文件会按照 MOTION_PATHS 中的顺序播放，播放完毕后循环
# - Space：暂停/继续，R：重播当前文件
# - N：下一个文件，P：上一个文件，Esc：关闭窗口
# - PLAYBACK_SPEED 控制播放倍速
# - KEEP_WORLD_POSITION=False 时会将每条轨迹的起始 XY 移到原点

import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

# Playback settings. Edit these values and run this file directly.
# Add or remove entries to choose the playback list.
MOTION_PATHS = [
    # Original motions:
    # ROOT_DIR / "datasets/mocap_motions/hop1.txt",
    # ROOT_DIR / "datasets/mocap_motions/hop2.txt",
    ROOT_DIR / "datasets/mocap_motions/trot1.txt",
    ROOT_DIR / "datasets/mocap_motions/trot2.txt",

    # Newly collected flat motions:
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/forward_0.6mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/backward_0.6mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/left_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/right_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/turn_ccw_0.8radps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/flat/turn_cw_0.8radps.txt",

    # # Newly collected hopping motions:
    ROOT_DIR / "datasets/mocap_motions_collected/hop/forward_0.6mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/hop/backward_0.6mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/hop/left_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/hop/right_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/hop/turn_ccw_0.8radps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/hop/turn_cw_0.8radps.txt",

    # Newly collected bounding motions:
    ROOT_DIR / "datasets/mocap_motions_collected/bound/forward_0.6mps.txt",
    ROOT_DIR / "datasets/mocap_motions_collected/bound/forward_1.0mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/bound/backward_0.6mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/bound/left_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/bound/right_0.4mps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/bound/turn_ccw_0.8radps.txt",
    # ROOT_DIR / "datasets/mocap_motions_collected/bound/turn_cw_0.8radps.txt",

    # Go2 motions:
    ROOT_DIR / "datasets/go2_motion/go2_backward_0.4mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_backward_0.7mps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_backward_1.0mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_forward_0.4mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_forward_0.8mps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_forward_1.3mps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_left_0.2mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_left_0.5mps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_right_0.2mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_right_0.5mps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_stance_0.0mps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_turn_left_0.2radps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_turn_left_0.5radps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_turn_left_0.8radps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_turn_right_0.2radps.txt",
    ROOT_DIR / "datasets/go2_motion/go2_turn_right_0.5radps.txt",
    # ROOT_DIR / "datasets/go2_motion/go2_turn_right_0.8radps.txt",
]

MODEL_PATH = ROOT_DIR / "resources/robots/go1/xml/mocap_world.xml"
PLAYBACK_SPEED = 1.0
KEEP_WORLD_POSITION = False
PLAYLIST_ONCE = False


def load_motion(path):
    with path.open("r", encoding="utf-8") as file:
        motion = json.load(file)

    frames = np.asarray(motion["Frames"], dtype=np.float64)
    frame_duration = float(motion["FrameDuration"])
    if frames.ndim != 2 or frames.shape[0] == 0 or frames.shape[1] < 19:
        raise ValueError(f"Expected non-empty motion frames with at least 19 values, got {frames.shape}")
    if frame_duration <= 0.0:
        raise ValueError(f"FrameDuration must be positive, got {frame_duration}")
    return frames, frame_duration


def set_pose(model, data, frame, xy_origin):
    # AMP layout: root xyz, root quaternion xyzw, 12 joint positions, ...
    data.qpos[:3] = frame[:3]
    data.qpos[:2] -= xy_origin

    quat_xyzw = frame[3:7]
    quat_norm = np.linalg.norm(quat_xyzw)
    if quat_norm < 1e-8:
        raise ValueError("Motion contains a zero-length root quaternion")
    quat_xyzw = quat_xyzw / quat_norm
    data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]  # MuJoCo expects wxyz.
    data.qpos[7:19] = frame[7:19]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    if PLAYBACK_SPEED <= 0.0:
        raise ValueError(f"PLAYBACK_SPEED must be positive, got {PLAYBACK_SPEED}")

    motion_paths = [path.expanduser().resolve() for path in MOTION_PATHS]
    if not motion_paths:
        raise ValueError("MOTION_PATHS must contain at least one motion file")
    missing_paths = [path for path in motion_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Motion files not found: {missing_paths}")

    model_path = MODEL_PATH.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nq < 19:
        raise ValueError(f"The selected model needs at least 19 qpos values, got {model.nq}")

    state = {"paused": False, "restart": False, "skip": 0}

    def key_callback(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["restart"] = True
        elif keycode in (ord("N"), ord("n")):
            state["skip"] = 1
        elif keycode in (ord("P"), ord("p")):
            state["skip"] = -1

    print(f"Found {len(motion_paths)} motion files")
    print("Controls: Space = pause/resume, R = restart, N/P = next/previous, Esc = close")

    motion_index = 0
    frame_index = 0

    def select_motion(index):
        selected_frames, selected_duration = load_motion(motion_paths[index])
        selected_origin = np.zeros(2) if KEEP_WORLD_POSITION else selected_frames[0, :2].copy()
        return selected_frames, selected_duration, selected_origin

    frames, frame_duration, xy_origin = select_motion(motion_index)
    set_pose(model, data, frames[0], xy_origin)

    def print_current_motion():
        relative_path = motion_paths[motion_index].relative_to(ROOT_DIR)
        print(
            f"[{motion_index + 1}/{len(motion_paths)}] Playing {relative_path}: "
            f"{len(frames)} frames, {frame_duration:.4f} s/frame, {PLAYBACK_SPEED:g}x"
        )

    print_current_motion()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_body_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20.0

        next_frame_time = time.monotonic()
        while viewer.is_running():
            if state["skip"]:
                motion_index = (motion_index + state["skip"]) % len(motion_paths)
                state["skip"] = 0
                frames, frame_duration, xy_origin = select_motion(motion_index)
                frame_index = 0
                next_frame_time = time.monotonic()
                print_current_motion()

            if state["restart"]:
                frame_index = 0
                state["restart"] = False
                next_frame_time = time.monotonic()

            if state["paused"]:
                viewer.sync()
                next_frame_time = time.monotonic()
                time.sleep(0.01)
                continue

            set_pose(model, data, frames[frame_index], xy_origin)
            viewer.sync()

            frame_index += 1
            if frame_index >= len(frames):
                if PLAYLIST_ONCE and motion_index == len(motion_paths) - 1:
                    break
                motion_index = (motion_index + 1) % len(motion_paths)
                frames, frame_duration, xy_origin = select_motion(motion_index)
                frame_index = 0
                print_current_motion()

            next_frame_time += frame_duration / PLAYBACK_SPEED
            time.sleep(max(0.0, next_frame_time - time.monotonic()))


if __name__ == "__main__":
    main()
