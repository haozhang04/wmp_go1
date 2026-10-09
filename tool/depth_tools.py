# Copyright (c) 2026-2027 zh
"""
内容：
    分模块生成墙面深度、比较后端误差并绘制科研图。
    原始深度以米保存，预览 PNG 不参与浮点深度的误差计算。
用法：
    python tool/depth_tools.py all --backend isaac warp mujoco --out /tmp/depth_run
    python tool/depth_tools.py generate --backend warp mujoco --out /tmp/depth_run
    python tool/depth_tools.py compare --root /tmp/depth_run
    python tool/depth_tools.py plot --root /tmp/depth_run
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

# 原生后端和项目模块的导入准备必须先于第三方库导入。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Isaac Gym 必须在 torch 之前导入。
from isaacgym import gymapi

import cv2
import matplotlib
import mujoco
import numpy as np
import torch
import warp as wp

# 必须在导入 pyplot 前选择无头绘图后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from legged_gym.sensors.warp_depth_camera import WarpDepthCamera

BACKENDS = ("isaac", "warp", "mujoco")
BASE_POSITION = np.array([0.0, 0.0, 0.35], dtype=np.float32)
CAMERA_OFFSET = np.array([0.28, 0.0, 0.15], dtype=np.float32)
CAMERA_PITCH = math.radians(15.0)
YAWS = (-30.0, 0.0, 30.0)


@dataclass
class CameraSpec:
    width: int = 64
    height: int = 64
    horizontal_fov_deg: float = 58.0
    near_clip: float = 0.0
    far_clip: float = 2.0

    def validate(self):
        values = (self.horizontal_fov_deg, self.near_clip, self.far_clip)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Camera parameters must be finite")
        if self.width <= 0 or self.height <= 0 or not 0 < self.horizontal_fov_deg < 180:
            raise ValueError("Image dimensions must be positive and FOV must be in (0, 180)")
        if not 0 <= self.near_clip < self.far_clip:
            raise ValueError("Expected 0 <= near_clip < far_clip")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def frame_name(yaw):
    return f"wall_yaw_{yaw:+.0f}".replace("+", "p").replace("-", "m")


def frame_key(name):
    match = re.fullmatch(r"wall_yaw_([pm])(\d+)", name)
    if match:
        return (0, int(match[2]) * (-1 if match[1] == "m" else 1))
    return (1, name)


# ---------- generate ----------

def camera_pose(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    body_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    c, s = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    pitch_rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return BASE_POSITION + body_rot @ CAMERA_OFFSET, body_rot @ pitch_rot


def clip_depth(depth, spec):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (spec.height, spec.width):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    # 非命中射线按最远距离处理；有效近处表面则夹到近裁剪距离。
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, spec.far_clip)
    return np.clip(depth, spec.near_clip, spec.far_clip)


def scene_mesh():
    # 墙体前表面 x=1 m，厚 0.1 m，宽 4 m，高 0.3 m；地面与其他后端一致。
    vertices = np.array([
        [-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0],
        [1, -2, 0], [1, 2, 0], [1, 2, 0.3], [1, -2, 0.3],
        [1.1, -2, 0], [1.1, 2, 0], [1.1, 2, 0.3], [1.1, -2, 0.3],
    ], dtype=np.float32)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 10, 9], [8, 11, 10],
        [4, 8, 9], [4, 9, 5], [7, 6, 10], [7, 10, 11], [4, 7, 11], [4, 11, 8],
        [5, 9, 10], [5, 10, 6],
    ], dtype=np.int32)
    return vertices, triangles


def render_warp(spec, device):
    vertices, triangles = scene_mesh()
    terrain = SimpleNamespace(vertices=vertices, triangles=triangles, cfg=SimpleNamespace(border_size=0.0))
    config = SimpleNamespace(resized=(spec.height, spec.width), horizontal_fov=spec.horizontal_fov_deg, near_clip=spec.near_clip, far_clip=spec.far_clip)
    quat = [0.0, math.sin(CAMERA_PITCH / 2), 0.0, math.cos(CAMERA_PITCH / 2)]
    # 相机实现中部分数组依赖 Warp 默认设备，须与显式设备保持一致。
    with wp.ScopedDevice(device):
        camera = WarpDepthCamera(terrain, config, local_positions=[CAMERA_OFFSET], local_quats=[quat], device=device)
    positions = torch.tensor(BASE_POSITION[None], device=device)
    results = {}
    for yaw in YAWS:
        angle = math.radians(yaw) / 2
        quats = torch.tensor([[0.0, 0.0, math.sin(angle), math.cos(angle)]], device=device)
        normalized = camera.render(positions, quats)[0].detach().cpu().numpy()
        results[frame_name(yaw)] = (normalized + 0.5) * (spec.far_clip - spec.near_clip) + spec.near_clip
    return results


def render_mujoco(spec):
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(spec.horizontal_fov_deg) / 2) * spec.height / spec.width))
    # 固定 extent=1，使 MuJoCo 的裁剪参数直接对应米；精确设置相机坐标系。
    xml = f"""<mujoco model="wall_depth">
      <statistic extent="1"/>
      <visual><global offwidth="{spec.width}" offheight="{spec.height}"/>
        <map znear="{max(spec.near_clip, 0.0001)}" zfar="{spec.far_clip}"/></visual>
      <worldbody>
        <geom type="box" pos="0 0 -0.05" size="4 4 0.05"/>
        <geom type="box" pos="1.05 0 0.15" size="0.05 2 0.15"/>
        <camera name="depth" fovy="{fovy}"/>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=spec.height, width=spec.width)
    results = {}
    try:
        renderer.enable_depth_rendering()
        for yaw in YAWS:
            position, rotation = camera_pose(yaw)
            # 机器人相机：X 前、Y 左、Z 上；MuJoCo：X 右、Y 上、-Z 前。
            optical = np.column_stack((-rotation[:, 1], rotation[:, 2], -rotation[:, 0]))
            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, optical.astype(np.float64).ravel())
            model.cam_pos[0] = position
            model.cam_quat[0] = quat
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=0)
            results[frame_name(yaw)] = np.asarray(renderer.render(), dtype=np.float32).copy()
    finally:
        renderer.close()
    return results


def render_isaac(spec, device):
    gym = gymapi.acquire_gym()
    device_id = int(device.split(":")[1])
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    sim = gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    try:
        env = gym.create_env(sim, gymapi.Vec3(-4, -4, -1), gymapi.Vec3(4, 4, 2), 1)
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        for name, size, position in (("floor", (8, 8, 0.1), (0, 0, -0.05)), ("wall", (0.1, 4, 0.3), (1.05, 0, 0.15))):
            asset = gym.create_box(sim, *size, options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*position)
            gym.create_actor(env, asset, pose, name, 0, 0)
        props = gymapi.CameraProperties()
        props.width, props.height = spec.width, spec.height
        props.horizontal_fov = spec.horizontal_fov_deg
        props.near_plane = max(spec.near_clip, 0.0001)
        props.far_plane = spec.far_clip
        # 离线测试使用 CPU 读回，避免多 GPU 环境中的 CUDA/图形外部内存映射。
        props.enable_tensors = False
        camera = gym.create_camera_sensor(env, props)
        if camera < 0:
            raise RuntimeError("Failed to create Isaac Gym camera")
        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        results = {}
        for yaw in YAWS:
            position, rotation = camera_pose(yaw)
            target = position + rotation[:, 0]
            gym.set_camera_location(camera, env, gymapi.Vec3(*position.tolist()), gymapi.Vec3(*target.tolist()))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            depth = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_DEPTH)
            if depth is None:
                raise RuntimeError("Isaac Gym returned no depth image")
            results[frame_name(yaw)] = -np.asarray(depth, dtype=np.float32).reshape(spec.height, spec.width).copy()
        return results
    finally:
        gym.destroy_sim(sim)


def generate_worker(args):
    config = json.loads((args.root / "summary.json").read_text())
    spec = CameraSpec(**config["camera"])
    spec.validate()
    backend = args.backend
    if backend == "warp":
        results = render_warp(spec, args.device)
    elif backend == "mujoco":
        results = render_mujoco(spec)
    else:
        results = render_isaac(spec, args.device)
    directory = args.root / backend
    directory.mkdir()
    for name, depth in results.items():
        depth = clip_depth(depth, spec)
        np.save(directory / f"{name}.npy", depth)
        preview = np.rint((depth - spec.near_clip) / (spec.far_clip - spec.near_clip) * 255).astype(np.uint8)
        if not cv2.imwrite(str(directory / f"{name}.png"), preview):
            raise RuntimeError(f"Failed to write depth preview: {name}")
    print(f"[{backend}] generated {len(results)} frames", flush=True)


def generate(args):
    spec = CameraSpec(args.width, args.height, args.horizontal_fov, args.near_clip, args.far_clip)
    spec.validate()
    if not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("Generation requires an explicit CUDA device, for example cuda:1")
    root = args.out.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "depth_encoding": "meters", "camera": asdict(spec),
        "scene": {"wall_front_x": 1.0, "wall_thickness": 0.1, "wall_width": 4.0, "wall_height": 0.3, "floor_size": [8, 8]},
        "base_position": BASE_POSITION.tolist(), "camera_offset": CAMERA_OFFSET.tolist(),
        "camera_pitch_deg": 15.0, "frames": [{"name": frame_name(yaw), "yaw_deg": yaw} for yaw in YAWS],
        "device": args.device, "backends": {},
    }
    save_json(root / "summary.json", metadata)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    failures = []
    for backend in args.backend:
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--root", str(root), "--backend", backend, "--device", args.device]
        with (root / f"{backend}.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if result.returncode == 0 else "failed"
        metadata["backends"][backend] = {"status": status, "returncode": result.returncode, "log": f"{backend}.log"}
        save_json(root / "summary.json", metadata)
        print(f"[{backend}] {status}; log: {root / (backend + '.log')}", flush=True)
        if result.returncode:
            failures.append(backend)
    if failures:
        raise RuntimeError(f"Generation failed for {', '.join(failures)}; inspect backend logs in {root}")
    return root


# ---------- compare ----------

def read_depth(path, encoding, spec):
    if path.suffix == ".npy":
        if encoding not in ("meters", "normalized"):
            raise ValueError("NPY units are unknown; provide --encoding meters or normalized")
        depth = np.load(path, allow_pickle=False).astype(np.float64)
        if encoding == "normalized":
            if not np.isfinite(depth).all() or np.any((depth < -0.50001) | (depth > 0.50001)):
                raise ValueError(f"Normalized depth must be in [-0.5, 0.5]: {path}")
            depth = (depth + 0.5) * (spec.far_clip - spec.near_clip) + spec.near_clip
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f"Expected an 8-bit linear grayscale PNG: {path}")
        depth = image.astype(np.float64) / 255 * (spec.far_clip - spec.near_clip) + spec.near_clip
    if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError(f"Expected a finite, nonnegative 2D depth matrix: {path}")
    return depth


def load_depths(args):
    root = args.root
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    summary = root / "summary.json"
    metadata = json.loads(summary.read_text()) if summary.exists() else {}
    camera = metadata.get("camera", {})
    spec = CameraSpec(**{key: value for key, value in camera.items() if key in CameraSpec.__dataclass_fields__})
    for key in ("near_clip", "far_clip"):
        if getattr(args, key) is not None:
            if key in camera and not math.isclose(getattr(args, key), camera[key]):
                raise ValueError(f"{key} conflicts with summary.json")
            setattr(spec, key, getattr(args, key))
    spec.validate()
    encoding = args.encoding
    if encoding == "auto":
        # 旧墙面脚本保存归一化矩阵，新脚本明确记录米制编码；不根据数值猜单位。
        encoding = metadata.get("depth_encoding", "normalized" if "camera" in metadata and "yaw_degrees" in metadata else "auto")
    backends = args.backend or [name for name in BACKENDS if (root / name).is_dir()]
    if not backends:
        raise ValueError("No backend directories found")
    files = {}
    for backend in backends:
        if backend in metadata.get("backends", {}) and metadata["backends"][backend]["status"] != "ok":
            raise ValueError(f"Backend {backend} was not generated successfully")
        directory = root / backend
        files[backend] = {p.stem: p for p in sorted(directory.glob("*.png"))}
        files[backend].update({p.stem: p for p in sorted(directory.glob("*.npy"))})
        if not files[backend]:
            raise ValueError(f"No depth files found in {directory}")
    common = sorted(set.intersection(*(set(value) for value in files.values())), key=frame_key)
    if not common:
        raise ValueError("No matching frame names across selected backends")
    for backend in backends:
        dropped = len(files[backend]) - len(common)
        if dropped:
            print(f"[{backend}] excluded {dropped} unmatched frames")
    if args.max_frames:
        common = common[:args.max_frames]
    depths = {backend: {} for backend in backends}
    sources = {}
    for name in common:
        shape = None
        for backend in backends:
            path = files[backend][name]
            depth = read_depth(path, encoding, spec)
            if shape is not None and depth.shape != shape:
                raise ValueError(f"Shape mismatch for {name}: {shape} vs {depth.shape}")
            shape = depth.shape
            depths[backend][name] = depth
            sources[f"{backend}/{name}"] = str(path.resolve())
    spec.height, spec.width = next(iter(depths[backends[0]].values())).shape
    if any(Path(path).suffix == ".png" for path in sources.values()):
        print(f"Warning: PNG input is quantized at {(spec.far_clip - spec.near_clip) / 255:.6f} m per level")
    return depths, spec, {"encoding": encoding, "sources": sources, "frames": common}


def depth_metrics(lhs, rhs):
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
    difference = np.abs(lhs.astype(np.float64, copy=False) - rhs.astype(np.float64, copy=False))
    return difference, {"pixels": int(difference.size), "mae_m": float(difference.mean()), "rmse_m": float(np.sqrt(np.square(difference).mean())), "max_m": float(difference.max())}


def compare(depths, spec, provenance, root):
    if len(depths) < 2:
        raise ValueError("Comparison requires at least two backends")
    output = root / "compare"
    output.mkdir(exist_ok=True)
    rows = []
    for lhs, rhs in combinations(depths, 2):
        directory = output / f"{lhs}_vs_{rhs}"
        directory.mkdir(exist_ok=True)
        for name in provenance["frames"]:
            difference, metrics = depth_metrics(depths[lhs][name], depths[rhs][name])
            rows.append({"lhs": lhs, "rhs": rhs, "frame": name, **metrics})
            np.save(directory / f"{name}.npy", difference.astype(np.float32))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "summary.json", {"camera": asdict(spec), **provenance, "comparisons": rows})
    print(f"Saved {len(rows)} frame comparisons: {output / 'metrics.csv'}")
    return rows


# ---------- plot ----------

def configure_plot(font_file):
    if font_file is not None:
        font_manager.fontManager.addfont(str(font_file))
    path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    if font_manager.FontProperties(fname=path).get_name() != "Times New Roman":
        raise ValueError("Times New Roman is required; provide --font_file if necessary")
    matplotlib.rcParams.update({
        "font.family": "Times New Roman", "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "text.color": "#242424", "axes.labelcolor": "#242424", "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.6, "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    print(f"Plot font: {path}")
    return path


def frame_label(name):
    key = frame_key(name)
    return f"Yaw {key[1]:+d}°" if key[0] == 0 else name


def plot_panels(panels, frames, title, color_label, cmap, lower, upper, output, stem, args):
    columns = list(panels)
    paths = []
    for page, offset in enumerate(range(0, len(frames), args.rows_per_page), start=1):
        selected = frames[offset:offset + args.rows_per_page]
        horizontal = len(columns) == 1
        rows, cols = (1, len(selected)) if horizontal else (len(selected), len(columns))
        figure, axes = plt.subplots(rows, cols, figsize=(3.1 * cols + 1.0, 2.9 * rows + 1.0), squeeze=False, constrained_layout=True)
        heading = f"{title}\n{columns[0]}" if horizontal else title
        figure.suptitle(heading, fontsize=18, fontweight="normal")
        try:
            for row, name in enumerate(selected):
                for col, label in enumerate(columns):
                    values, caption = panels[label][name]
                    axis = axes[0, row] if horizontal else axes[row, col]
                    image = axis.imshow(values, cmap=cmap, vmin=lower, vmax=upper, interpolation="nearest", aspect="equal")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if horizontal:
                        axis.set_title(frame_label(name), pad=10)
                    elif row == 0:
                        axis.set_title(label, pad=10)
                    if col == 0 and not horizontal:
                        axis.set_ylabel(frame_label(name), labelpad=12)
                    axis.set_xlabel(caption, labelpad=8)
            colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025, shrink=0.86)
            colorbar.set_label(color_label, labelpad=12)
            colorbar.outline.set_visible(False)
            path = output / f"{stem}_{page:02d}.pdf"
            figure.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
            paths.append(str(path))
        finally:
            plt.close(figure)
    return paths


def plot(depths, spec, provenance, root, args):
    font = configure_plot(args.font_file)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    names = {"isaac": "Isaac Gym", "warp": "NVIDIA Warp", "mujoco": "MuJoCo"}
    error_map = LinearSegmentedColormap.from_list("error_orange", ["#fff8ef", "#f6cf9c", "#db8938", "#874315"])
    panels = {}
    for backend, values in depths.items():
        panels[names[backend]] = {name: (depth, f"{depth.shape[1]} × {depth.shape[0]} pixels") for name, depth in values.items()}
    all_values = [value for values in depths.values() for value in values.values()]
    if any(np.any((value < spec.near_clip - 1e-6) | (value > spec.far_clip + 1e-6)) for value in all_values):
        raise ValueError("Depth exceeds the plotting range; supply matching near_clip/far_clip")
    # 原始米制深度直接映射到共享灰度范围，不使用伪彩色或逐图归一化。
    paths = plot_panels(panels, provenance["frames"], "Depth comparison", "Axial depth (m)", "gray", spec.near_clip, spec.far_clip, output, "depth", args)
    errors = {}
    maximum = 0.0
    for lhs, rhs in combinations(depths, 2):
        label = f"{names[lhs]} − {names[rhs]}"
        errors[label] = {}
        for name in provenance["frames"]:
            difference, metrics = depth_metrics(depths[lhs][name], depths[rhs][name])
            maximum = max(maximum, metrics["max_m"] * 1000)
            errors[label][name] = (difference * 1000, f"MAE = {metrics['mae_m'] * 1000:.2f} mm")
    if errors:
        # 所有面板和分页共享同一色标，零误差时保留 1 mm 色标范围。
        paths += plot_panels(errors, provenance["frames"], "Absolute depth difference", "Absolute error (mm)", error_map, 0, maximum or 1.0, output, "error", args)
    save_json(output / "summary.json", {"font": font, "dpi": args.dpi, "format": "pdf", "depth_colormap": "gray", "depth_range_m": [spec.near_clip, spec.far_clip], "error_range_mm": [0, maximum or 1.0], **provenance, "outputs": paths})
    print(f"Saved {len(paths)} figure files: {output}")


# ---------- main ----------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate, compare, and plot depth images for research.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{generate,compare,plot,all}")
    descriptions = {"generate": "Generate depth frames.", "compare": "Compare saved depth frames.", "plot": "Plot research figures.", "all": "Generate, compare, and plot."}
    for name in ("generate", "compare", "plot", "all"):
        child = commands.add_parser(name, help=descriptions[name])
        generating = name in ("generate", "all")
        child.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS) if generating else None, help="Backends to generate or select.")
        child.add_argument("--near_clip", type=float, default=0.0 if generating else None, help="Near depth in meters.")
        child.add_argument("--far_clip", type=float, default=2.0 if generating else None, help="Far depth in meters.")
        if generating:
            child.add_argument("--out", type=Path, default=ROOT / "logs" / "depth_backend_test" / datetime.now().strftime("%Y%m%d_%H%M%S"), help="Empty output directory.")
            child.add_argument("--width", type=int, default=64)
            child.add_argument("--height", type=int, default=64)
            child.add_argument("--horizontal_fov", type=float, default=58.0)
            child.add_argument("--device", default="cuda:0", help="Warp/Isaac device index; use full CUDA visibility.")
        else:
            child.add_argument("--root", type=Path, required=True, help="Root directory containing backend subdirectories.")
            child.add_argument("--encoding", choices=("auto", "meters", "normalized"), default="auto", help="NPY encoding; normalized means [-0.5, 0.5]. PNG is linear 8-bit depth.")
            child.add_argument("--max_frames", type=int, default=0, help="Maximum common frames; 0 means all.")
        if name in ("plot", "all"):
            child.add_argument("--font_file", type=Path, help="Optional Times New Roman font file.")
            child.add_argument("--dpi", type=int, default=300)
            child.add_argument("--rows_per_page", type=int, default=3)
    worker = commands.add_parser("_worker")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--backend", choices=BACKENDS, required=True)
    worker.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.command != "_worker":
        if args.backend and len(args.backend) != len(set(args.backend)):
            parser.error("Backend names must be unique")
        if getattr(args, "max_frames", 0) < 0 or getattr(args, "dpi", 1) <= 0 or getattr(args, "rows_per_page", 1) <= 0:
            parser.error("max_frames must be nonnegative; dpi and rows_per_page must be positive")
        if args.command == "all" and len(args.backend) < 2:
            parser.error("all requires at least two backends; use generate and plot for a single backend")
    return args


def main(args):
    if args.command == "_worker":
        generate_worker(args)
        return
    if args.command in ("generate", "all"):
        args.root = generate(args)
        if args.command == "generate":
            return
        args.encoding, args.max_frames = "auto", 0
    depths, spec, provenance = load_depths(args)
    if args.command in ("compare", "all"):
        compare(depths, spec, provenance, args.root)
    if args.command in ("plot", "all"):
        plot(depths, spec, provenance, args.root, args)


if __name__ == "__main__":
    args = parse_args()
    exit_code = 0
    try:
        main(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    if args.command == "_worker" and args.backend == "isaac":
        # 仿真已显式释放；绕过旧版原生库在解释器销毁阶段的崩溃，保留真实失败码。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    sys.exit(exit_code)
