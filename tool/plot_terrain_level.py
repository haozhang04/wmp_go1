#!/usr/bin/env python3
#  (c) 2026-2027 zh

# 用法：
# 1. 绘制一个 TensorBoard run 目录中的地形等级趋势：
#    python tool/plot_terrain_level.py --paths logs/a1_wmp_example/<run>
# 2. 同时比较多个 run 目录或 event 文件：
#    python tool/plot_terrain_level.py --paths <run_dir_1> <event_file_1> <run_dir_2>
#
# 说明：
# - 读取 TensorBoard event 中的 Episode/terrain_level 标量
# - 每隔 SAMPLE_EVERY 个数据点采样一次
# - 输出图片保存到 logs/plot_terrain_level
# - 输入既可以是 run 目录，也可以是 events.out.tfevents.* 文件

import argparse
from datetime import datetime
from pathlib import Path


SAMPLE_EVERY = 20
OUTPUT_DIR = Path("logs/plot_terrain_level")
TERRAIN_TAG = "Episode/terrain_level"
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Run directories or TensorBoard event files to plot",
    )
    return parser.parse_args()


def find_event_file(log_dir):
    log_dir = Path(log_dir).expanduser()
    candidates = [path for path in log_dir.glob("events.out.tfevents.*") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found in {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_event_terrain_level(event_file):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    event_accumulator.Reload()
    if TERRAIN_TAG not in event_accumulator.Tags().get("scalars", []):
        return []
    return [(event.step, event.value) for event in event_accumulator.Scalars(TERRAIN_TAG)]


def load_terrain_level(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        if not path.name.startswith("events.out.tfevents."):
            raise ValueError(f"Expected a run directory or TensorBoard event file: {path}")
        return parse_event_terrain_level(path), path

    event_file = find_event_file(path)
    return parse_event_terrain_level(event_file), event_file


def make_label(log_dir):
    path = Path(log_dir)
    if path.is_file():
        path = path.parent
    return path.name or path.parent.name


def plot_runs(all_runs, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_FAMILY,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    for label, points in all_runs:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(
            x_values,
            y_values,
            linewidth=1.4,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.5,
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode terrain_level")
    ax.set_title("Terrain Level Trend")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    ax.tick_params(direction="in", length=3.5, width=0.8, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    all_runs = []
    for path in args.paths:
        points, source_file = load_terrain_level(path)
        if not points:
            raise ValueError(f"No terrain_level data found in {source_file}")
        sampled_points = points[::SAMPLE_EVERY]
        all_runs.append((make_label(path), sampled_points))
        print(f"{path}: {len(sampled_points)}/{len(points)} plotted points from {source_file}")

    output_dir = OUTPUT_DIR.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"terrain_level_trend_{datetime.now():%Y%m%d_%H%M%S}.png"
    plot_runs(all_runs, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
