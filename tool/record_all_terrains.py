#!/usr/bin/env python3
# Copyright (c) 2026-2027 zh
"""
内容：
    依次调用 play.py，为多个模型 run 录制指定地形视频，可选使用 headless 模式。

用法：
    python tool/record_all_terrains.py
    python tool/record_all_terrains.py --runs logs/a1_wmp_example/<run1> <run2>
    python tool/record_all_terrains.py --headless --runs logs/a1_wmp_example/<run1> <run2>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
PLAY_SCRIPT = ROOT_DIR / "legged_gym" / "scripts" / "play.py"
TASK = "a1_amp"
SIM_DEVICE = "cuda:0"

DEFAULT_RUNS = []

SUPPORTED_TERRAINS = {
    "wave",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "discrete",
    "gap",
    "continuous_gap",
    "bream",
    "stairs_cliff",
    "slope_cliff",
    "stepping_stones",
    "stepping_one_bridge",
    "climb",
    "tilt",
    "stool",
    "crawl",
    "rough_flat",
}
DEFAULT_TERRAINS = [
    # "wave",
    # "rough_slope",
    # "stairs_up",
    # "stairs_down",
    # "discrete",
    "gap",
    "continuous_gap",
    "bream",
    "stairs_cliff",
    "slope_cliff",
    # "stepping_stones",
    "stepping_one_bridge",
    "climb",
    # "tilt",
    "stool",
    "crawl",
    # "rough_flat",
]


def resolve_run_dir(run: str) -> Path:
    run_dir = Path(run).expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT_DIR / run_dir
    run_dir = run_dir.resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not any(run_dir.glob("model_*.pt")):
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")
    return run_dir


def record_all_terrains(runs: Sequence[str], terrains: Sequence[str], headless: bool = False) -> None:
    unknown_terrains = sorted(set(terrains) - SUPPORTED_TERRAINS)
    if unknown_terrains:
        raise ValueError(f"Unsupported terrains: {', '.join(unknown_terrains)}")
    if not headless and not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is not set; non-headless video recording needs a viewer.")

    jobs = [(resolve_run_dir(run), terrain) for run in runs for terrain in terrains]
    environment = os.environ.copy()
    environment.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    conda_lib = str(Path(sys.prefix) / "lib")
    environment["LD_LIBRARY_PATH"] = ":".join(part for part in (conda_lib, environment.get("LD_LIBRARY_PATH")) if part)

    for job_index, (run_dir, terrain) in enumerate(jobs, start=1):
        experiment_name = run_dir.parent.name
        command = [
            sys.executable,
            str(PLAY_SCRIPT),
            f"--task={TASK}",
            f"--sim_device={SIM_DEVICE}",
            f"--experiment_name={experiment_name}",
            f"--load_run={run_dir.name}",
            f"--terrain={terrain}",
        ]
        if headless:
            command.append("--headless")

        print(f"[{job_index}/{len(jobs)}] Recording {run_dir.name} / {terrain}", flush=True)
        subprocess.run(command, cwd=ROOT_DIR, env=environment, check=True)

    print(f"Completed {len(jobs)} terrain videos.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run play.py sequentially and record videos for every terrain.")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--terrains", nargs="+", choices=sorted(SUPPORTED_TERRAINS), default=DEFAULT_TERRAINS)
    parser.add_argument("--headless", action="store_true", help="Run play.py without creating a viewer.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_all_terrains(args.runs, args.terrains, args.headless)
