#  (c) 2026-2027 zh

# 用法：
# 1. 预览默认日志目录下将被删除的旧模型文件，不会真的删除：
#    python tool/logs_delete.py
# 2. 删除默认日志目录下的旧模型文件：
#    python tool/logs_delete.py --y
# 3. 指定其他日志目录并预览：
#    python tool/logs_delete.py --logpath path
# 4. 指定其他日志目录并删除：
#    python tool/logs_delete.py --logpath path --y
#
# 说明：
# - 默认日志目录是：logs
# - 脚本会递归查找 model_<训练次数>.pt 和 model_<训练次数>.pth
# - 每个训练目录只保留训练次数最大的模型文件，删除其他较旧模型文件
# - 不加 --y 时只预览，不会删除文件

import re
import argparse
from pathlib import Path
from collections import defaultdict


MODEL_RE = re.compile(r"^model_(\d+)\.(pt|pth)$")
DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"


def iter_model_files(logs_root):
    for path in logs_root.rglob("*"):
        if not path.is_file():
            continue
        match = MODEL_RE.match(path.name)
        if match is None:
            continue
        yield path, int(match.group(1))


def find_old_checkpoints(logs_root):
    by_dir = defaultdict(list)
    for path, iteration in iter_model_files(logs_root):
        by_dir[path.parent].append((iteration, path))

    old_files = []
    kept_files = []
    checkpoint_plan = []
    for run_dir, models in sorted(by_dir.items()):
        max_iteration = max(iteration for iteration, _ in models)
        kept_in_dir = []
        old_in_dir = []
        for iteration, path in sorted(models):
            if iteration == max_iteration:
                kept_files.append(path)
                kept_in_dir.append(path)
            else:
                old_files.append(path)
                old_in_dir.append(path)
        checkpoint_plan.append((run_dir, kept_in_dir, old_in_dir))

    return kept_files, old_files, checkpoint_plan


def print_checkpoint_tree(logs_root, checkpoint_plan):
    print("\nCheckpoint tree:")
    for run_index, (run_dir, kept_files, old_files) in enumerate(checkpoint_plan):
        is_last_run = run_index == len(checkpoint_plan) - 1
        run_prefix = "`-- " if is_last_run else "|-- "
        child_prefix = "    " if is_last_run else "|   "
        relative_run_dir = run_dir.relative_to(logs_root)

        print(f"{run_prefix}{relative_run_dir}/")

        entries = []
        entries.extend(("keep", path) for path in kept_files)
        entries.extend(("delete", path) for path in old_files)

        for entry_index, (label, path) in enumerate(entries):
            is_last_entry = entry_index == len(entries) - 1
            entry_prefix = "`-- " if is_last_entry else "|-- "
            print(f"{child_prefix}{entry_prefix}[{label}] {path.name}")


def delete_old_checkpoints(logs_root, dry_run=True):
    logs_root = Path(logs_root).resolve()
    if not logs_root.exists():
        print(f"Can't find log directory: {logs_root}")
        return 1
    if not logs_root.is_dir():
        print(f"Not a directory: {logs_root}")
        return 1

    kept_files, old_files, checkpoint_plan = find_old_checkpoints(logs_root)

    print(f"Log directory: {logs_root}")
    print(f"Keep latest checkpoints: {len(kept_files)}")
    print(f"Old checkpoints: {len(old_files)}")

    if not old_files:
        print("No old checkpoints found.")
        return 0

    print_checkpoint_tree(logs_root, checkpoint_plan)

    if not dry_run:
        for path in old_files:
            path.unlink()

    if dry_run:
        print("\nDry run only. Add --y to actually delete these files.")
    else:
        print(f"\nDeleted {len(old_files)} old checkpoint files.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete old model_*.pt/.pth checkpoint files under logs, keeping only the max iteration in each run directory."
    )
    parser.add_argument(
        "--logpath",
        default=str(DEFAULT_LOGS_DIR),
        help=f"Root logs directory. Default: {DEFAULT_LOGS_DIR}",
    )
    parser.add_argument(
        "--y",
        action="store_true",
        help="Actually delete files.",
    )
    args = parser.parse_args()

    raise SystemExit(delete_old_checkpoints(args.logpath, dry_run=not args.y))
