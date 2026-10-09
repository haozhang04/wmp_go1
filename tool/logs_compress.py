#  (c) 2026-2027 zh

# 用法：
# 1. 预览默认日志目录下将被压缩的内容，不会真的压缩：
#    python tool/logs_compress.py
# 2. 压缩默认日志目录下的内容：
#    python tool/logs_compress.py --y
# 3. 指定其他日志目录并预览：
#    python tool/logs_compress.py --logpath path
# 4. 指定其他日志目录并压缩：
#    python tool/logs_compress.py --logpath path --y
#
# 说明：
# - 默认日志目录是：logs
# - 每个实验目录生成一个 <实验目录名>.tar.zst
# - 直接压缩每个实验目录下的所有内容，不排除任何文件
# - 默认只预览，加 --y 才执行；需要系统可用 tar 和 zstd

import argparse
import subprocess
from pathlib import Path


DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"


def build_compress_plan(logs_path):
    return [
        (project, logs_path / f"{project.name}.tar.zst")
        for project in sorted(d for d in logs_path.iterdir() if d.is_dir())
    ]


def print_compress_tree(logs_path, compress_plan):
    print("\nCompress tree:")

    for project_index, (project, output_zst) in enumerate(compress_plan):
        is_last_project = project_index == len(compress_plan) - 1
        project_prefix = "`-- " if is_last_project else "|-- "
        child_prefix = "    " if is_last_project else "|   "

        print(f"{project_prefix}{project.relative_to(logs_path)}/ -> {output_zst.name}")
        print(f"{child_prefix}`-- [include] everything")


def run_compress_plan(compress_plan):
    compressed_count = 0
    failed_count = 0
    for project, output_zst in compress_plan:
        tar_cmd = [
            "tar",
            "-I", "zstd -T0 -3",
            "-C", str(project.parent),
            "-cf", str(output_zst),
            project.name,
        ]

        try:
            subprocess.run(tar_cmd, check=True)
            compressed_count += 1
            final_size = output_zst.stat().st_size / (1024 * 1024)
            print(f"Compressed {project.name} -> {output_zst.name} ({final_size:.2f} MB)")
        except subprocess.CalledProcessError as e:
            failed_count += 1
            print(f"Compression failed for {project.name}: {e}")

    return compressed_count, failed_count


def smart_compress(logs_root, dry_run=True):
    logs_path = Path(logs_root).resolve()
    if not logs_path.exists():
        print(f"Can't find log directory: {logs_path}")
        return 1
    if not logs_path.is_dir():
        print(f"Not a directory: {logs_path}")
        return 1

    compress_plan = build_compress_plan(logs_path)

    print(f"Log directory: {logs_path}")
    print(f"Archives to create: {len(compress_plan)}")

    if not compress_plan:
        print("No project directories found.")
        return 0

    print_compress_tree(logs_path, compress_plan)

    if dry_run:
        print("\nDry run only. Add --y to actually compress these files.")
        return 0

    compressed_count, failed_count = run_compress_plan(compress_plan)
    print(f"\nCreated {compressed_count} archive files.")
    if failed_count:
        print(f"Failed to create {failed_count} archive files.")
        return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress each project directory under logs, including all files."
    )
    parser.add_argument(
        "--logpath",
        default=str(DEFAULT_LOGS_DIR),
        help=f"Root logs directory. Default: {DEFAULT_LOGS_DIR}",
    )
    parser.add_argument(
        "--y",
        action="store_true",
        help="Actually compress files.",
    )
    args = parser.parse_args()

    raise SystemExit(smart_compress(args.logpath, dry_run=not args.y))
