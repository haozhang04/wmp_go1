# 远程使用 TensorBoard

默认启用快速加载，并通过 SSH 隧道在本地安全访问。

## 1. 准备日志

单个实验可直接将 `--logdir` 指向日志目录。对比多个实验时，使用符号链接汇总日志：

```bash
log_root="/path/to/logs"
tb_runs_dir="/tmp/tensorboard_runs"
mkdir -p "$tb_runs_dir"

ln -sfn "$log_root/run_1" "$tb_runs_dir/run_1"
ln -sfn "$log_root/run_2" "$tb_runs_dir/run_2"
ln -sfn "$log_root/run_3" "$tb_runs_dir/run_3"
```

将示例路径和实验名替换为实际值。原始日志目录应包含 `events.out.tfevents.*` 文件。

## 2. 服务器启动

建议在 `tmux` 中运行，避免 SSH 断开后服务退出：

```bash
tmux new -s tensorboard

conda activate "your_env_name"
tensorboard \
  --logdir="/tmp/tensorboard_runs" \
  --host=127.0.0.1 \
  --port=6006 \
  --reload_interval=15 \
  --load_fast=true
```

启动成功后，按 `Ctrl-b`、`d` 退出 tmux；使用 `tmux attach -t tensorboard` 恢复窗口。

## 3. 本地访问

在本地终端执行并保持连接：

```bash
ssh -N -L 6006:127.0.0.1:6006 your_username@your_server_address
```

浏览器打开：<http://127.0.0.1:6006>

## 注意事项

- 命令不存在：激活正确环境，或在该环境安装 TensorBoard。
- 端口被占用：用 `ss -ltnp 'sport = :6006'` 检查；只停止确认属于旧 TensorBoard 的进程。
- 本地端口被占用：改用 `-L 16006:127.0.0.1:6006`，并访问 <http://127.0.0.1:16006>。
- 快速加载报错：将 `--load_fast=true` 改为 `--load_fast=false`。
- 页面缺少实验：强制刷新页面，并在 **Runs** 面板勾选实验。
- 保留 `--host=127.0.0.1`，不要将 TensorBoard 直接暴露到公网。
