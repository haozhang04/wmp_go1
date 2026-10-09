# 各地形 Command 课程设计

- 本文对应 `a1_amp` 默认训练配置。
- 速度课程分为 `omni`、`rough_flat`、`stairs_discrete`、`forward`、`slope_cliff` 五组。
- 同一课程组共享速度上限，不同课程组独立升级。

## 当前配置与 Command 含义

```python
curriculum = True
heading_command = True
resampling_time = 10.0
limit_vel_prob = 0.0
zero_command_prob = 0.05
limit_vel = [-1.0, 0.0, 1.0]
direct_yaw_values = [-1.5, 0.0, 1.5]
```

- 指令在机器人重置时及每个 episode 内每隔 10 秒重新采样。
- 当前默认关闭极限速度采样。

- `lin_vel_x`、`lin_vel_y`：机体坐标系中的目标线速度，单位为 m/s。
- `heading`：世界坐标系中的目标朝向，单位为 rad。
- `ang_vel_yaw`：目标偏航角速度，单位为 rad/s。

- 启用 `heading_command` 后，每步按朝向误差计算角速度：

```text
ang_vel_yaw = clip(2.0 × wrap_to_pi(目标 heading − 当前 heading), −π, π)
```

## 五个速度课程组

### 1. 普通全方向组（`omni`）

- 包含 wave、rough_slope，两种地形共享速度上限。
- 允许前后、左右运动和转向。

- 初始范围：X、Y 均为 `±0.5 m/s`。
- 最终范围：X 为 `±1.5 m/s`，Y 为 `±1.0 m/s`。
- heading 范围：`±π`。

### 2. 粗糙平地全方向组（`rough_flat`）

- 仅包含 rough_flat，速度上限独立于 `omni` 组。
- 允许前后、左右运动和转向。

- 初始范围：X、Y 均为 `±0.5 m/s`。
- 最终范围：X 为 `±2.0 m/s`，Y 为 `±1.0 m/s`。
- heading 范围：`±π`。

### 3. 楼梯与离散障碍全方向组（`stairs_discrete`）

- 包含 stairs_up、stairs_down、discrete，三种地形共享速度上限。
- 允许前后、左右运动和转向。

- 初始范围：X、Y 均为 `±0.5 m/s`。
- 最终范围：X 为 `±1.0 m/s`，Y 为 `±0.8 m/s`。
- heading 范围：`±π`。

### 4. 普通只向前障碍组（`forward`）

- 以下地形共享速度上限：

- gap、continuous_gap、bream。
- stairs_cliff、stepping_stones、stepping_one_bridge。
- climb、tilt、stool、crawl。

- 初始范围：X 为 `[0, 0.5] m/s`，Y 为 `0 m/s`。
- 最终范围：X 为 `[0, 1.0] m/s`，Y 为 `0 m/s`。
- heading 范围：固定为 `0`。

### 5. 坡道悬崖只向前组（`slope_cliff`）

- 仅包含 slope_cliff，速度上限独立于 `forward` 组。

- 初始范围：X 为 `[0, 0.5] m/s`，Y 为 `0 m/s`。
- 最终范围：X 为 `[0, 1.5] m/s`，Y 为 `0 m/s`。
- heading 范围：固定为 `0`。

## 各地形目标范围

- X、Y 的单位为 m/s，Heading 的单位为 rad。
- “最终”表示课程允许达到的最大范围，不保证训练一定达到。

| 地形 | 方向 | 初始 X | 最终 X | 初始 Y | 最终 Y | Heading |
|---|---|---:|---:|---:|---:|---:|
| wave | 全方向 | `±0.5` | `±1.5` | `±0.5` | `±1.0` | `±π` |
| rough_slope | 全方向 | `±0.5` | `±1.5` | `±0.5` | `±1.0` | `±π` |
| rough_flat | 全方向 | `±0.5` | `±2.0` | `±0.5` | `±1.0` | `±π` |
| stairs_up | 全方向 | `±0.5` | `±1.0` | `±0.5` | `±0.8` | `±π` |
| stairs_down | 全方向 | `±0.5` | `±1.0` | `±0.5` | `±0.8` | `±π` |
| discrete | 全方向 | `±0.5` | `±1.0` | `±0.5` | `±0.8` | `±π` |
| stairs_cliff | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| slope_cliff | 只向前 | `[0, 0.5]` | `[0, 1.5]` | `0` | `0` | `0` |
| stepping_stones | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| stepping_one_bridge | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| gap | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| continuous_gap | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| bream | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| climb | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| tilt | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| stool | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |
| crawl | 只向前 | `[0, 0.5]` | `[0, 1.0]` | `0` | `0` | `0` |

- 当前 wave、stepping_stones、tilt 的比例为 0%，不创建对应环境，但保留课程定义。

## 课程升级原则

- 各组独立累计已结束 episode 的线速度跟踪奖励与数量。
- 数量达到组内环境总数后，评估并清空统计。
- 零步 reset 不计入，提前终止计入。

```text
linear_score = accumulated_tracking_lin_vel_reward / (episode_count × max_episode_length × tracking_lin_vel_reward_scale)
```

- 奖励系数为 `self.reward_scales["tracking_lin_vel"]`，角速度奖励不参与判断。
- 提前终止仍按完整 episode 长度归一化，会降低分数。

- `linear_score > 0.8`：X 上限增加 `0.2 m/s`，全方向组的 Y 上限同步增加。
- `linear_score ≤ 0.8`：保持不变。

- 上限不超过表中最终值，只升级、不降级；无环境的组不更新。

## 零速度采样、极限采样与方向处理

### 全方向组：`omni`、`rough_flat`、`stairs_discrete`

- Command 课程开启时，先按当前组范围均匀采样，再用同一随机数选择互斥分支：

| 采样分支 | 概率 | 当前配置 |
|---|---|---|
| 极限采样 | `limit_vel_prob` | `0%`，关闭 |
| 零速度采样 | `zero_command_prob` | `5%` |
| 连续采样 | `1 - limit_vel_prob - zero_command_prob` | `95%` |

#### 零速度采样

- 仅清零 X、Y，保留 heading，仍可原地转向。
- 全方向组不执行 `0.2 m/s` 低速自动清零，也不使用 `cheat` 方向惩罚。

#### 极限采样

- `limit_vel_prob > 0` 时启用，X、Y、yaw 独立均匀采样，共 27 种等概率组合：

```text
X   ∈ {-当前组 X 上限, 0, +当前组 X 上限}
Y   ∈ {-当前组 Y 上限, 0, +当前组 Y 上限}
yaw ∈ {-1.5, 0, +1.5} rad/s
```

- yaw 由 `direct_yaw_values` 指定，不随课程增长。
- 命中时设置 `direct_yaw_mask = True`，跳过 heading 换算。
- 下次重采样或重置时重新确定标记。

- 极限组合中有 `1/9` 满足 `X = Y = 0`。
- 零线速度总概率为 `zero_command_prob + limit_vel_prob / 9`。

### 只向前组：`forward`、`slope_cliff`

- 不参与上述零速度采样和极限采样。
- XY 合速度不超过 `0.2 m/s` 时清零。
- 固定 `heading = 0`，偏离时按朝向误差转向。
- 世界坐标系偏航角绝对值超过 `1.0 rad` 时，施加 `cheat` 惩罚。

## Play 与地形查看

- 两者均关闭 Command 课程，固定 `vy = 0`、`heading = 0`，角速度仍由朝向误差计算。
- `play.py` 设置 `vx = 0.6 m/s`，`view_terrain.py` 设置 `vx = 0`。
