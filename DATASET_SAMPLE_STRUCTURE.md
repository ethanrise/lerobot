# LeRobotDataset 单个样本(sample)结构笔记

基于 `datasets/libero`(`HuggingFaceVLA/libero`)实测，配套脚本见同目录 `test_libero_data.py`。

**阅读顺序**：① 数据集宏观结构(task/episode/frame 三层) → ② 单帧 sample 在代码里长什么样(扁平dict+实测例子) → ③ 逐字段速查表 → ④ 一个会改变字段形状的特殊机制(`delta_timestamps`) → ⑤ 一图归纳全部。

---

## 1. 数据集宏观结构：task → episode → frame

数据集不是一堆孤立的帧，是按 **task → episode → frame** 三层组织的，每一层都是"一对多"：

- **一个数据集** 包含多个 **task**(不同的语言指令)
- **一个 task** 被反复执行/演示了多次，**每一次完整的执行记录就是一个 episode**
- **一个 episode** 包含连续的多个 **frame**(从复位到结束)

`datasets/libero` 的真实统计数字：

```
数据集：40 个 task，1693 个 episode，273465 帧
  → 平均每个 task  ≈ 42 个 episode   (1693 / 40)
  → 平均每个 episode ≈ 161 帧         (273465 / 1693)
```

展开看某一个具体 task：

```
数据集(40 tasks, 1693 episodes, 273465 frames)
└─ task("pick up the alphabet soup...")
    ├─ episode 813(一次完整演示，155帧)
    ├─ episode 818(另一次演示，物体摆放不同)
    ├─ ...                              (该 task 共 44 个 episode)
    └─ episode N
        └─ episode 813 内部：
           frame_index = 0, 1, 2, ..., 154   ← 每一帧就是 ds[i] 取出的一个 sample
           这155帧的 task_index 全部指向同一句话
```

**episode 怎么存的**：`meta/episodes` 表里一行元数据，关键字段是 `dataset_from_index`/`dataset_to_index`(这个 episode 在全局 `index` 上的起止区间，左闭右开)和 `length`(帧数)：

```python
meta.episodes[0] = {'episode_index': 0, 'dataset_from_index': 0,   'dataset_to_index': 214, 'length': 214,
                     'tasks': ['put the white mug on the left plate and put the yellow and white mug on the right plate']}
meta.episodes[1] = {'episode_index': 1, 'dataset_from_index': 214, 'dataset_to_index': 498, 'length': 284,
                     'tasks': ['put the white mug on the plate and put the chocolate pudding to the right of the plate']}
```

episode 1 的起点(214)正好接上 episode 0 的终点(214)——**所有 episode 在全局 `index` 上首尾相接、无缝拼起来**。`lerobot_train.py` 的 `EpisodeAwareSampler` 就是用 `dataset_from_index`/`dataset_to_index` 这两列切分每个 episode 的采样边界(训练时按 episode 为单位 shuffle，不会把不同 episode 的帧混进同一个窗口)。

**跟 task 的关系是多对一**：
- `tasks` 字段是个列表，schema 支持一个 episode 标多个任务描述，但 LIBERO 每个 episode 都只填了 1 个
- 一个 task 通常对应很多个 episode(上面"alphabet soup"那个例子是 44 个)，同一任务被不同初始摆放反复演示
- 单帧上的 `task`/`task_index` 就是它所属 episode 的任务描述，**同一 episode 内所有帧的 `task_index` 都一致**(中途不会换任务)

---

## 2. 单帧 sample 在代码里长什么样

`ds = LeRobotDataset(repo_id, root, episodes=[...], delta_timestamps=...)`，`ds[i]` 取出的就是第1节最底层的那个单元——**一帧**。返回值不是嵌套结构，是**一个扁平 dict**：key 是字符串(用 `.` 分隔表示层级，比如 `observation.images.image`)，value 是 `torch.Tensor`(`task` 例外，是 `str`)。区分"观测 vs 动作 vs 元信息"完全靠 key 的命名前缀，不是字典结构本身。

实测一下(`test_libero_data.py`，配置 `action_horizon=16`，`state_history_offsets=image_history_offsets=[-0.2,-0.1,0.0]`)，`ds[0]` 的 14 个 key 和形状：

```
index             : shape=(),      dtype=int64    value=0
episode_index     : shape=(),      dtype=int64    value=0
frame_index       : shape=(),      dtype=int64    value=0
timestamp         : shape=(),      dtype=float32  value=0.0
task_index        : shape=(),      dtype=int64    value=0
task              : str                          value="put the white mug on the left plate..."
observation.state : shape=(3, 8),  dtype=float32  # 3个时间戳 × 8维状态
action            : shape=(16, 7), dtype=float32  # 16个时间戳 × 7维动作
observation.images.image  : shape=(3, 3, 256, 256), dtype=float32  # 3个时间戳 × (C,H,W)
observation.images.image2 : shape=(3, 3, 256, 256), dtype=float32
action_is_pad / observation.state_is_pad / observation.images.image*_is_pad : BoolTensor，形状跟对应字段的"时间戳数"一致
```

跟第1节对上号：`episode_index=0`，`len(ds)=214` 正好等于 `meta.episodes[0]['length']`(脚本里 `episodes=[0]` 只选了这一个)。

**关键观察**：`index`/`episode_index`/`frame_index`/`timestamp`/`task_index` 全是**标量**，而 `action`/`observation.state`/两个相机 key 都带了**时间维度**(3 或 16)——原因见第4节。

---

## 3. 字段速查表

### 索引/标识类(定位"这一帧是谁"，具体概念见第1节，这里只列速查)

| Key | 类型 | 取值 |
|---|---|---|
| `index` | `int64` 标量 | 这一帧在**整个数据集**里的全局行号(0 ~ total_frames-1)，所有 episode 的帧首尾拼接连续编号，换 episode **不重置**。换算关系：`index = dataset_from_index(当前episode) + frame_index`(比如 episode 1 从 `dataset_from_index=214` 开始，它的第0帧 `frame_index=0` 对应 `index=214`，第1帧 `frame_index=1` 对应 `index=215`) |
| `episode_index` | `int64` 标量 | 第几个 episode(全局编号，从 0 开始) |
| `frame_index` | `int64` 标量 | 这一帧在**当前 episode 内部**的局部编号，每个新 episode 开始都从 0 重新数，跟 `index` 用的是不同的计数基准 |
| `timestamp` | `float32` 标量 | episode 内时间戳(秒) = `frame_index / fps`，跟 `frame_index` 一样是 episode 内局部量，不是全局时间 |
| `task_index` | `int64` 标量 | 语言指令在 `meta/tasks.parquet` 里的整数索引 |
| `task` | `str` | 语言指令字符串，`__getitem__` 时拿 `task_index` 反查得到，不是原始列 |

### 观测类(模型的输入)

| Key | 形状(单帧) | 含义 |
|---|---|---|
| `observation.images.image` | `(3,256,256)` CHW | agentview(第三人称)相机，float32 归一化到 `[0,1]` |
| `observation.images.image2` | `(3,256,256)` CHW | robot0_eye_in_hand(腕部)相机 |
| `observation.state` | `(8,)` | 本体状态，**已处理成扁平向量** = eef位置(3)+轴角朝向(3)+夹爪qpos(2)。跟仿真环境(`LiberoEnv`)实时输出的结构化 `robot_state` 字典是两套格式，数据集里存的是采集时已经拍平好的版本 |

`observation.*` 是 LeRobot 全局统一命名规范，不管哪个机器人/仿真环境都用这套前缀，让不同数据集能共用同一套训练代码。

### 动作类(模型的标签)

| Key | 形状(单帧) | 含义 |
|---|---|---|
| `action` | `(7,)` | LIBERO：6D末端笛卡尔增量 + 1D夹爪，范围 `[-1,1]` |

### padding 标记类(条件性出现)

| Key | 含义 |
|---|---|
| `action_is_pad` / `observation.state_is_pad` / `observation.images.image(2)_is_pad` | `BoolTensor`，跟对应字段的时间维度对齐。`True` = 该时间戳超出当前 episode 边界，是被钳制(clamp)补出来的，不是真实帧 |

只有构造 `LeRobotDataset` 时传了 `delta_timestamps`、且其中包含对应 key，才会出现这四个字段；不传 `delta_timestamps` 则完全没有 `_is_pad`。

---

## 4. 为什么有的字段是标量、有的带时间维度

由 `delta_timestamps` 参数决定：

```python
delta_timestamps = {
    "observation.state": [-0.2, -0.1, 0.0],          # 3 个时间偏移
    "action": [t / meta.fps for t in range(16)],      # 16 个时间偏移
}
for cam_key in meta.camera_keys:
    delta_timestamps[cam_key] = [-0.2, -0.1, 0.0]
```

**只有出现在 `delta_timestamps` 里的 key 才会被展开成时间窗口**(最前面多一维)；没出现的 key(`index`/`episode_index`/`frame_index`/`timestamp`/`task_index`/`task`)永远只描述"当前这一帧"，保持标量。

对应实测形状：`observation.state` → `(3,8)`；`action` → `(16,7)`；`observation.images.image` → `(3,3,256,256)`。

完全不传 `delta_timestamps`(最常见用法)时，`action`/`observation.state`/相机图像都只是单帧张量(`(7,)`/`(8,)`/`(3,256,256)`)，跟 `index` 一样不带时间维度，也就没有任何 `_is_pad` 字段。

---

## 5. 一图归纳

```
数据集(40 tasks, 1693 episodes, 273465 frames)
└─ task("pick up the alphabet soup...")              ← 一句语言指令，对应多个episode
    └─ episode 813(一次完整演示，155帧)                ← meta/episodes 里一行元数据
        └─ frame(frame_index=0..154)                  ← 每一帧 = ds[i] 一次取值，一个扁平dict
            │
            ├─ 标识/索引(始终标量，不受 delta_timestamps 影响)
            │   ├─ index / episode_index / frame_index / timestamp
            │   └─ task_index / task
            │
            ├─ 观测 observation.*(可被 delta_timestamps 展开成时间窗口)
            │   ├─ observation.images.image   agentview 相机
            │   ├─ observation.images.image2  腕部相机
            │   └─ observation.state          8维本体状态
            │
            ├─ 动作(可被 delta_timestamps 展开成时间窗口)
            │   └─ action                     7维
            │
            └─ padding 标记(只有对应字段配了 delta_timestamps 才出现)
                └─ action_is_pad / observation.state_is_pad / observation.images.image(2)_is_pad
```
