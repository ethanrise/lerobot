# LeRobot 仿真环境对比笔记

涉及的都是 `lerobot/envs/configs.py` 里 `EnvConfig.register_subclass(...)` 注册的 env 类型，用于 `lerobot-train`/`lerobot-eval` 的 `--env.type=...`。配套的手动预览脚本见同目录下的 `test_liberoenv_data.py` / `test_metaworldenv_data.py` / `test_pushtenv_data.py` / `test_alohaenv_data.py`。

- ✅ **已实测**：装了对应第三方包，跑过脚本，数值是真实输出
- ⚠️ **未实测**：第三方包没装(跟本仓库本身有依赖版本冲突，或体量太大没装)，信息来自读 `lerobot/envs/*.py` 源码整理

---

## 总对比表

| | 状态 | 本体 | 相机数(视角，是否固定) | image 尺寸 | state 维度 | action 维度/范围 | 语言指令 | fps | 底层引擎 |
|---|---|---|---|---|---|---|---|---|---|
| **LIBERO** | ✅ | 单臂 Franka+夹爪 | 2，固定 —— `image`(agentview，第三人称正面) + `image2`(robot0_eye_in_hand，腕部第一视角) | 256×256 或 360×360(可配) | 原始字典25维分散；处理后 **8维** | **7维**，[-1,1] | 有，固定句 | 80 | robosuite+MuJoCo |
| **MetaWorld** | ✅ | 单臂 Sawyer | 1，固定 —— `corner2`(桌角斜上方俯视，第三人称) | 480×480(可配) | **4维**(xyz+夹爪) | **4维**，[-1,1] | 有，固定句(50任务) | 80 | MuJoCo |
| **PushT** | ✅ | 2D 推子(非机械臂) | 1，固定 —— 纯俯视(top-down 2D) | 96×96(可视化384×384) | **2维**(像素xy) | **2维**，**[0,512]** | 无 | 10 | pymunk(2D) |
| **ALOHA** | ✅ | **双臂**，14 DOF | 1，固定 —— `top`(顶部俯视，覆盖双臂工作台) | 480×640 | **14维**(关节角) | **14维**，[-1,1] | 无 | 50 | dm_control+MuJoCo |
| RoboCasa | ⚠️ | 单臂 Franka(厨房) | 3，固定 —— `robot0_agentview_left`/`right`(左右两侧第三人称) + `robot0_eye_in_hand`(腕部第一视角) | 256×256(可视化512×512) | 16维 | 12维 | 有，**动态**(随episode变) | 20 | robosuite+MuJoCo |
| VLABench | ⚠️ | 单臂 Franka | 3 —— `image`(主视角/第三人称) + `second_image`(第二视角，具体角度未确认) + `wrist_image`(腕部第一视角) | 480×480(可配) | 7维 | 7维(eef模式) | 有，运行时取 | 10 | MuJoCo+dm_control |
| Isaac Lab Arena | ⚠️ | 可配，默认人形GR1 | 可配(`camera_keys`)，**默认关闭**，无固定视角 | 默认512×512 | 默认54维(可配) | 默认36维(可配) | 有，固定句 | 30(默认) | Isaac Sim/PhysX |
| LIBERO-plus | ⚠️ | 跟LIBERO一样 | 跟LIBERO一样(agentview+腕部)，但**视角本身是扰动维度之一**，变体里会换不同机位 | 跟LIBERO一样 | 跟LIBERO一样 | 跟LIBERO一样 | 有，**多种措辞变体** | 80 | 跟LIBERO一样 |
| RoboTwin 2.0 | ⚠️ | **双臂**Aloha-AgileX，14DOF | 3，固定 —— `head_camera`(躯干/胸前俯视) + `left_camera`/`right_camera`(双腕第一视角) | 240×320 | 14维 | 14维 | 有，仅任务名转写 | 25 | SAPIEN |
| RoboMME | ⚠️ | 未指定 | 2 —— `image`(第三人称主视角) + `wrist_image`(腕部第一视角) | 256×256 | 8维 | 8维或7维 | 无 | 10 | ManiSkill/SAPIEN |

---

## is_success 判断方式对比

| | is_success 来源 | 判断逻辑 |
|---|---|---|
| **LIBERO** | `self._env.check_success()`(`envs/libero.py:365`) | BDDL **符号谓词逻辑**：`goal_state` 里若干条 `On(物体,区域)` 谓词做逻辑**与**，全部满足才成功；判据来自 MuJoCo 实时物理状态(物体位置/接触) |
| **MetaWorld** | `info.get("success", 0)`(`envs/metaworld.py:253`) | **连续距离/角度阈值**：物体位置与随机采样的目标位置欧氏距离 ≤ 阈值(如 0.07m，按任务不同) |
| **PushT** | `gym_pusht/envs/pusht.py` 内部 | **2D 几何面积重叠率**：T形块与目标区域多边形交集面积/目标面积 > 0.95(95%覆盖率) |
| **ALOHA** | `gym_aloha/env.py:180` | **离散阶段奖励+接触传感器**：`reward∈{0..4}`，`is_success=(reward==4)`；接触判据按任务区分(如 `pin_touched`、`touch_left_gripper and not touch_table`) |
| RoboCasa | `info.get("success", False)`(`envs/robocasa.py:282`) | 转发 robosuite/robocasa 内部判定，没装包看不到具体逻辑 |
| VLABench | `task.should_terminate_episode(physics)`(`envs/vlabench.py:496`) | 调用任务对象自己的终止条件方法，具体逻辑在包内部 |
| Isaac Lab Arena | 完全未知 | 远程托管，本地没有任何代码(运行时从 HF Hub 下载 `env.py` 执行) |
| LIBERO-plus | 跟 LIBERO 一样 | 沿用 `_check_success()` |
| RoboTwin | `eval_success` 属性，否则 `check_success()`(`envs/robotwin.py:349-351`) | 转发 RoboTwin 框架自己的判定，具体逻辑在包内部 |
| RoboMME | `status == "success"`(`envs/robomme.py:118`) | 比较字符串字段，具体怎么产生在 ManiSkill/RoboMME 包内部 |

---

## 各环境补充说明 / 来源 / 未装原因

**LIBERO** — `LiberoEnv`(gym 类，`envs/libero.py`)默认 256×256；实际发布 eval 配置(`envs/configs.py`)用 360×360。原始 `robot_state` 字典：`eef.pos`(3)+`eef.quat`(4)+`eef.mat`(3×3，跟quat冗余)+`gripper.qpos`(2)+`gripper.qvel`(2)+`joints.pos`(7)+`joints.vel`(7)；经 `LiberoProcessorStep` 处理后喂给策略的 `observation.state` 是 8 维(eef位置3+轴角朝向3+夹爪qpos2)。action 是 6D末端笛卡尔增量+1D夹爪。语言指令来自 BDDL 文件 `(:language ...)`(`task_description`，`envs/libero.py:172`)。来源：`hf-libero` pip 包(`pyproject.toml [libero]`)。

**MetaWorld** — state `agent_pos=raw_obs[:4]`(`envs/metaworld.py:181`)=末端xyz+夹爪开合度。语言指令来自 `metaworld_config.json` 的 `TASK_DESCRIPTIONS`(50任务)。来源：`metaworld==3.0.0`(`pyproject.toml [metaworld]`)。

**PushT** — action 是**绝对像素坐标目标**(不是增量)，所以"无操作"要让 `action=当前agent_pos`，不能用全零。目标位姿固定在 `[256,256,π/4]`。来源：`gym-pusht`+`pymunk`(2D物理，跟MuJoCo无关，`pyproject.toml [pusht]`)。

**ALOHA** — state/action 都是 14 维同一套关节空间编号：`[waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,gripper]×2`(左右臂，`gym_aloha/constants.py:8-27`)，**直接关节位置控制**，跟 LIBERO/MetaWorld 的末端笛卡尔增量控制本质不同。没有专属 wrapper 类(`lerobot/envs/` 下没有 `aloha.py`)，是 `AlohaEnv` dataclass 直接 `gym.make()` 接 `gym_aloha` 包。来源：`gym-aloha`(`pyproject.toml [aloha]`)。

**RoboCasa** — 相机默认 `robot0_agentview_left`+`robot0_eye_in_hand`+`robot0_agentview_right`。语言指令取自每个 episode 元数据 `ep_meta["lang"]`(`envs/robocasa.py:222`)，同一任务不同episode描述可能不同。**未装原因**：`setup.py` 反向锁 `lerobot==0.3.3`，跟本仓库 workspace 版本循环冲突(见 `docs/source/robocasa.mdx`)。

**VLABench** — 相机 `image`+`second_image`+`wrist_image`。语言指令运行时从任务对象取 `task_description`/`language_instruction`(`envs/vlabench.py:252-257`)，取不到则退化成任务名。**未装原因**：只在 GitHub 发布(`OpenMOSS/VLABench`)，无 PyPI 包(见 `docs/source/vlabench.mdx`)。

**Isaac Lab Arena** — 走 `HubEnvConfig`，本地无实现代码，运行时从 HF Hub `nvidia/isaaclab-arena-envs` 下载 `env.py` 执行 `make_env()`。`task` 字段本身就是指令句(默认 `"Reach out to the microwave and open it."`)。

**LIBERO-plus** — `LiberoPlusEnv` 直接继承 `LiberoEnv`，默认 `is_libero_plus=True`、默认套件 `libero_spatial`。新增 7 种扰动维度：相机视角/物体摆放/机器人初始状态/**语言指令**(同任务多种措辞，文件名带 `_language_X`)/光照/背景纹理/传感器噪声，组合出约1万变体。**未装原因**：非 pip 包，需手动 clone GitHub fork `sylvestf/LIBERO-plus`(见 `docker/Dockerfile.benchmark.libero_plus`)。

**RoboTwin 2.0** — 相机 `head_camera`+`left_camera`+`right_camera`，尺寸匹配D435配置。语言指令只是 `task_name.replace("_"," ")`(`envs/robotwin.py:241`)，非独立标注。**未装原因**：非 pip 包，需手动装并把 `envs/` 加进 `PYTHONPATH`(见 robotwin-platform.github.io)。

**RoboMME** — action 8维(`joint_angle`模式)或7维(`ee_pose`模式)。**未装原因**：ManiSkill 强制锁 `numpy<2`，跟 lerobot 自身 `numpy>=2` 直接冲突，只能在专门Docker镜像装(`docker/Dockerfile.benchmark.robomme`)。

---

## 非仿真：`gym_manipulator`(HILSerlRobotEnvConfig)

跟以上 10 个完全不是一类——接的是**真实机器人**(`robot: RobotConfig`)+ 真实 teleop 设备，给 HIL-SERL(human-in-the-loop 在线强化学习)用，不在仿真器里跑，没有统一的"image尺寸/state维度"规格，完全由接的具体机器人型号决定。

---

## 共性小结

- **state 全部来自仿真器(或真实机器人)内部的精确状态**，不经过视觉感知，跟摄像头画面是同一个物理瞬间的两种不同读出方式
- **is_success 判断范式分四种**：符号谓词逻辑(LIBERO/LIBERO-plus) / 连续距离或角度阈值(MetaWorld) / 几何覆盖面积比(PushT) / 离散阶段+接触传感器(ALOHA)；未实测的几个都是直接转发第三方框架自己算好的字段——**本质都是"读仿真里的某个量，跟标准比较"，没有一个靠模型自己判断或视觉识别**
- **action 语义差异最大**：末端笛卡尔空间(增量或绝对，LIBERO/MetaWorld/RoboTwin/RoboMME的`ee_pose`模式) vs 关节空间绝对位置(ALOHA/RoboTwin的`joint_angle`模式) vs PushT 非归一化绝对像素坐标(独此一家)
- **语言指令有无+来源差异很大**：固定预写句子(LIBERO/MetaWorld/Isaac Lab Arena) / 动态随episode变化(RoboCasa) / 仅任务名转写非真正标注(RoboTwin) / 完全没有(PushT/ALOHA/RoboMME)——这也是为什么不是所有 env 都适合拿来做语言条件的 VLA 训练/评测
