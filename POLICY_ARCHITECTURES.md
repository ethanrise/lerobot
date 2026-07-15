# LeRobot Policy 架构对比笔记

涉及的都是 `lerobot/configs/policies.py` 里 `PreTrainedConfig.register_subclass(...)` 注册的 policy 类型，用于 `lerobot-train`/`lerobot-eval` 的 `--policy.type=...`。

**信息来源说明**：这份笔记纯粹基于读源码(`configuration_<name>.py`/`modeling_<name>.py`/`README.md`)整理，没有实际跑训练验证，所以不用 `ENV_SIMULATORS.md` 里"是否实测"的 ✅/⚠️ 标记。结论都标注了来源文件，数值/参数规模没在代码或文档里明确写出的，统一标"未提及"，不臆测。

**阅读顺序**：① 分类树+独立标签表(先搞清楚"训练范式/生成机制"是切树的两刀，"骨干来源/网络架构"是挂在叶子上的独立标签，不是同一层级的概念) → ② 总对比表(按"动作生成机制"分4组) → ③ 各 policy 补充说明(关键config字段+来源论文) → ④ pi0 / pi0_fast / pi0.5 同源三者细节差异 → ⑤ 共性小结

---

## 1. 分类框架：两刀切树 + 两个独立标签

16 个 policy 容易被误归成"VLA / RL / 扩散 / 离散token / Transformer"5个并列的桶，但这5个词分散在**4个互相独立的维度**里，混在一起比较才会理不清楚。从"为什么训练"到"具体怎么算"，依次是：

```
Layer 0  训练范式(监督信号从哪来)        —— Imitation Learning  vs  Reinforcement Learning
Layer 1  (仅IL内)动作生成机制            —— 单步回归 / 扩散+流匹配(同源) / 离散token
Layer 2  骨干初始化来源(独立标签)        —— 经典骨干(从零/仅ImageNet预训练)  vs  VLA骨干(复用预训练VLM)
Layer 3  网络层计算方式(独立标签)        —— CNN(卷积)  vs  Transformer(self-attention)
```

Layer 0→1 是**切树的两刀**，决定每个policy落在哪个分支；Layer 2、Layer 3 是**挂在每个叶子节点上的独立标签**，同一个生成机制分支下可以同时存在"经典骨干"和"VLA骨干"、"CNN"和"Transformer"的policy，不参与切树。

### 1.1 分类树(Layer 0 + Layer 1)

```
16 个 policy.type
│
├─ Imitation Learning(模仿学习，14个)
│   ├─ 单步回归
│   │   └─ act
│   ├─ 扩散 / 流匹配(同源：都是多步迭代去噪生成连续动作，区别只在离散时间步还是连续ODE)
│   │   ├─ diffusion
│   │   ├─ multi_task_dit
│   │   ├─ pi0
│   │   ├─ pi05
│   │   ├─ smolvla
│   │   ├─ groot
│   │   ├─ xvla
│   │   ├─ wall_x(默认模式)
│   │   ├─ eo1
│   │   ├─ molmoact2(continuous模式)
│   │   └─ vla_jepa
│   └─ 离散token(量化后自回归解码 / codebook查表)
│       ├─ vqbet
│       ├─ pi0_fast
│       ├─ wall_x(prediction_mode="fast"时)
│       └─ molmoact2(discrete模式)
│
└─ Reinforcement Learning(强化学习，2个；不再细分"回归/扩散/离散token"——
    它们"怎么出动作"是规划/高斯采样，是RL自己的机制)
    ├─ tdmpc(基于模型RL + CEM在线规划)
    └─ gaussian_actor(SAC算法的可插拔actor组件)
```

### 1.2 独立标签表(Layer 2 + Layer 3)

骨干来源和网络架构是两个正交的标签，挂在上面每个policy身上。这张表顺便回答了"是不是都是transformer"——**不是**，`diffusion`的去噪网络是1D卷积U-Net，`tdmpc`/`gaussian_actor`是纯CNN+MLP；而且VLA家族内部"骨干"和"动作头"的架构也可能不同(`eo1`/`wall_x`骨干是Transformer，但动作头退化成了纯MLP，没有self-attention)：

| type | 骨干来源 | 骨干架构 | 动作头架构 |
|---|---|---|---|
| `act` | 经典 | Transformer(整体即Transformer，无独立骨干/头划分) | — |
| `diffusion` | 经典 | CNN(ResNet18视觉编码) | **CNN**(`DiffusionConditionalUnet1d`，1D卷积U-Net) |
| `multi_task_dit` | 经典(仅CLIP，非大型VLM) | Transformer(CLIP视觉+文本双塔) | Transformer(DiT) |
| `vqbet` | 经典 | CNN(ResNet18视觉编码) | Transformer(miniGPT) |
| `pi0` | VLA | Transformer(PaliGemma) | Transformer(`gemma_300m` action expert，跟主干联合attention) |
| `pi05` | VLA | Transformer(PaliGemma) | Transformer(同pi0) |
| `smolvla` | VLA | Transformer(SmolVLM2) | Transformer(`SmolVLMWithExpertModel`，有自己的`self_attn`层) |
| `groot` | VLA | Transformer(Eagle2-VLM+Llama) | Transformer(DiT，`flow_matching_action_head.py`) |
| `xvla` | VLA | Transformer(Florence2) | Transformer(`SoftPromptedTransformer`，显式`Attention`+`TransformerBlock`) |
| `wall_x` | VLA | Transformer(Qwen2.5-VL MoE) | **MLP**(`ActionHead`类纯`nn.Linear`堆叠，无self-attention) |
| `eo1` | VLA | Transformer(Qwen2.5-VL) | **MLP**(`EO1VisionActionProjector`，`nn.Sequential`纯线性层) |
| `molmoact2` | VLA | Transformer(Molmo) | Transformer(action_expert，含`self_attn`+RoPE) |
| `vla_jepa` | VLA | Transformer(Qwen3-VL) | Transformer(DiT-B) |
| `pi0_fast` | VLA | Transformer(PaliGemma) | 复用主干自回归解码，没有独立动作头 |
| `tdmpc` | 经典 | CNN+MLP | CNN+MLP(规划，非生成式动作头) |
| `gaussian_actor` | 经典 | CNN+MLP | — |

一句话总结这张表：**Transformer几乎是VLA家族的标配(骨干必然是Transformer，因为预训练VLM本身就是Transformer)，但"动作头"未必跟着用Transformer**——`wall_x`和`eo1`偏偏选了最简单的MLP投影层去接流匹配的velocity预测，没有再加一层attention。

---

## 2. 总对比表(按动作生成机制分组)

### 2.1 单步回归式BC

| type | VLA骨干 | 骨干网络 | 语言指令 | chunk_size / n_action_steps | 生成方式 | 损失类型 | 图像输入 |
|---|---|---|---|---|---|---|---|
| `act` | 否 | ResNet18(ImageNet预训练) | 无 | 100 / 100 | 单次前向回归(+可选CVAE采样) | L1 + KL(VAE) | 多摄像头，要求各路shape一致 |

### 2.2 扩散 / 流匹配(连续动作，本质同源)

| type | VLA骨干 | 骨干网络(预训练checkpoint) | 语言指令 | chunk_size / n_action_steps | 生成方式 | 损失类型 | 图像输入 |
|---|---|---|---|---|---|---|---|
| `diffusion` | 否 | ResNet18(ImageNet预训练) | 无 | horizon=64 / 32 | 迭代去噪(DDPM/DDIM，训练100步) | 噪声预测MSE | 多摄像头，可配置resize/crop |
| `multi_task_dit` | 否 | CLIP-ViT-base-patch16(视觉+文本双塔) | 必需 | horizon=32 / 24 | `objective`可选`diffusion`(DDPM 100步)或`flow_matching`(100步积分) | 扩散噪声MSE 或 流匹配loss | 多摄像头，可配置resize/crop |
| `pi0` | 是 | PaliGemma(`gemma_2b`) + Gemma动作专家(`gemma_300m`) | 必需，tokenizer_max_length=48 | 50 / 50 | Flow matching ODE反向积分，10步 | 流匹配MSE | 多摄像头，224×224，支持空镜头 |
| `pi05` | 是 | 同pi0 | 必需，tokenizer_max_length=200 | 50 / 50 | 同pi0，10步ODE，AdaRMS时间条件化 | 流匹配MSE | 同pi0 |
| `smolvla` | 是 | SmolVLM2-500M-Video-Instruct(`HuggingFaceTB`) | 必需，tokenizer_max_length=48 | 50 / 50 | Flow matching，10步(`num_steps`) | 流匹配MSE | 多摄像头，resize到512×512后归一化到[-1,1] |
| `groot` | 是 | Eagle2-VLM(视觉)+Llama(语言)，来自`nvidia/GR00T-N1.5-3B` | 支持但非强制(缺省退化为"Perform the task.") | 50 / 50 | DiT流匹配去噪(`flow_matching_action_head.py`) | 流匹配loss(DiT内部) | 至少1个视觉特征，224×224 |
| `xvla` | 是 | Florence2 + BART tokenizer(`facebook/bart-large`) | 必需 | 32 / 32 | ODE求解器去噪，`num_denoising_steps=10` | 流匹配loss | 至少1个视觉特征，多视角+空镜头padding |
| `wall_x` | 是 | Qwen2.5-VL MoE，来自`x-square-robot/wall-oss-flow` | 必需 | 32 / 32 | Beta噪声流匹配(`torchdiffeq.odeint`)，默认模式(`prediction_mode="diffusion"`) | 流匹配loss + 交叉熵(LM部分) | 至少1个视觉特征，智能缩放+padding |
| `eo1` | 是 | Qwen2.5-VL-3B-Instruct | 必需，dataset `task`字段 | 8 / 8 | Beta时间采样+去噪，`num_denoise_steps=10` | 流匹配MSE | 至少1个视觉特征，走Qwen`image_grid_thw` |
| `molmoact2`(continuous模式) | 是 | Molmo，来自`allenai/MolmoAct2` | 必需，`task`字段 | 30 / 30 | `action_mode="continuous"`时走流匹配 | 流匹配loss + 辅助loss | 至少1个视觉特征，多镜头 |
| `vla_jepa` | 是 | Qwen3-VL-2B-Instruct + V-JEPA2世界模型(`facebook/vjepa2-vitl-fpc64-256`，仅训练辅助) | 必需，`lang`字段 | 7 / 7 | DiT-B流匹配，Beta时间采样 | 流匹配loss + 世界模型L1 loss(权重0.1，仅训练用) | 需要video帧(默认8帧)+图像 |

### 2.3 离散token(量化动作)

| type | VLA骨干 | 骨干网络 | 语言指令 | 关键动作参数 | 生成方式 | 损失类型 | 图像输入 |
|---|---|---|---|---|---|---|---|
| `vqbet` | 否 | ResNet18(ImageNet预训练) | 无 | n_action_pred_token=3 × action_chunk_size=5 = 15步 | 两阶段：先训残差VQ-VAE码本，再训GPT自回归预测codebook token | 重构L1(VQ-VAE阶段) + 交叉熵(GPT主/次码) + offset回归 | 单张图像，crop到84×84 |
| `pi0_fast` | 是 | PaliGemma(`gemma_2b`)+Gemma专家(`gemma_300m`) + FAST tokenizer(`lerobot/fast-action-tokenizer`) | 必需，tokenizer_max_length=200 | chunk_size=50/50，max_action_tokens=256 | 自回归贪心解码(temperature=0)，支持KV cache加速 | 交叉熵(token预测) | 同pi0，224×224 |
| `wall_x`(fast模式) | 是 | 同上(`prediction_mode="fast"`时) | 必需 | 32 / 32 | 切到FAST离散tokenizer(同`action_tokenizer_path`机制) | 交叉熵 | 同上 |
| `molmoact2`(discrete模式) | 是 | 同上(`action_mode="discrete"`或`"both"`时) | 必需 | 30 / 30 | tokenizer离散解码 | 交叉熵(离散token) + 辅助loss | 同上 |

### 2.4 RL(非imitation learning)

| type | VLA骨干 | 骨干网络 | 语言指令 | 关键动作参数 | 生成方式 | 损失类型 | 图像输入 |
|---|---|---|---|---|---|---|---|
| `tdmpc` | 否 | 自定义CNN+MLP(随机初始化) | 无 | horizon=5 / 1(n_action_repeats=2) | CEM在线规划(Gaussian采样+策略采样混合) | 动力学损失+奖励损失+TD值损失+策略损失+一致性损失，多项加权 | 单张方形图像，**不支持多摄像头** |
| `gaussian_actor` | 否 | 自定义CNN+MLP(可选冻结视觉encoder) | 无 | 单步动作，无chunk概念 | 对角高斯采样 + 可选tanh squash(`use_tanh_squash`) | **不在policy内部算loss**，由`src/lerobot/rl/algorithms/sac/sac_algorithm.py`的SAC actor梯度负责(直接import `GaussianActorPolicy`) | 可选(支持纯state输入)，视觉encoder可冻结 |

---

## 3. 各 policy 补充说明(关键config字段 + 来源论文)

**`act`** —— 关键字段：`chunk_size=100`、`use_vae=True`、`kl_weight=10.0`、`n_encoder_layers=4`、`dim_model=512`。来源：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*(ALOHA, arXiv:2304.13705, Zhao et al.)。

**`diffusion`** —— 关键字段：`horizon=64`、`num_train_timesteps=100`、`num_inference_steps=None`(默认等于训练步数)、`noise_scheduler_type="DDPM"`、`prediction_type="epsilon"`。来源：*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*(arXiv:2303.04137, Chi et al.)。

**`multi_task_dit`** —— 关键字段：`objective="diffusion"`、`noise_scheduler_type="DDPM"`、`num_train_timesteps=100`、`hidden_dim=512`、`num_layers=6`、`vision_encoder_name="openai/clip-vit-base-patch16"`。来源：*A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*(arXiv:2507.05331, TRI LBM Team) + Bryson Jones 的拆解博客。

**`pi0`** —— 关键字段：`num_inference_steps=10`、`paligemma_variant="gemma_2b"`、`action_expert_variant="gemma_300m"`、`max_state_dim=32`、`max_action_dim=32`，归一化方式`MEAN_STD`。来源：*π₀: A Vision-Language-Action Flow Model for General Robot Control*(arXiv:2410.24164, Black et al., Physical Intelligence)。

**`pi05`** —— 关键字段同pi0，但`tokenizer_max_length=200`、归一化方式改为`QUANTILES`、用AdaRMS做时间条件化。来源：*π₀.₅: a Vision-Language-Action Model with Open-World Generalization*(arXiv:2504.16054, Physical Intelligence, 2025)。

**`smolvla`** —— 关键字段：`vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`、`num_steps=10`、`resize_imgs_with_padding=(512,512)`、`train_expert_only=True`、`freeze_vision_encoder=True`。来源：*SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*(arXiv:2506.01844, Shukor et al., Hugging Face, 2025)。

**`groot`** —— 关键字段：`base_model_path="nvidia/GR00T-N1.5-3B"`、`tune_diffusion_model=True`、`use_bf16=True`、`max_state_dim=64`。来源：*GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*(arXiv:2503.14734, NVIDIA, 2025)。参数规模：3B(GR00T-N1.5-3B checkpoint)。

**`xvla`** —— 关键字段：`action_mode="ee6d"`、`num_denoising_steps=10`、`use_proprio=True`、`max_action_dim=20`、`num_image_views`(支持多视角)。来源README未给出独立论文链接。

**`wall_x`** —— 关键字段：`pretrained_name_or_path="x-square-robot/wall-oss-flow"`、`prediction_mode="diffusion"`(可切`"fast"`)、`action_tokenizer_path="lerobot/fast-action-tokenizer"`、`max_action_dim=20`。来源：*WALL-OSS*(arXiv:2509.11766, X-Square Robot, 2025)。

**`eo1`** —— 关键字段：`vlm_base="Qwen/Qwen2.5-VL-3B-Instruct"`、`num_denoise_steps=10`、`max_state_dim=32`、`max_action_dim=32`、`supervise_padding_action_dims=True`。来源：*EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control*(arXiv:2508.21112)。参数规模：3B(Qwen2.5-VL-3B-Instruct)。

**`molmoact2`** —— 关键字段：`checkpoint_path="allenai/MolmoAct2"`、`action_mode="both"`、`num_flow_timesteps=8`、`expected_max_action_dim=32`。来源：[MolmoAct2](https://allenai.org/blog/molmoact2)(Allen AI)；LIBERO评测有专门提醒：`num_steps_wait`需设为50而非默认10，否则场景没稳定会拉低成功率。

**`vla_jepa`** —— 关键字段：`qwen_model_name="Qwen/Qwen3-VL-2B-Instruct"`、`jepa_encoder_name="facebook/vjepa2-vitl-fpc64-256"`、`enable_world_model=True`、`num_video_frames=8`、`world_model_loss_weight=0.1`。来源：[ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) 的LeRobot移植版；推理时只用Qwen骨干+动作头，世界模型只在训练时提供辅助loss。

**`vqbet`** —— 关键字段：`n_vqvae_training_steps=20000`、`vqvae_n_embed=16`、`vqvae_embedding_dim=256`、`gpt_n_layer=8`、`gpt_block_size=500`。来源：*Behavior Generation with Latent Actions*(VQ-BeT, arXiv:2403.03181, Lee et al.)。

**`pi0_fast`** —— 关键字段：`max_action_tokens=256`、`action_tokenizer_name="lerobot/fast-action-tokenizer"`、`tokenizer_max_length=200`、`temperature=0.0`、`use_kv_cache=True`。来源：OpenPI框架下π₀的离散token变体，未单独成文。

**`tdmpc`** —— 关键字段：`horizon=5`、`use_mpc=True`、`cem_iterations=6`、`n_gaussian_samples=512`、`n_pi_samples=51`。来源：*Temporal Difference Learning for Model Predictive Control*(arXiv:2203.04955) + *Finetuning Offline World Models in the Real World*(FOWM, arXiv:2310.16029)。

**`gaussian_actor`** —— 关键字段：`use_tanh_squash=True`、`std_min=1e-5`、`std_max=10.0`、`shared_encoder=True`、`freeze_vision_encoder=True`。无独立README，本身是给`src/lerobot/rl/algorithms/sac/`用的actor组件，不是端到端的imitation policy。

---

## 4. pi0 / pi0_fast / pi0.5：同源三者的关键差异

三者共享PaliGemma(`gemma_2b`)+Gemma动作专家(`gemma_300m`)的骨干结构，但动作表示和条件化方式不同：

| 特性 | `pi0` | `pi0_fast` | `pi05` |
|---|---|---|---|
| 动作表示 | 连续，flow matching | **离散token**，自回归 | 连续，flow matching |
| 时间条件化 | 时间与动作concat，过`action_time_mlp_*` | 无显式时间条件(自回归解码不需要) | AdaRMS(Adaptive RMS)时间条件化 |
| 推理方式 | ODE反向积分(`x_t += dt·v_t`)，10步 | 自回归token循环解码，`max_decoding_steps=256` | 同pi0，ODE 10步 |
| 语言tokenizer长度 | 48 | 200(+最多256个action token) | 200 |
| 归一化方式 | `MEAN_STD` | `MEAN_STD` | **`QUANTILES`** |
| KV cache | 否 | **是**(`use_kv_cache=True`，加速自回归解码) | 否 |
| 损失类型 | 流匹配MSE | 交叉熵(token预测) | 流匹配MSE |

一句话区分：**pi0** 是基础的流匹配VLA；**pi0_fast** 把连续动作离散化成token、用自回归解码换取推理工程上的简化(可复用LLM那套KV cache基础设施)；**pi0.5** 是pi0的迭代版，换了归一化方式和时间条件化机制，主打更强的开放世界泛化。

---

## 5. 共性小结

- **4个维度不要混着比**：训练范式(IL/RL)、动作生成机制(回归/扩散流匹配/离散token)、骨干来源(经典/VLA)、网络架构(CNN/Transformer)是相互独立的4层，"VLA"和"Transformer"都不能跟"扩散/离散token"并列成同一层的分类——前两者是挂在叶子节点上的标签，后者是切树的刀。详见第1节的树+标签表。
- **架构演进的分水岭很清楚**：`act`(2023)→`diffusion`/`vqbet`/`tdmpc` 这一批是"经典"行为克隆/规划时代，骨干都是从零训练或仅ImageNet预训练的小型CNN，**不需要语言**；`pi0`系列起(2024年后)进入"VLA"时代，**直接复用现成的开源视觉语言模型**(PaliGemma/SmolVLM2/Qwen2.5-VL/Qwen3-VL/Florence2/Molmo/Eagle2)做骨干，再接一个轻量动作头微调，**几乎全部需要语言指令**——因为骨干本身就是VLM，不给语言等于浪费了它的核心能力。
- **动作生成机制本质是三分天下，不是四分**：① 单步连续回归(`act`，配CVAE缓解多模态分布问题，最快但表达力较弱)；② 扩散与流匹配同源——都是多步迭代去噪生成连续动作，区别只在离散时间步(DDPM)还是连续ODE(flow matching)；③ 离散token自回归(VQ codebook查表或FAST tokenizer，能复用LLM的KV cache等成熟自回归基础设施，但要先解决动作tokenizer设计问题)；RL(`tdmpc`/`gaussian_actor`)是单独的训练范式，不属于这三种"生成机制"的讨论范畴。
- **VLA骨干必然是Transformer，但动作头未必**：预训练VLM本身就是Transformer，所以VLA家族的"骨干"清一色Transformer；可一到"动作头"这一层就分裂了——`pi0`/`pi05`/`smolvla`/`xvla`/`groot`/`molmoact2`/`vla_jepa`都另起一个小Transformer(action expert/DiT)去预测velocity，但`wall_x`和`eo1`偏偏只用了纯`nn.Linear`堆出来的MLP头，没有额外attention。说明"动作头要不要上Transformer"是设计者自己的取舍，不是VLA的必选项。
- **chunk_size 在新一代VLA里普遍变小**：经典`act`用100，`pi0`系列用50，`wall_x`/`xvla`用32，`eo1`/`vla_jepa`只有7~8——更短的chunk配合更高频率的重新规划，在大模型VLA上比"一次性吐一大段动作"更常见。
- **`gaussian_actor` 是唯一的例外**：它不是端到端的imitation learning policy，而是 SAC 等RL算法的可插拔actor组件(loss不在policy内部算，由 `sac_algorithm.py` 的算法逻辑负责)，所以"动作生成方式"对它而言是单步高斯采样，没有chunk概念。
- **图像输入约束差异很大**：`tdmpc`/`vqbet` 受限于自身小骨干，只支持单张(方形/裁剪)图像；VLA家族普遍能吃多摄像头、支持空镜头padding，这是大模型骨干带来的灵活性。
