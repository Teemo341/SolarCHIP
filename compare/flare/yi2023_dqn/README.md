# A1 — Yi 2023 DenseNet-DQN 调研与复现

状态：**已实现可训练的 PyTorch Lightning 版本；论文事实与 SolarCHIP 工程默认仍按证据标签分开。**

核心来源：K. Yi, Y.-J. Moon, and H.-J. Jeong, “Application of Deep Reinforcement Learning to Major Solar Flare Forecasting,” *The Astrophysical Journal Supplement Series*, 265:34, 2023，[DOI `10.3847/1538-4365/acb76d`](https://doi.org/10.3847/1538-4365/acb76d)。截至本次核验，未找到可固定 commit 的作者实现，因此结构和训练事实只按论文记录。

证据标签含义见[上级 README](../README.md)。

## 1. 原始问题与数据协议

- **[论文原文]** 每个样本使用当天 `00:00 UT` 的一张全日面 line-of-sight magnetogram，预测同一天是否发生 major flare。
- **[论文原文]** 原始标签是二分类：当天最大耀发达到 `M1.0+` 为 positive，C 级及以下为 negative；原文没有 `0AB/C/M/X` 四分类实验。
- **[论文原文]** 太阳活动周 23 的训练集由 SOHO/MDI 组成，时间为 1996-05 至 2008-12，共 3,914 日：X/M/C/<C 分别为 `93/647/1478/1696`，即 positive/negative 为 `740/3174`。
- **[论文原文]** 测试覆盖太阳活动周 24：2009-01 至 2010-12 的 405 张 MDI 和 2011-01 至 2019-12 的 3,087 张 HMI，共 3,492 日；X/M/C/<C 为 `40/360/1240/1852`，即 positive/negative 为 `400/3092`。
- **[论文原文]** 划分按时间和太阳活动周完成，不是随机交叉验证；作者另对代表性的 Model 3 用 10 个不同随机种子重复训练以估计不确定性。
- **[SolarCHIP 适配]** 主比较不使用上述 MDI/HMI 数据和论文 split，只复用项目现有 HMI-only DataModule、日期、标签与 split。
- **[SolarCHIP 适配]** 当前目标仍是 D 日 00:00 HMI 对 D 日最高 `0AB/C/M/X`，而不是另建论文的 M1+ 数据集。

## 2. 图像预处理

- **[论文原文]** MDI 和 HMI 全日面图都通过 block averaging 统一为 `512 x 512`。
- **[论文原文]** MDI 磁场按 Liu et al. (2012) 的仪器标定关系转换到 HMI proxy 后再与 HMI 合并。
- **[论文原文]** 磁场范围裁剪到约 `[-100 G, +100 G]`，再映射为 byte image；这一压缩会饱和强磁场。
- **[SolarCHIP 适配]** 当前 dataloader 已给出 signed-`log1p` + z-score 的 `[B,1,1024,1024]`，不重新执行 MDI 标定、`±100 G` 裁剪或 byte 映射。
- **[SolarCHIP 适配]** 模型入口只使用 area interpolation 把 `1024 x 1024` 缩到 `512 x 512`。因此复现的是论文的网络尺度，而不是论文的像素值分布。

## 3. 论文中的精确 DenseNet

- **[论文原文]** 网络以 `3 x 3` convolution、26 个通道开始，随后是 `2 x 2` max pooling。
- **[论文原文]** 主干包含五个 dense block。第 `n` 个 block 依次执行 BN、ReLU、`1 x 1` convolution（`13n` 通道）、BN、ReLU、`3 x 3` convolution（39 通道），再把这 39 个新通道与 block 输入拼接。
- **[论文原文]** 每个 dense block 后使用 `2 x 2` average pooling。第五个 block 后再执行最后的 BN 与 `2 x 2` average pooling，flatten 后接两维 fully connected 输出。

按论文图示，`512 x 512` 输入的形状合同为：

| 阶段 | 操作 | 输出形状（不含 batch） | 证据 |
| --- | --- | --- | --- |
| stem | `Conv3x3, D=26`；`MaxPool2` | `26 x 256 x 256` | **[论文原文]** |
| dense block 1 | `1x1:13`；`3x3:39`；concat；`AvgPool2` | `65 x 128 x 128` | **[论文原文]** |
| dense block 2 | `1x1:26`；`3x3:39`；concat；`AvgPool2` | `104 x 64 x 64` | **[论文原文]** |
| dense block 3 | `1x1:39`；`3x3:39`；concat；`AvgPool2` | `143 x 32 x 32` | **[论文原文]** |
| dense block 4 | `1x1:52`；`3x3:39`；concat；`AvgPool2` | `182 x 16 x 16` | **[论文原文]** |
| dense block 5 | `1x1:65`；`3x3:39`；concat；`AvgPool2` | `221 x 8 x 8` | **[论文原文]** |
| tail | BN；`AvgPool2`；flatten；FC | `221 x 4 x 4 -> 3536 -> 2` | **[论文原文]** |

- **[SolarCHIP 适配]** 后续实现应逐层断言上表形状；不得用 torchvision 的通用 DenseNet 替代该特定结构后仍称为结构复现。
- **[SolarCHIP 适配]** 为三个累计任务共享到 `3536` 维主干，只把最后 FC 扩展为三个独立两动作 head，最终 Q 张量为 `[B,3,2]`。

## 4. DQN、Double-DQN 与 reward 搜索

- **[论文原文]** 一个磁图被视为 state，网络在 `major flare / no major flare` 两个 action 中选择；根据 action 与真实类别的 TP/FP/FN/TN 情况获得 reward。
- **[论文原文]** DQN 使用 online network 与 target network、experience replay 和 epsilon-greedy exploration；论文同时比较 vanilla DQN 与 Double-DQN。
- **[论文原文]** TD objective 使用 mean squared error；DQN 优化器为 RMSprop。作为监督对照的 CNN 使用 Adam 和 MSE。
- **[论文原文]** target network 每 1 epoch 或每 2 epochs 同步一次。
- **[论文原文]** 论文组合 16 个 reward、DQN/Double-DQN 两种算法和两种 target 同步周期，共 64 个强化学习设置。

16 组 reward 如下，列顺序固定为 `TP / FP / FN / TN`：

| # | TP | FP | FN | TN | 证据 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | -1 | -1 | 1 | **[论文原文]** |
| 2 | 4 | -2 | -8 | 1 | **[论文原文]** |
| 3 | 4 | -4 | -16 | 1 | **[论文原文]** |
| 4 | 4 | -16 | -32 | 1 | **[论文原文]** |
| 5 | 4 | -16 | -64 | 1 | **[论文原文]** |
| 6 | 8 | -4 | -16 | 1 | **[论文原文]** |
| 7 | 8 | -4 | -32 | 1 | **[论文原文]** |
| 8 | 8 | -8 | -16 | 1 | **[论文原文]** |
| 9 | 8 | -8 | -64 | 1 | **[论文原文]** |
| 10 | 8 | -16 | -8 | 1 | **[论文原文]** |
| 11 | 8 | -16 | -32 | 1 | **[论文原文]** |
| 12 | 8 | -16 | -64 | 1 | **[论文原文]** |
| 13 | 8 | -32 | -16 | 1 | **[论文原文]** |
| 14 | 4 | -64 | -4 | 1 | **[论文原文]** |
| 15 | 4 | -64 | -8 | 1 | **[论文原文]** |
| 16 | 8 | -96 | -8 | 1 | **[论文原文]** |

- **[论文原文]** 论文报告的 DQN 最佳代表配置采用 `TP/FP/FN/TN = 8/-64/-8/1`，target network 每 2 epochs 同步；其表中 HSS/F1/TSS/ApSS 为 `0.44/0.51/0.59/0.10`。
- **[论文原文]** 论文摘要层面的最佳强化学习结果相对监督 CNN 把 HSS 从 0.38 提高到 0.44、F1 从 0.47 提高到最高约 0.52、TSS 从 0.53 提高到 0.59、ApSS 从 0.09 提高到最高约 0.12；不同指标的最佳值不一定来自同一个配置。
- **[SolarCHIP 适配]** A1 默认锁定 vanilla DQN、`8/-64/-8/1` 和每 2 epochs 同步，不默认使用 Double-DQN。
- **[SolarCHIP 适配]** 同一个 reward 表会分别作用于 `C+ / M+ / X+` 三个 head。这是输出适配，尤其对极少数 X 样本的统计行为没有论文依据，必须单独报告每头结果。

## 5. 训练与原始评估

- **[论文原文]** DQN 训练 100 epochs，在一张 NVIDIA RTX 2080 上约 28 小时；监督 CNN 训练 250 epochs，约 4 小时。
- **[论文原文]** 评价指标为 Heidke skill score（HSS）、F1、true skill statistic（TSS）和 Appleman skill score（ApSS）。
- **[论文原文]** 64 个设置用于筛选代表模型；论文只明确对 Model 3 进行 10 次不同随机种子的重复训练，并报告 HSS/F1/TSS/ApSS 为 `0.42±0.02 / 0.49±0.01 / 0.58±0.02 / 0.10±0.01`。
- **[论文原文]** 论文在测试期结果上搜索/选择训练 epoch 与 prediction threshold；这使其汇报值带有 test-set model-selection 成分，不能直接作为严格盲测基准。
- **[SolarCHIP 适配]** SolarCHIP 只允许用 validation 选择 checkpoint；累计头用 Q 值 `argmax` 的固定动作解码，不在未来 test 上调阈值。
- **[SolarCHIP 适配]** validation 和未来 test 记录四分类 accuracy；同时对 C+/M+/X+ 三头分别记录 confusion counts、HSS、F1、TSS、ApSS。

## 6. 官方代码审计、原文未报告与复现阻断项

下列参数在论文中不足以唯一重建训练；在获得作者材料前必须保留为未知：

- **[原文未报告]** experience replay buffer 容量、替换策略和是否分层采样。
- **[原文未报告]** discount factor `gamma`。
- **[原文未报告]** epsilon-greedy 的初值、终值、衰减函数与衰减步数。
- **[原文未报告]** minibatch size、replay warm-up 大小和每个环境 step 的 gradient update 次数。
- **[原文未报告]** RMSprop learning rate、momentum/alpha、epsilon 与 weight decay。
- **[原文未报告]** 训练样本的状态转移顺序、episode 起止、terminal flag 与跨 epoch 转移语义。
- **[原文未报告]** 随机种子的具体数值、参数初始化细节和完整 checkpoint-selection 程序。
- **[原文未报告]** 论文没有给出代码仓库或 checkpoint 链接；截至本次调研也未核验到可固定 commit 的作者实现，因此不存在可执行的官方代码对照。

这些未知项不能从“常见 DQN 默认值”推断。后续实现若必须给值，应逐项标成 **[SolarCHIP 适配]** 并进入实验配置，而不是补写为论文设置。

## 7. SolarCHIP 输出与训练合同

### 7.1 Forward 与解码

- **[SolarCHIP 适配]** 输入：现有 `batch['hmi']`，形状 `[B,1,1024,1024]`。
- **[SolarCHIP 适配]** 模型内 area resize：`[B,1,512,512]`。
- **[SolarCHIP 适配]** 共享主干输出三个两动作 Q head：`q.shape == [B,3,2]`，head 顺序固定为 C+/M+/X+，action 顺序固定为 negative/positive。
- **[SolarCHIP 适配]** 每头通过 `argmax(q_head)` 得到二值动作；四分类取最高阳性 head：X+ 为 X，否则 M+ 为 M，否则 C+ 为 C，否则为 0AB。
- **[SolarCHIP 适配]** 需要额外统计非单调动作模式，如 M+ 阳性但 C+ 阴性；只记录，不做投影或静默修复。

### 7.2 DQN 适配边界

- **[SolarCHIP 适配]** 三个 head 共享视觉表示，但分别形成 TD loss；默认 reward 矩阵与 target 同步周期按上文锁定。
- **[SolarCHIP 适配]** 整体训练与指标已经封装为 `Yi2023DQN`，直接读取现有 batch 字典，并使用 Lightning manual optimization 执行 replay 更新。
- **[SolarCHIP 适配]** 论文未报告的 replay、`gamma`、epsilon、batch 和 optimizer 数值采用下文明确列出的工程默认；这些值不是论文设置。
- **[SolarCHIP 适配]** 训练日志必须区分 reward、TD loss、每头 action rate、每头指标与最终四分类 accuracy，避免只看 reward 判断模型优劣。
- **[SolarCHIP 适配]** DQN 是代价敏感分类适配，不把每日样本解释为具有物理控制含义的序列决策。

## 8. 论文协议与项目协议差异

| 项目 | 论文 | SolarCHIP 计划 | 证据 |
| --- | --- | --- | --- |
| 仪器 | MDI 训练为主，测试跨 MDI/HMI | HMI-only | **[论文原文] / [SolarCHIP 适配]** |
| 输入值 | `±100 G` clip、byte image | 现有 signed-`log1p` + z-score | **[论文原文] / [SolarCHIP 适配]** |
| 输出 | M1+ 二分类 | C+/M+/X+ 三个 DQN head，解码四类 | **[论文原文] / [SolarCHIP 适配]** |
| 数据切分 | cycle 23 train、cycle 24 test | 现有项目 split | **[论文原文] / [SolarCHIP 适配]** |
| 选择规则 | 测试期参与 epoch/threshold 搜索 | validation 选 checkpoint，固定 argmax | **[论文原文] / [SolarCHIP 适配]** |
| 框架 | 论文列出 PyTorch 环境 | PyTorch Lightning 包装 | **[论文原文] / [SolarCHIP 适配]** |

## 9. 实现后的验收建议

- **[SolarCHIP 适配]** 用 forward hook 核对 `26/65/104/143/182/221` 通道和最终 `3536` 维 flatten，防止 padding/pooling 偏一。
- **[SolarCHIP 适配]** 对四个标签各构造一个样本，验证三个累计 target 和最高阳性解码；再覆盖 `C-=0,M+=1` 等不一致动作。
- **[SolarCHIP 适配]** 验证 target network 参数只在设定的 epoch 边界同步，且不参与梯度。
- **[SolarCHIP 适配]** 固定一个小 replay fixture，分别核对 vanilla DQN 和 Double-DQN 的 target 公式；默认训练只启用前者。
- **[SolarCHIP 适配]** 在一个真实 DataModule batch 上完成 train/validation smoke test，确认四分类 accuracy 和三组论文指标均可更新。
- **[SolarCHIP 适配]** 当前没有 test split，只验收 `test_step/test_accuracy` 的代码路径可构造，不产生测试数值。

## 10. 已锁定选择

1. **[SolarCHIP 适配]** C+/M+/X+ 三个 head 统一使用 `TP/FP/FN/TN = 8/-64/-8/1`。
2. **[SolarCHIP 适配]** 使用 vanilla DQN，target network 每两个 epoch 同步。
3. **[SolarCHIP 适配]** transition 遵循现有 DataLoader 迭代顺序，跨 batch 连续，epoch 最后一个样本为 terminal；不修改 DataLoader。
4. **[SolarCHIP 适配]** 同时提供普通监督三头 control，用于分离结构与 DQN reward 的收益。
5. **[SolarCHIP 适配]** 正式实验使用一个随机种子；checkpoint 以 validation macro-F1 选择。

## 11. 可训练实现与工程默认

实现文件：

- `architecture.py`：论文特定的 dense concat 主干及累计标签/解码。
- `replay.py`：定长 CPU replay ring、独立采样 RNG 和 checkpoint 状态。
- `metrics.py`：四分类 accuracy/macro-F1，以及每头 F1/TSS/HSS/ApSS。
- `module.py`：Lightning DQN、target sync、epsilon-greedy、manual optimization、监督 control 和预测接口。
- `test_module.py`：结构、解码、replay/RNG 恢复、episode 语义及监督 loss 自检。

论文没有报告下列数值。为使基线可直接训练，默认采用固定的 **[SolarCHIP 适配]** 工程配置：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `learning_rate` | `1e-4` | 面对绝对值 64 的 reward 采用保守学习率；DQN 使用论文指定的 RMSprop，监督 control 使用 Adam |
| `weight_decay` | `0` | 原文未报告 |
| `gamma` | `0.99` | 常用 vanilla-DQN 工程起点；原文未报告，不能视为论文值 |
| epsilon | `1.0 -> 0.05` | 按已观察样本数线性衰减 |
| `epsilon_decay_steps` | `20000` | 约四个现有训练集 epoch；原文未报告 |
| `replay_capacity` | `128` | 512² float16 state/next-state 约占 128 MiB；原文未报告 |
| `replay_batch_size` | `8` | 与当前 A1 DataLoader batch 对齐；原文未报告 |
| `replay_warmup` | `32` | 四个 replay minibatch 后开始更新；原文未报告 |
| `gradient_updates_per_batch` | `1` | 每个 DataLoader batch 一次 replay update；原文未报告 |
| RMSprop `alpha/eps/momentum` | `0.99/1e-8/0` | PyTorch 工程默认；原文未报告 |
| `checkpoint_replay` | `true` | 保存 replay、pending transition、探索 RNG、采样 RNG 和环境步数 |
| `seed` | `42` | 唯一正式种子；与训练入口的全局 seed 同时固定 |

float16 只用于 CPU replay 存储，采样后恢复 float32；网络输入和优化不使用 float16，除非 Trainer 的 precision 另有配置。启用 replay checkpoint 时单个 checkpoint 会额外增加约 128 MiB。若在 epoch 中间恢复，精确 transition 延续还依赖 Lightning/DataLoader 恢复到同一 batch 位置；epoch 边界 checkpoint 不存在该歧义。

卷积 bias、padding 和参数初始化也没有被论文完整列出。实现为保持图示尺寸使用 `3x3 padding=1`，conv bias 与初始化采用当前 PyTorch 默认；这些同样属于 **[SolarCHIP 适配]**，不是论文事实。

训练模式：

- 主模型：`compare.flare.yi2023_dqn.module.Yi2023DQN`，默认 `training_mode: dqn`。
- 监督 control：`compare.flare.yi2023_dqn.module.Yi2023Supervised`，或给主类传入 `training_mode: supervised`。

验证/测试固定记录 `*_accuracy`、`*_macro_f1`、`*_inconsistent_head_rate`，以及 `Cplus/Mplus/Xplus` 各自的 `f1/tss/hss/apss/positive_rate`。DQN 另记录 TD loss、TD error、epsilon、replay size 和 reward；验证 checkpoint 应监控 `val_macro_f1`，不在 test 上调阈值。

## 12. 已执行验证

- 真实 512² 前向的逐层尺寸与 `3536 -> [3,2]` 输出合同通过。
- 四类累计 target、最高阳性解码及非单调 flag 自检通过。
- replay ring 与采样 RNG checkpoint round-trip 通过。
- epoch 末的 terminal transition 与两-epoch target 同步会在 validation checkpoint 保存前幂等完成；合成 Lightning 两 epoch 训练已核对保存边界语义。
- 使用实际 YiDenseNet 的监督反向传播和 DQN TD 反向传播通过。
- 使用 Lightning Trainer 的 manual-optimization 一轮 train/validation smoke test 通过，并产生 `val_accuracy`、`val_macro_f1` 及全部三头指标。
- 当前 `solargpt` 环境没有安装 `pytest`；`test_module.py` 已通过逐函数直接执行，未因此安装或修改依赖。
