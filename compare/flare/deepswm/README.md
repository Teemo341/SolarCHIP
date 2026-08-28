# A7 — DeepSWM HMI-only 调研与复现

状态：**已实现可训练的 DeepSWM-derived HMI-only adaptation；它不是十通道原模型的完整复现。**

核心来源：

- **[论文原文]** S. Nagashima and M. Sugiura, “Deep Space Weather Model: Long-Range Solar Flare Prediction from Multi-Wavelength Images,” *ICCV 2025*, pp. 9396–9405，[CVF 正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Nagashima_Deep_Space_Weather_Model_Long-Range_Solar_Flare_Prediction_from_Multi-Wavelength_ICCV_2025_paper.html)，[DOI `10.1109/ICCV51701.2025.00877`](https://doi.org/10.1109/ICCV51701.2025.00877)。
- **[官方代码]** 作者仓库固定在 [`278779fb8fc99cab2c68b5ff583ccc76e68c9cc0`](https://github.com/keio-smilab25/DeepSWM/tree/278779fb8fc99cab2c68b5ff583ccc76e68c9cc0)。
- **[论文原文]** 公开基准为 [FlareBench](https://huggingface.co/datasets/sh237/FlareBench)。

证据标签含义见[上级 README](../README.md)。本项目版本必须称为 **DeepSWM-derived HMI-only adaptation**，不能称为完整 DeepSWM 复现。

## 1. FlareBench 原始任务

- **[论文原文]** FlareBench 覆盖 2011-06 至 2022-11，目标是预测输入时刻之后 24 小时内最大 X/M/C/O 耀发等级。
- **[论文原文]** 初始数据有 100,801 个逐小时样本；移除 2,440 个标签缺失样本和 2,524 个输入通道缺失率超过 25% 的样本后，论文写为 95,837 个样本。
- **[论文原文]** 保留样本的输入通道平均缺失率为 `0.93% ± 4.03%`；缺失率不超过 25% 时，对缺失通道做 zero padding。
- **[论文原文]** 论文同时写出 X/M/C/O 类别数为 `1,750 / 13,263 / 34,978 / 47,775`。
- **[论文原文]** 上述四项相加为 **97,766**，与论文声明的总数 95,837 相差 1,929；并且 `100,801 - 2,440 - 2,524 = 95,837`。这是原文内部不一致，本调研不替作者改写某一项。
- **[论文原文]** 每个时刻包含 10 个通道：一张 HMI LOS magnetogram 和九个 AIA 波段图像。
- **[论文原文]** 近期输入长度 `k=4`，四帧按 1 小时间隔排列；长期分支使用 `m=672` 个逐小时表征，即 28 天历史。
- **[SolarCHIP 适配]** 当前项目只使用 HMI，时间步间隔改为一天；同一个 `T` 同时控制近期 SSE 输入和 LT-SSM 历史，默认 `T=1`。

## 2. 原始时间序列 cross-validation

- **[论文原文]** 论文使用三个时间折；每折均包含多年的 train、独立 validation 和后置 test，并让 test 覆盖不同太阳活动状态。

| Fold | Train | Validation | Test | 证据 |
| --- | --- | --- | --- | --- |
| 1 | 2011-12 至 2019-05 | 2011-06 至 2011-11，以及 2019-06 至 2019-11 | 2019-12 至 2021-11 | **[论文原文]** |
| 2 | 2011-12 至 2019-11 | 2011-06 至 2011-11，以及 2019-12 至 2020-05 | 2020-06 至 2022-05 | **[论文原文]** |
| 3 | 2011-12 至 2020-05 | 2011-06 至 2011-11，以及 2020-06 至 2020-11 | 2020-12 至 2022-11 | **[论文原文]** |

- **[SolarCHIP 适配]** 主比较不复刻这些逐小时 folds，只使用现有日级 DataModule 的 train/validation split；当前没有 test split。
- **[官方代码]** 当前 CLI 接受 fold 1–5，但论文只定义和汇报三个 folds；额外 fold 4/5 不能写成 ICCV 论文协议。

## 3. 原始图像预处理与增强

- **[论文原文]** HMI/AIA 全日面图从 `1024 x 1024` 处理为 `256 x 256`。
- **[论文原文]** HMI 图左下角时间戳区域被遮盖，AIA 图像经过对齐和裁剪，以减少非太阳内容及跨通道错位。
- **[论文原文]** normalization statistics 由训练数据按通道计算。
- **[论文原文]** 训练增强包含 rotation、scaling、brightness/contrast、Gaussian blur 和 channel noise。
- **[SolarCHIP 适配]** 当前输入已完成项目的 signed-`log1p` + z-score；模型内只统一 resize 至 `256 x 256`，不引入 AIA 对齐或缺通道 zero padding。
- **[SolarCHIP 适配]** 多日窗口的随机空间变换必须对全部 T 帧一致；不能让同一个物理结构在相邻日因独立增强发生人为跳动。

## 4. 原始网络结构

### 4.1 Solar Spatial Encoder（SSE）

- **[论文原文]** SSE 输入 `x in R^(k x C x H x W)`，其中 `k=4`、`C=10`，输出短期特征 `h_sse in R^(L x D)`。
- **[论文原文]** SSE 有 `L_SSE=3` 个层级，反复进行空间下采样、Depth-wise Channel Selective Module（DCSM）和 Spatio-Temporal State Space Model（ST-SSM）。
- **[论文原文]** DCSM 并行使用 depthwise `3 x 3 x 3` 与 `1 x 3 x 3` convolution，经过 image/channel attention、point-wise refinement 和 residual connection 融合时空与波段信息。
- **[论文原文]** ST-SSM 把空间/通道 token 展平后用基于 S5 的 state-space block 建模，再恢复空间结构。

### 4.2 Sparse MAE 与长期历史

- **[论文原文]** 论文先独立训练 Sparse Masked Autoencoder；随后对 672 个历史时刻逐时编码，形成 `h_pre` 序列。
- **[论文原文]** Sparse MAE 不是均匀随机 masking：先依据 patch 标准差保留高信息区域，再对其余区域进行两阶段稀疏 masking，以尽量保留太阳黑子等关键结构。
- **[论文原文]** encoder 为 8 个 Transformer blocks，decoder 为 12 个 blocks，历史特征维度 `D_pre=128`。
- **[论文原文]** LT-SSM 使用 S5 blocks 对 `m=672` 的历史特征建模，再用 1-D convolutions 调整为与 SSE 兼容的 `L x D` 表征。

### 4.3 融合与分类

- **[论文原文]** SSE 与 LT-SSM 输出在 sequence 维连接，经过 mixing SSM，再由 feed-forward classification head 输出四类预测。
- **[论文原文]** 论文实验表设置 `D=64`、`L_LT=1`。
- **[官方代码]** 固定快照的 `DeepSWM` constructor 使用 sequence length `L=128`；这一数值来自实现合同，论文实验表没有单列 `L`。
- **[论文原文]** 四类语义为 O/C/M/X；这与当前 `0AB/C/M/X` 的等级顺序接近，但 O 和 0AB 的事件目录定义仍不是字面相同的数据标签。
- **[官方代码]** 推理服务按 `O/C/M/X` 顺序解释四维输出。
- **[SolarCHIP 适配]** 当前输出维度仍为四，索引固定为 `0AB/C/M/X -> 0/1/2/3`；GMGS score matrix 和所有 one-hot 处理必须按这个顺序显式核对。

## 5. 原始 Sparse MAE 预训练

- **[论文原文]** Sparse MAE 训练 20 epochs，batch size 32。
- **[论文原文]** optimizer 为 AdamW，betas `(0.9, 0.95)`，learning rate `4e-3`，weight decay `0.05`。
- **[论文原文]** 高标准差 patch 比例 `alpha=20%`；两阶段 masking 参数为 `r_l=0.3`、`r_h=0.5`、`r_f=0.5`。
- **[论文原文]** reconstruction objective 为 normalized-pixel mean squared error。
- **[SolarCHIP 适配]** 当前方案**不执行**这一步，也不加载其 encoder checkpoint；不能把 HMI-only 模型描述为“经过 DeepSWM Sparse MAE 预训练”。

## 6. 原始主模型训练

- **[论文原文]** 主训练分两阶段：stage 1 在自然不平衡训练集上训练 20 epochs；stage 2 冻结 feature extractor，仅训练 classifier 15 epochs。
- **[论文原文]** stage 2 每类在一个 batch 中取 8 个样本，总 batch size 32。
- **[论文原文]** 主模型 optimizer 为 AdamW，betas `(0.9, 0.95)`，learning rate `4e-5`，weight decay `0.05`，batch size 32。
- **[论文原文]** 总损失由 cross-entropy、GMGS-oriented loss 和 BSS-oriented loss 组成，权重为 `lambda_CE=1`、`lambda_GMGS=1`、`lambda_BSS=2`。
- **[论文原文]** checkpoint 以 validation GMGS 选择。
- **[官方代码]** 当前 CLI 的 optimizer 默认值却是 `adam`，虽然可选 `adamw`；按论文复现时不能依赖 CLI 默认值。
- **[官方代码]** `DeepSWM` constructor 默认 `L_LT=2`，与论文列出的 `L_LT=1` 不一致；运行配置必须显式覆盖，不能依赖 constructor 默认值。
- **[SolarCHIP 适配]** 保留 CE/GMGS/BSS 的论文语义，但所有概率型 loss 必须从 logits 先计算一次 softmax；class weights 只能由当前 train split 得出。
- **[SolarCHIP 适配]** 原 stage-2 balanced batch sampler 会改变现有 DataModule，因而不作为默认主比较路径。当前实现冻结除 classifier 外的参数，并用 train-derived class-weighted loss 代替 sampler；这是项目适配，不是论文原设置。

## 7. 原始指标、结果与算力

- **[论文原文]** 主要指标为四分类 Gandin–Murphy–Gerrity score（GMGS）、M+ 事件的 Brier skill score（BSS-M+）与 M+ true skill statistic（TSS-M+）。
- **[论文原文]** 三折结果为 GMGS `0.582 ± 0.032`、BSS-M+ `0.334 ± 0.299`、TSS-M+ `0.543 ± 0.074`。
- **[论文原文]** 模型约 1.59M parameters、4.64G MACs；论文报告在 NVIDIA H200 140 GB 上训练约 3 小时，推理约 12 ms/sample。
- **[SolarCHIP 适配]** 这些结果来自 10 通道、逐小时、4 帧+672 小时历史和独立预训练；不能作为 HMI-only、日间隔、默认 T=1 版本的对照数值。
- **[SolarCHIP 适配]** validation 和未来 test 额外记录四分类 accuracy、confusion matrix 与 macro-F1，同时保留 GMGS、BSS-M+ 和 TSS-M+。

## 8. 固定代码快照审计

### 8.1 维度硬编码

- **[官方代码]** `DeepSWM` 直接用 `SolarSpatialEncoder(4, ...)`，把近期时间长度 `k=4` 硬编码为第一层 `Conv3d.in_channels`。
- **[官方代码]** 官方输入形状为 `[B,k,C,H,W]`；第一层 Conv3d 把 `k` 当输入 channel，而模态数 `C=10` 保留为 3-D convolution 的 depth 维。
- **[官方代码]** SSE 尾部写死 `Conv2d(D*10, D, ...)`，forward 也写死 reshape 为 `D*10`，因此不是只改数据 shape 就能运行 HMI-only。
- **[官方代码]** 对 `256 x 256` 输入，stem 的空间 stride 4 加三个 stride-2 层得到 `8 x 8`；随后代码把输出整理为 `[B,L=128,D=64]`。
- **[官方代码]** LT-SSM 期望历史特征约为 `[B,m,128]`，再输出与 SSE 一致的 `[B,128,64]`。
- **[SolarCHIP 适配]** 后续实现必须显式参数化 `time_length=T` 和 `num_modalities=1`，移除所有 `4` 与 `10` 的形状假设。

### 8.2 softmax / CrossEntropy 问题

- **[官方代码]** `DeepSWM.forward()` 先对 classification logits 调用 `Softmax(dim=1)`，返回概率。
- **[官方代码]** 自定义 `Losser` 随后把该概率直接传给 `nn.CrossEntropyLoss`，而 PyTorch CE 期望未归一化 logits；这相当于对概率再次执行内部 log-softmax。
- **[SolarCHIP 适配]** Lightning 版本必须返回 raw logits；CE 直接消费 logits，GMGS/BSS 与指标需要概率时只调用一次 softmax。这是数值正确性修复，必须在结果说明中记录。

### 8.3 快照、权重与依赖

- **[官方代码]** 固定 commit 的作者和提交者均为 `github-actions`，message 为自动更新数据/日志；滚动 `main` 会持续被自动提交，因此所有审计链接必须使用 commit hash。
- **[官方代码]** 官方 requirements 锁定 `s5-pytorch==0.2.1`、`timm==1.0.15`、`torch==2.1.1`、`torchvision==0.16.1`，并包含其余数据/训练依赖。
- **[官方代码]** 官方发布权重按 10 模态和 `k=4` 构建，SSE 与 Sparse MAE 参数形状均不兼容单通道 HMI/T=1。
- **[SolarCHIP 适配]** 当前 `solargpt` 环境已安装并实测 `s5-pytorch==0.2.1` 与 `timm==1.0.15`；核验环境为 Python 3.13.2、PyTorch 2.6.0、torchvision 0.21.0，安装过程没有降级现有 PyTorch。
- **[SolarCHIP 适配]** 复用上述官方依赖，不重写 S5；固定版本记录在上级目录的 `requirements.txt`。官方完整 requirements 中的旧版 PyTorch/torchvision 不用于本适配。
- **[SolarCHIP 适配]** 不加载官方 DeepSWM 或 Sparse MAE 权重。

## 9. 已锁定的 HMI-only adaptation

### 9.1 日级 Dataset 子类

- **[SolarCHIP 适配]** 在本子目录内继承现有 `FlareDataset`，单样本返回 `hmi` 形状 `[T,1,1024,1024]`，DataLoader collate 后为 `[B,T,1,1024,1024]`。
- **[SolarCHIP 适配]** 窗口目标为最后一天 D 的现有四分类标签；历史日期为 `D-(T-1), ..., D`，严格相差一个日历日。
- **[SolarCHIP 适配]** 只有所有 T 天的 HMI 都真实存在才保留窗口；不插值、不复制、不用最近邻日期填补。
- **[SolarCHIP 适配]** 窗口全部日期必须在同一现有 split 内；train/validation 边界处不从另一 split 借历史帧。
- **[SolarCHIP 适配]** 所有随机空间增强对同一序列共享参数；当前版本只启用现有管线的共享 flip/rotation，不添加 brightness、contrast、blur 或 channel noise。
- **[SolarCHIP 适配]** 默认 `T=1`；增大 T 会减少可用目标日，实际样本数必须在训练日志中记录。

### 9.2 HMI-only 网络

- **[SolarCHIP 适配]** Dataset 子类由现有 DataModule 实例化，网络、损失和指标已封装为 `pytorch_lightning.LightningModule`，入口见本目录的 `dataset.py`、`model.py` 与 `module.py`。
- **[SolarCHIP 适配]** 输入在模型内逐帧 resize 至 `[B,T,1,256,256]`。
- **[SolarCHIP 适配]** SSE 保留三层 downsampling、DCSM 与 ST-SSM，但参数化为 `time_length=T`、`num_modalities=1`，输出仍对齐为 `[B,128,64]`。
- **[SolarCHIP 适配]** 长期分支采用论文 Sparse MAE **encoder 结构**的单通道版本：patch embedding、8 个 encoder Transformer blocks 与 final norm；移除 decoder、masking 和 reconstruction head。
- **[SolarCHIP 适配]** 该 encoder 对每一天输出一个 128 维表征；与官方 feature-extraction 脚本一致，表征由 final norm 后的 patch tokens（排除 cls token）取均值得到。T 天组成 `[B,T,128]` 输入 LT-SSM；权重从零初始化，并与 SSE、LT-SSM、mixing SSM 和 classifier 联合训练。
- **[SolarCHIP 适配]** 不运行独立 Sparse MAE 预训练，也不加载任何多通道预训练 encoder。
- **[SolarCHIP 适配]** 短期 `k` 和长期 `m` 均由同一个日级 `T` 控制。`T=1` 时 LT-SSM 只处理长度一的序列，保留结构但不具备论文的长期依赖信息。
- **[SolarCHIP 适配]** classifier 输出 `[B,4]` raw logits，类别顺序固定 `0AB/C/M/X`；禁止在送入 CE 前 softmax。

## 10. 论文协议与项目协议差异

| 项目 | 论文 | SolarCHIP 实现 | 证据 |
| --- | --- | --- | --- |
| 模态 | HMI + 9 AIA | HMI-only | **[论文原文] / [SolarCHIP 适配]** |
| 近期输入 | 4 个逐小时帧 | T 个逐日帧，默认 1 | **[论文原文] / [SolarCHIP 适配]** |
| 长期输入 | 672 个逐小时 encoder 特征 | 同一个 T 个逐日特征 | **[论文原文] / [SolarCHIP 适配]** |
| 预训练 | 独立 10 通道 Sparse MAE | 无预训练，单通道 encoder 联合训练 | **[论文原文] / [SolarCHIP 适配]** |
| 缺失模态 | 阈值筛除后 zero padding | 严格要求每日 HMI 存在 | **[论文原文] / [SolarCHIP 适配]** |
| 两阶段平衡 | stage 2 每类 8 个样本 | train-derived inverse-frequency weighted loss 替代 | **[论文原文] / [SolarCHIP 适配]** |
| 输出 | O/C/M/X | 0AB/C/M/X | **[论文原文] / [SolarCHIP 适配]** |
| split | 三个 FlareBench 时间折 | 现有 DataModule split | **[论文原文] / [SolarCHIP 适配]** |

## 11. 原文未报告或适配说明

- **[原文未报告]** 论文没有 HMI-only、日间隔或 `T=1` 的 DeepSWM 成绩。
- **[原文未报告]** 论文没有“移除 Sparse MAE 预训练、单通道 encoder 从零联合训练”的实验，因此无法预估该改动损失多少性能。
- **[原文未报告]** FlareBench 总数与分项类别计数冲突，无法仅凭论文判定哪一组数字应被修正。
- **[SolarCHIP 适配]** stage 2 已启用；所有逐样本损失项均使用当前 train split 计算的逆频率权重，权重归一化为均值 1，不读取 validation 分布。
- **[SolarCHIP 适配]** `T>1` 的可用窗口数、各类分布和 split 边界损失必须在真正构建 Dataset 后统计，不在文档阶段估算。
- **[SolarCHIP 适配]** `s5-pytorch==0.2.1` 已在当前 PyTorch 2.6.0 环境完成独立 S5、完整 T=1/T=2 前向及完整反向兼容性测试。

## 12. 实现后的验收建议

- **[SolarCHIP 适配]** 用包含缺日与 split 边界的人工日期表验证窗口过滤，确保不会补洞或跨 split。
- **[SolarCHIP 适配]** 给整段序列施加一次已知 flip/rotation，逐帧核对增强矩阵完全一致。
- **[SolarCHIP 适配]** 对 `T=1`、`T=2` 做真实 forward，逐层断言 SSE depth=1、空间 `256 -> 64 -> 32 -> 16 -> 8`，以及 SSE/LT 输出均为 `[B,128,64]`。
- **[SolarCHIP 适配]** 用同一 logits 同时核对 CE、一次 softmax 后的 GMGS/BSS 和四分类 accuracy，防止复现官方 double-softmax 问题。
- **[SolarCHIP 适配]** 对 `0AB/C/M/X` 四个 one-hot 样本验证 GMGS matrix、M+ 合并与 TSS/BSS 的类别方向。
- **[SolarCHIP 适配]** 用一个真实 DataModule batch 完成 Lightning train/validation smoke test；当前无 test split，只验收 test hooks 可构造，不产生测试结果。
## 13. 当前实现入口与训练合同

- **[SolarCHIP 适配]** `dataset.py:DeepSWMSequenceDataset` 继承现有 `FlareDataset`，仅允许 `modal_list=['hmi']`。它严格过滤缺日和 split 边界，返回 `[T,1,H,W]`，并让同一窗口共享 flip/rotation 参数。
- **[SolarCHIP 适配]** `model.py:HMIOnlyDeepSWM` 保留 SSE、DCSM、ST-SSM、LT-SSM、mixing SSM 和 FFN 的官方拓扑；所有 SSM 块直接使用固定依赖 `s5-pytorch==0.2.1`。
- **[SolarCHIP 适配]** 历史分支使用单通道、patch size 8、width 128、8 个 Transformer blocks 的 SparseMAE encoder-only 结构；包含固定 2-D sine/cosine position embedding 与 cls token，不包含 masking、decoder 或预训练权重。
- **[SolarCHIP 适配]** `module.py:DeepSWM` 是 LightningModule。输入可为原有单帧 `[B,1,H,W]`（仅 `T=1`）或序列 `[B,T,1,H,W]`，在模型内 area resize 至 256；输出始终为四类 raw logits。
- **[SolarCHIP 适配]** loss 为 `CE + GMGS-oriented + 2 * multiclass-Brier`；三个逐样本分量均使用 train-derived inverse-frequency weights，GMGS-oriented loss 的固定 score matrix 也只由 train climatology 构造。报告四分类 accuracy、macro-F1、GMGS、BSS-M+ 与 TSS-M+；评估 GMGS 按论文定义从当前 split 的 contingency-table 真实类别边际重建 score matrix，不复用训练 loss 的固定矩阵。
- **[SolarCHIP 适配]** epoch `0..19` 联合训练；epoch `20..34` 将 SSE、SparseMAE encoder、LT-SSM 与 mixing SSM 冻结并置为 eval，仅训练 classifier。正式 Trainer 必须设 `max_epochs=35`。
- **[SolarCHIP 适配]** 当前实现不加载任何官方权重。训练 checkpoint 的默认选择仍应由运行配置监控 `val_macro_f1`；论文指标 `val_gmgs` 同时保留以供审计。
- **[SolarCHIP 适配]** 正式配置使用 micro-batch 8 与 `accumulate_grad_batches=4`，在单 GPU 上得到 effective batch size 32；这是适配现有设备内存的工程设置，不是论文的物理 batch 32。

### 已完成的实现验收

- **[SolarCHIP 适配]** 使用更新后的 `hmi_exist_idx.pkl` 实例化正式 split：默认 `T=1` 时 train/validation 分别保留 4,963/397 个样本，train 类别计数为 `2262/1846/770/85`。`T=2` 和 `T=4` 的 train 窗口过滤也已验证，分别保留 4,928 和 4,858 个无缺日窗口。
- **[SolarCHIP 适配]** 合成 HMI 验证了同一 T 日窗口共享 flip/rotation，并验证 Dataset 输出严格为 `[T,1,H,W]`；异常存储维度会在 Dataset 内直接报错。
- **[SolarCHIP 适配]** 默认完整网络已通过 `[1,1,1024,1024] -> [1,4]` 前向、CE/GMGS/Brier 完整反向以及 `T=2` 前向；所有输出和 232 组实际梯度均为 finite。
- **[SolarCHIP 适配]** 两 epoch Lightning smoke test 已验证 stage 1 全网络可训练、stage 2 四个 feature modules 全部冻结且保持 eval、classification head 继续训练；validation 会生成 `val_accuracy`、`val_macro_f1`、`val_gmgs`、`val_bss_mplus` 与 `val_tss_mplus`。
- **[SolarCHIP 适配]** 正式 YAML/DataModule 的 batch 合流已通过：在只用合成 HMI 替代不可访问的 NAS 像素读取后，真实日期索引、标签、窗口 Dataset 和默认 collate 产生 `[8,1,1,1024,1024]`，并直通完整网络得到 `[8,4]` finite logits/loss。当前机器的 `DATA_ROOT` 仍指向训练服务器路径，因此真实 HMI 文件 I/O 仍需在该路径可用的训练环境做首批次检查。

### 推荐的 DataModule target

训练和验证 split 均应把 dataset target 设置为：

```yaml
target: compare.flare.deepswm.dataset.DeepSWMSequenceDataset
params:
  modal_list: [hmi]
  window_length: 1
```

其余 `enhance_type`、`torch_augment_type`、`time_interval`、标签文件和 `class_groups` 延用现有 flare 配置。将 `window_length` 增大至 `T>1` 时，model 和两个 dataset split 必须同时使用相同的值。
