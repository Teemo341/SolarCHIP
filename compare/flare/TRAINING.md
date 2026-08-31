# A1/A2/A7 训练入口与已采用配置

本页记录当前可执行版本的统一训练合同。论文事实、官方代码审计及逐项适配依据仍在各模型的 README 中。

## 直接训练

先进入 SolarCHIP 的训练环境和仓库根目录。A7 还要求安装固定依赖：

```bash
python -m pip install -r compare/flare/requirements.txt
```

四个入口分别为：

```bash
bash compare/flare/yi2023_dqn/train_dqn.sh
bash compare/flare/yi2023_dqn/train_supervised.sh
bash compare/flare/pcnn/train.sh
bash compare/flare/deepswm/train.sh
```

脚本默认调用当前环境中的 `python`。需要指定解释器时使用，例如：

```bash
SOLARCHIP_PYTHON_BIN=/path/to/python bash compare/flare/pcnn/train.sh
```

额外的 SolarCHIP 命令行覆盖项可直接附在脚本末尾。四个脚本均固定正式实验 seed 为 `42`，并使用现有训练入口、DataModule、标签和 split。当前没有 test split，训练不会自行创建 test 日期。

## 统一推荐配置

| 项目 | 当前采用值 |
| --- | --- |
| 正式 seed | 仅 `42`，共 1 个 seed |
| 输入与标签 | 现有 HMI、signed-`log1p` + z-score、当天最高 `0AB/C/M/X` |
| split | train `[0,5000)`、validation `[5000,5400)`，再由 HMI 可用索引过滤 |
| 当前索引结果 | train `4963`（`2262/1846/770/85`），validation `397`（`5/121/229/42`） |
| 空间增强 | train `[1024, 0.5, 360]`（完整角度范围），validation `[1024, 0, 0]`；A7 对整段序列共享同一变换 |
| checkpoint | validation 四分类 `macro-F1` 最大；另保留完整 `last.ckpt` |
| 数值设置 | `32-true`、deterministic、单 GPU |
| test | 仅保留 hook；不创建 split、不报告结果 |

## A1：Yi 2023 DQN adaptation

| 项目 | 当前采用值 |
| --- | --- |
| 输出 | 共享 DenseNet 主干 + C+/M+/X+ 三个二动作 Q 头；最高阳性头解码四分类 |
| RL | vanilla DQN，`gamma=0.99` |
| reward | 三头共同使用 TP/FP/FN/TN=`8/-64/-8/1` |
| 探索 | epsilon `1.0 -> 0.05`，按 20,000 个环境动作线性衰减 |
| replay | 容量 128、warm-up 32、batch 8、每个 dataloader batch 更新 1 次 |
| transition | 当前训练 DataLoader 顺序；epoch 末 transition 为 terminal |
| target | 每 2 epochs 从 online 硬同步；vanilla max target，不用 Double-DQN |
| optimizer | RMSprop，`lr=1e-4`、alpha `0.99`、epsilon `1e-8`、momentum `0` |
| 预算 | batch 8，100 epochs |
| checkpoint | 保存 replay、pending transition 和 RNG 状态，支持一致续训 |
| 对照 | 同架构三累计头 supervised CE，对应独立脚本和配置 |

论文未报告的 replay、gamma、epsilon、batch 和学习率均是显式的 SolarCHIP 工程默认值，不冒充论文参数。

## A2：P-CNN adaptation

| 项目 | 当前采用值 |
| --- | --- |
| backbone | torchvision EfficientNetV2-S `IMAGENET1K_V1`；不提供 scratch fallback |
| 预处理 | 中心半高裁剪，bicubic 到 `224x448`，切 8 个 `112x112` patch |
| 归一化 | 保留 SolarCHIP z-score，不再叠加 ImageNet mean/std |
| 网络 | 一个共享 backbone，三个累计 MIL head，patch logits 取 max |
| BN | backbone BN 参数与运行统计冻结；新增 head BN 正常训练 |
| loss | C+ 正负平衡；M+ 四类权重 `2/1/8/8`；X+ 正负平衡；各头先归一尺度后等权平均 |
| 解码 | 固定 `0.5`，最高通过阈值决定四分类；不做单调投影，记录不一致率 |
| optimizer | AdamW，`lr=1e-5`、weight decay `1e-4`，无 scheduler |
| 预算 | batch 16，15 epochs，单一现有 split/checkpoint，不做五折集成 |

正式训练节点首次运行需能下载 `EfficientNet_V2_S_Weights.IMAGENET1K_V1`，或预先将同一文件放入该节点的 Torch Hub cache。

## A7：DeepSWM HMI-only adaptation

| 项目 | 当前采用值 |
| --- | --- |
| 序列 | 严格连续日窗口，默认 `T=1`；窗口不跨 split，标签取最后一天 |
| 输入 | `[B,T,1,1024,1024]`，模型内 area resize 到 `256x256` |
| 网络 | 单通道 Sparse-MAE encoder 结构从零联合训练；保留 SSE 与 LT-SSM |
| 预训练 | 不运行 Sparse-MAE decoder/masking/独立预训练，不加载不兼容的 10 通道权重 |
| loss | train 逆频率四类权重同时作用于 CE + `1*GMGS-oriented + 2*multiclass Brier (BSS-oriented)` |
| 两阶段 | 前 20 epochs 联合训练；后 15 epochs 冻结 feature path，仅训练 classifier |
| optimizer | AdamW，`lr=4e-5`、weight decay `0.05`、betas `(0.9,0.95)` |
| 模型配置 | `D=64`、`L=128`、Sparse encoder depth 8 / heads 8 / patch 8、LT depth 1、mixing depth 1 |
| dropout | SSE/DCSM/ST-SSM/LT-SSM/mixing=`0.6`，head=`0.7`，采用官方运行配置而非构造器备用默认值 |
| 预算 | micro-batch 8、梯度累积 4（有效 batch 32），共 35 epochs |
| 增强 | 同一序列共享空间增强；第一版不加亮度/噪声扰动 |

## 之后可能需要改动、但不阻塞当前训练的项目

这些项目目前已经按推荐值固定，不需要在第一次训练前继续等待决定：

1. A1 的全部原文未报告 DQN 超参数，以及是否保留约占百 MB 的 replay 完整 checkpoint。
2. A1 把 shuffled DataLoader 的相邻 batch 当作环境序列，是否改成严格日期有序或把每张图视为独立 terminal transition；当前 episode 边界还假设只在 epoch 末 validation，若改成 epoch 中验证需复核。
3. A1 supervised control 是否纳入最终论文表格，还是只用于判断 RL 增益。
4. A2 的中心半高近似是否改成按太阳半径/像素尺度实现论文 `±614 arcsec` 物理裁剪。
5. A2 是否额外叠加 ImageNet mean/std；当前推荐不叠加，以避免破坏已有 dataloader 合同。
6. A2 是否改成三个完全独立的 backbone，或补论文五折集成；当前采用共享 backbone、单 split。
7. A2 的 M+ grouped 权重、X+ 自动平衡、三头等权与固定阈值是否做消融；当前不做单调投影，并以四分类 macro-F1 而不是论文的二分类 validation TSS 选 checkpoint。
8. A7 的正式窗口长度 `T`。当前先跑 `T=1`；改变时必须同时修改 model、train dataset 与 validation dataset 的 `window_length`。
9. A7 是否恢复独立 Sparse-MAE 预训练或尝试官方 10 通道结构；当前明确是 HMI-only、from-scratch joint-training adaptation。
10. A7 的高 dropout、逆频率权重作用范围、GMGS/BSS 系数、两阶段切换点和有效 batch 32；当前 micro-batch 8 + 累积 4 不等于论文的物理 batch 32。
11. A7 第二阶段当前从 epoch 20 的连续状态直接冻结训练，不重载 stage-1 最佳 checkpoint、也不重置 optimizer；若要贴近官方训练脚本，应拆成两个运行阶段。
12. A7 当前以 macro-F1 选 checkpoint，而论文以 GMGS；论文提到 label smoothing 却未报告系数，当前按官方代码使用 one-hot。论文 GMGS loss 公式与冻结代码的额外负号也存在冲突，当前遵循论文公式。
13. A7 当前手写 epoch confusion accumulator 针对正式单 GPU 配置；若改成多 GPU，需要补跨 rank 汇总后再训练。
14. 三个模型当前沿用 train 的随机 flip 与完整角度旋转；若认为耀发形态不应接受完整旋转增强，可统一缩小角度再做公平重训。
15. 三个模型的 epoch 预算、batch size、精度、checkpoint 指标和 seed 数；其中 seed 数 1 与 A2 仅 ImageNet 预训练是用户已明确锁定项。

## 运行前环境边界

- 训练节点必须能从 `global_settings.DATA_ROOT` 读取 HMI 文件；索引与标签能成功初始化不代表另一台机器上的原图挂载路径一定相同。
- 当前依赖验证环境为 torch `2.6.0`、torchvision `0.21.0`、Lightning `2.5.0.post0`、`s5-pytorch==0.2.1`、`timm==1.0.15`。
- 修改 `T`、类别分组或 split 后，应重新核对训练类计数和每个 split 的严格连续窗口数。
