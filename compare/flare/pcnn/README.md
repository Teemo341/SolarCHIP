# A2 — P-CNN 调研与复现

状态：**已实现，可进入正式训练；当前实现是论文结构在 SolarCHIP 日级四分类协议上的 HMI-only adaptation。**

核心来源：

- **[论文原文]** G. Francisco et al., “Limits of Solar Flare Forecasting Models and New Deep Learning Approach,” *The Astrophysical Journal*, 985:108, 2025，[DOI `10.3847/1538-4357/adc56d`](https://doi.org/10.3847/1538-4357/adc56d)，[正式 PDF](https://eprints.whiterose.ac.uk/id/eprint/226838/1/Francisco_2025_ApJ_985_108.pdf)。
- **[官方代码]** 论文仓库固定在 [`9a7ef5984fd3d714bb78da0edad3c5a10dd80641`](https://github.com/gfrancisco20/flare_limits_pcnn/tree/9a7ef5984fd3d714bb78da0edad3c5a10dd80641)。
- **[官方代码]** 实际网络与数据工具依赖 `sundl`，本次审计固定在 [`48a79c9acffa26c86800d6c2f136d328104d318c`](https://github.com/gfrancisco20/sundl/tree/48a79c9acffa26c86800d6c2f136d328104d318c)。
- **[论文原文]** 论文数据存档为 [SDO-2H-ML / Zenodo](https://zenodo.org/records/10465437)。

证据标签含义见[上级 README](../README.md)。

## 1. 原始问题、数据与标签

- **[论文原文]** 数据共 56,664 个样本，覆盖 2010-05-14 至 2023-04-18，每 2 小时采样一次。
- **[论文原文]** HMI-only 实验每个时刻输入一张全日面 line-of-sight magnetogram，源序列为 JSOC `hmi.M_45s` Level 1.5；论文另有 AIA 94/193/211 多通道实验，但不属于本 HMI-only 基线。
- **[论文原文]** 标签为从输入时刻开始的未来 24 小时内 GOES SXR 最大耀发强度。
- **[论文原文]** 作者分别训练 `C+` 和 `M+` 两个二分类器；不是一个四分类网络，也没有 X+ 分类器。
- **[SolarCHIP 适配]** 当前模型输入仍是每天一张 D 日 00:00 HMI，标签仍是现有 UTC 日历日最高 `0AB/C/M/X`；不把 2 小时重叠样本或论文标签生成逻辑引入 dataloader。

## 2. 论文的数据切分与抽样

- **[论文原文]** 2010-05 至 2019-12 用于训练和五折 cross-validation，2020-01 至 2023-04 用作跨太阳活动周的 chronological operational test。
- **[论文原文]** CV 使用 81 日有效时间块；块之间保留 27 日 buffer，以降低相邻样本和同一太阳自转造成的泄漏。
- **[论文原文]** 时间块分配时使用 quiet/B/C/M/X 为 `4/1/1/2/4` 的平衡权重。
- **[论文原文]** 训练折对 quiet、B、C、M 做有限欠采样，使其数量大致相等并保留全部 X；验证折按论文描述保留自然发生率。
- **[官方代码]** `instantiateFolds()` 在 test 边界后应用 27 日 buffer，因此实际 test 起点约为 2020-01-28，而不是文字表述的 2020-01-01。
- **[官方代码]** CV notebook 构造验证折时使用 `SC_25_Asc` 比例；这与论文“验证保持自然气候比例”的描述不完全一致。
- **[SolarCHIP 适配]** 主比较不调用上述 fold/undersampling 代码；所有样本仍由现有 DataModule 和 split 决定。论文 split 只作为 paper-protocol 背景。
- **[SolarCHIP 适配]** 若将来补充论文协议实验，应以五个独立 run 训练并在 test 上平均概率，不能在模型内部临时重排当前 batch 来冒充论文 CV。

## 3. HMI 预处理

- **[论文原文]** 原始磁图先插值到 `1024 x 1024`，约为 `2.4 arcsec/pixel`，并与 AIA 图像空间对齐。
- **[论文原文]** 磁场值经过对称 logarithmic transform。
- **[论文原文]** 截断阈值来自 2010–2019 CV 像素联合分布的 99.9 百分位，HMI 物理范围约为 `±4644 G`。
- **[论文原文]** 截断后映射为 8-bit：负饱和值为 0、零磁场为 127、正饱和值为 255，并保存为 JPEG。
- **[论文原文]** 裁去太阳南北 `±614 arcsec` 之外的高纬区域；在统一 1024 图上相当于保留中心约 512 行。
- **[论文原文]** 裁剪结果 resize 为 `224 x 448`，再切成 `2 x 4`、共 8 个无重叠的 `112 x 112` patch。
- **[官方代码]** 灰度 HMI patch 在进入 ImageNet backbone 前重复为 3 通道。
- **[官方代码]** 论文仓库的 `small_blos.zip` 已是处理后的 `224 x 448` 输入，因此训练 notebook 本身不重做完整物理预处理。
- **[SolarCHIP 适配]** 现有 dataloader 的 signed-`log1p` + z-score 保持不变；模型内部只做中心半高裁剪、bicubic resize、通道重复和切 patch。
- **[SolarCHIP 适配]** 因而本实现不能声称复刻 `±4644 G` 截断、8-bit 映射或 JPEG 压缩，只能称为相同空间布局上的 P-CNN 结构适配。

## 4. 原始 P-CNN 结构

- **[论文原文]** 八个 patch 使用同一个共享权重的 EfficientNetV2-S，不是八套独立 CNN。
- **[论文原文]** backbone 使用 ImageNet 预训练；所有非 BatchNorm 层参与 fine-tuning，BatchNorm 层保持冻结。
- **[论文原文]** 每个 patch 的顶部为 global average pooling、BatchNorm、dropout `0.2`、一维 Dense sigmoid 输出。
- **[论文原文]** 只有全日面标签，没有 patch 标签；训练属于 inexact weak supervision / multiple-instance learning。
- **[论文原文]** 全日面概率由 patch 概率最大值给出：`p_disk = max_i p_patch_i`。这使任意一个高风险 patch 足以触发整盘阳性。
- **[官方代码]** `sundl.models.blueprints.build_pretrained_PatchCNN` 实现 patch extraction、共享 CNN 和 max aggregation；`__build_pretrained_innerPatch` 实现 EfficientNetV2-S 及 patch head。
- **[官方代码]** `sundl.dataloader.sdocml.builDS_image_feature` 负责灰度转 RGB、裁剪/缩放和样本权重。

论文二分类形状合同为：

| 阶段 | 原始行为 | 证据 |
| --- | --- | --- |
| full disk | `224 x 448 x 3` | **[论文原文]** |
| patching | `2 x 4` 个 `112 x 112 x 3`，无重叠 | **[论文原文]** |
| shared encoder | 每个 patch 共用 EfficientNetV2-S | **[论文原文] / [官方代码]** |
| patch head | GAP -> BN -> dropout 0.2 -> sigmoid | **[论文原文] / [官方代码]** |
| MIL | 八个 patch 概率取最大 | **[论文原文] / [官方代码]** |

## 5. 原始训练设置与样本权重

- **[论文原文]** 框架为 TensorFlow/Keras，训练硬件包含 NVIDIA V100。
- **[论文原文]** optimizer 为 AdamW，learning rate `1e-5`，weight decay `1e-4`。
- **[论文原文]** batch size 为 16，固定训练 15 epochs；不使用 learning-rate scheduler 或 early stopping。
- **[论文原文]** 每折保留 validation TSS 最佳 checkpoint。
- **[论文原文]** C+ 的 weighted BCE 使 positive 与 negative 两侧的总贡献相等。
- **[论文原文]** M+ 的类别贡献比例为 quiet/B/C/M/X = `2/2/1/8/8`，用于强调 M/X recall。
- **[论文原文]** 判别阈值固定为 0.5，不在 validation 上调阈值。
- **[论文原文]** operational test 时把五折模型的预测概率做算术平均，再按 0.5 判别。
- **[原文未报告]** 原文没有 X+ head 的损失权重，不能从 M+ 权重直接声称得到论文设置。
- **[SolarCHIP 适配]** C+/M+ 可把原文权重作为候选 paper-aligned 设置；X+ 权重只能由当前 train 标签统计后明确指定，并标成项目适配。

## 6. 原始评估结果

- **[论文原文]** 主要指标包括 TSS、HSS、Matthews correlation coefficient（MCC）、F1、recall 和 false alarm ratio（FAR）；论文还分析 persistence-relative F1 及 activity-change/no-change 子集。
- **[官方代码]** 训练日志同时包含 binary accuracy，但论文主结果表没有把 accuracy 作为核心技能指标。
- **[论文原文]** HMI operational-test C+ 结果为 TSS `0.67`、HSS `0.67`、MCC `0.68`、F1 `0.82`、recall `0.77`、FAR `0.11`。
- **[论文原文]** HMI operational-test M+ 结果为 TSS `0.58`、HSS `0.37`、MCC `0.43`、F1 `0.50`、recall `0.84`、FAR `0.65`。
- **[SolarCHIP 适配]** 上述数值不能作为当前日级四分类版本的预期结果，因为采样频率、标签窗、预处理、split、框架和输出头均已改变。

## 7. 官方代码可复现性审计

- **[官方代码]** 固定仓库包含 `0_CV_Folds.ipynb`、`1_Training.ipynb`、`2_Model_Selection.ipynb`、`3_Test_Predictions.ipynb`、`4_Test_Evaluation.ipynb`、`5_Explainability.ipynb`、`config.py`、`utilsTraining.py` 和 `utilsTest.py`。
- **[官方代码]** 仓库提供 CV 结果表，但 README 明确没有公开训练 checkpoint 和 test predictions；复现实验需要自行训练或联系作者。
- **[官方代码]** TensorFlow/Keras 及依赖没有完整锁定到可逐位复现的版本。
- **[官方代码]** 欠采样路径调用 pandas `.sample()` 时没有统一 `random_state`，因此原 fold 内训练样本不保证逐次完全相同。
- **[官方代码]** 网络主体不在论文仓库本身，而在外部 `sundl`；只固定一个仓库 commit 不足以冻结模型。
- **[SolarCHIP 适配]** 后续实现必须同时记录 SolarCHIP commit、torch/torchvision/Lightning 版本、EfficientNet 权重枚举、随机种子和 split hash。
- **[SolarCHIP 适配]** 不下载或转换作者 checkpoint，因为公开仓库没有对应权重，而且本任务的三头输出也不兼容二分类 head。

## 8. SolarCHIP 结构适配

### 8.1 Forward 路径

- **[SolarCHIP 适配]** 整个 P-CNN、损失和指标已经封装为一个 `pytorch_lightning.LightningModule`，直接读取现有 batch 字典。
- **[SolarCHIP 适配]** 输入保持 `batch['hmi']`，形状 `[B,1,1024,1024]`；不修改 dataloader。
- **[SolarCHIP 适配]** 在模型内取中心半高区域；对标准 1024 输入即保留中心约 512 行。
- **[SolarCHIP 适配]** 使用 bicubic interpolation resize 到 `[B,1,224,448]`，再重复为 `[B,3,224,448]`。
- **[SolarCHIP 适配]** 用 kernel/stride 均为 112 的无重叠 unfold 得到 `[B,8,3,112,112]`，reshape 为 `[B*8,3,112,112]` 一次送入唯一共享 backbone。
- **[SolarCHIP 适配]** patch head 输出 `[B,8,3]` logits，顺序固定为 C+/M+/X+；对 patch 维取 maximum 得到 `[B,3]` disk logits。
- **[SolarCHIP 适配]** 因 sigmoid 单调，`sigmoid(max(logit)) = max(sigmoid(logit))`，训练可用数值稳定的 `BCEWithLogitsLoss` 而不改变 max-MIL 判别语义。

### 8.2 输出与解码

- **[SolarCHIP 适配]** 三个累计目标为 `1[y>=C]`、`1[y>=M]`、`1[y==X]`。
- **[SolarCHIP 适配]** 三个概率均使用固定阈值 0.5；X+ 通过则输出 X，否则 M+ 通过输出 M，否则 C+ 通过输出 C，否则输出 0AB。
- **[SolarCHIP 适配]** 即使 C+/M+/X+ 不满足概率单调性，也不做单调投影；记录 `p_C < p_M`、`p_M < p_X` 和二值动作不一致率。
- **[SolarCHIP 适配]** 直接使用四类 softmax 后逐类做 patch max 不被采用，因为逐类最大值不再构成归一化概率分布，也改变了论文的二元 MIL 语义。

### 8.3 Backbone 边界

- **[论文原文]** 原模型用 Keras EfficientNetV2-S 与其 ImageNet 权重。
- **[SolarCHIP 适配]** PyTorch Lightning 版本使用 torchvision `efficientnet_v2_s` 和 `EfficientNet_V2_S_Weights.IMAGENET1K_V1`；这是框架级、实现级偏差。
- **[SolarCHIP 适配]** 应显式冻结所有 BatchNorm 的参数和运行统计，而不只设置 `requires_grad=False` 后仍让 running mean/variance 更新。
- **[原文未报告]** Keras 与 torchvision 在 stem、padding、默认预处理或权重训练细节上的差异没有论文等价性证明；必须通过适配实验而非宣称严格复现来处理。

## 9. 指标与 checkpoint 合同

- **[SolarCHIP 适配]** validation 与未来 test 必须记录最高阈值解码后的四分类 accuracy。
- **[SolarCHIP 适配]** 每个累计 head 同时记录 TSS、HSS、MCC、F1、recall、FAR 和 binary accuracy，以保留论文比较语义。
- **[SolarCHIP 适配]** 记录四类 confusion matrix、macro-F1、balanced accuracy 和累计头不一致率；这些是当前多分类任务的补充指标。
- **[SolarCHIP 适配]** 只有 validation 可以选择 checkpoint；当前没有 test split，因此只预留 `test_step/test_accuracy`，不产生 test 结果。
- **[SolarCHIP 适配]** 主比较训练单一模型，不做论文五折模型平均；若以后补充五折结果，应单列为 paper-protocol experiment。

## 10. 论文协议与项目协议差异

| 项目 | 论文 | SolarCHIP 计划 | 证据 |
| --- | --- | --- | --- |
| 采样 | 每 2 小时 | 每天 00:00 一张 | **[论文原文] / [SolarCHIP 适配]** |
| 目标窗 | 输入后未来 24 h | 当前 UTC 日历日 | **[论文原文] / [SolarCHIP 适配]** |
| 输出 | 分开的 C+、M+ 二分类 | C+/M+/X+ 三累计头并解码四类 | **[论文原文] / [SolarCHIP 适配]** |
| 像素预处理 | 对称 log、99.9% clip、8-bit JPEG | 现有 signed-`log1p` + z-score | **[论文原文] / [SolarCHIP 适配]** |
| split | 五折 CV + 2020–2023 test | 现有 DataModule split | **[论文原文] / [SolarCHIP 适配]** |
| 框架 | TensorFlow/Keras | PyTorch Lightning + torchvision | **[论文原文] / [SolarCHIP 适配]** |
| 集成 | 五折概率平均 | 主比较单 checkpoint | **[论文原文] / [SolarCHIP 适配]** |

## 11. 原文未报告或仍待决定

- **[原文未报告]** 公开材料没有可直接加载的官方 checkpoint/test predictions。
- **[原文未报告]** 完整、锁定的 TensorFlow/CUDA 依赖环境和所有随机种子未提供。
- **[原文未报告]** 三累计头中的 X+ loss weighting、三头 loss 相对权重和四分类 checkpoint-selection 指标不存在于原任务。
- **[SolarCHIP 适配]** 当前已固定为 C+ 正负平衡、M+ grouped `2/1/8/8`、X+ 正负平衡，并在各头期望权重归一后等权平均；这仍是项目设置，不是论文三头方案。
- **[SolarCHIP 适配]** 用户已锁定只训练 ImageNet 预训练版本；实现会拒绝 `pretrained=false`，不提供从零训练或静默 fallback。

## 12. 实现后的验收建议

- **[SolarCHIP 适配]** 用可辨识编号图验证中心裁剪边界、2x4 patch 顺序和无重叠覆盖，防止 H/W 或 unfold 维度交换。
- **[SolarCHIP 适配]** 检查八个 patch 确实引用同一组 encoder/head 参数，并用反向传播确认 max patch 之外的梯度行为符合 MIL。
- **[SolarCHIP 适配]** 数值核对 `sigmoid(max(logits))` 与 `max(sigmoid(logits))`，并对极端 logits 检查稳定性。
- **[SolarCHIP 适配]** 覆盖 C+/M+/X+ 的所有阈值组合，验证最高通过阈值解码和不一致率统计。
- **[SolarCHIP 适配]** 用一个真实 DataModule batch 完成 Lightning train/validation smoke test，确认 `[B,1,1024,1024] -> [B,8,3] -> [B,3] -> [B]`。
- **[SolarCHIP 适配]** 当前没有 test split，只验收 test hooks 能构造，不报告任何 test accuracy。
## 13. 已实现训练合同（2026-08-26）

- **[SolarCHIP 适配]** `module.py::PCNN` 是可直接交给现有训练入口的 `LightningModule`，直接消费 `batch['hmi']` 与 `batch['label']`。
- **[SolarCHIP 适配]** 预处理严格执行中心半高裁剪、bicubic `224 x 448`、灰度重复三通道、`2 x 4` 个无重叠 `112 x 112` patch；不会额外施加 ImageNet mean/std。
- **[SolarCHIP 适配]** 只允许 torchvision `EfficientNet_V2_S_Weights.IMAGENET1K_V1`；权重缺失或下载失败会给出明确错误，禁止随机初始化或小 CNN fallback。
- **[论文原文] / [SolarCHIP 适配]** backbone 的所有 BatchNorm affine 参数和 running statistics 均冻结并持续处于 eval；三组新增 patch head 的 BatchNorm 正常训练。
- **[SolarCHIP 适配]** 三个 patch head 分别执行 BN -> dropout `0.2` -> Linear，八个 patch 的各头 logits 取最大值，输出 `[B,3]` C+/M+/X+ logits。
- **[SolarCHIP 适配]** 固定阈值 `0.5`，以通过阈值的最高级 head 解码为 `0AB/C/M/X`；不做单调投影。
- **[论文原文] / [SolarCHIP 适配]** C+ 使用训练集上正负总贡献相等的权重；M+ 将原 quiet/B/C/M/X=`2/2/1/8/8` 映射为 0AB/C/M/X=`2/1/8/8`；X+ 使用训练集正负平衡。每行按训练分布归一为期望权重 1，三个 head loss 等权平均。
- **[SolarCHIP 适配]** `on_fit_start` 自动从 `DataModule.datasets['train'].class_counts` 派生权重；也支持显式 `set_train_class_counts()` 和 `fetch_train_class_counts()`。计数和权重写入 checkpoint。
- **[论文原文]** optimizer 固定为 AdamW，learning rate `1e-5`、weight decay `1e-4`、15 epochs，无 scheduler。
- **[SolarCHIP 适配]** train/validation/test 整轮统计四分类 accuracy、macro-F1、balanced accuracy；每个累计 head 统计 accuracy、TSS、HSS、MCC、F1、recall、FAR；额外统计两种概率次序违反率和阈值动作不一致率。
- **[SolarCHIP 适配]** `predict_step` 返回累计 logits、概率、四分类 prediction，以及 dataloader 提供时的 `date_id`。

## 14. 已验证边界

- **[SolarCHIP 适配]** 已通过 Python 编译、合成 Lightning train/validation 闭环、自动训练类计数派生、BN 冻结、所有阈值组合解码和指标键检查。
- **[SolarCHIP 适配]** 已实际下载并加载 `IMAGENET1K_V1`，用 `[1,1,1024,1024]` 完成 `patch_logits [1,8,3]` 与 `disk_logits [1,3]` 前向验证。
- **[SolarCHIP 适配]** ImageNet 权重本次仅缓存于临时测试目录；正式训练节点首次运行仍需能访问 torchvision 权重 URL，或预先把同一权重放入该节点的 Torch Hub cache。
- **[SolarCHIP 适配]** 当前没有 test split，测试 hook 仅作为接口保留，不报告 test 数值。
