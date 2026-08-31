# A1/A2/A7 耀发对比模型

状态：**三套 PyTorch Lightning 复现已进入可训练版本；均为论文结构在 SolarCHIP 日级四分类协议上的适配，不宣称指标级严格复刻。**

训练命令、已采用推荐配置和之后可能调整的项目统一记录在 [`TRAINING.md`](TRAINING.md)。较广的候选论文筛选保留在[近期 HMI 耀发预测综述](recent_hmi_flare_prediction_literature_2026-08-25.md)。

## 模型索引

| 编号 | 项目 | 网络与配置 | 运行入口 | 角色 |
| --- | --- | --- | --- | --- |
| A1 | [Yi 2023 DQN adaptation](yi2023_dqn/README.md) | `yi2023_dqn/module.py`；`configs/compare_flare/yi2023_dqn.yaml` | `yi2023_dqn/train_dqn.sh` | 与 00:00 日级协议最接近；另有 supervised control |
| A2 | [P-CNN adaptation](pcnn/README.md) | `pcnn/module.py`；`configs/compare_flare/pcnn.yaml` | `pcnn/train.sh` | ImageNet 预训练的现代 patch-MIL 基线 |
| A7 | [DeepSWM HMI-only adaptation](deepswm/README.md) | `deepswm/model.py`、`dataset.py`、`module.py`；`configs/compare_flare/deepswm.yaml` | `deepswm/train.sh` | ICCV 2025 时空建模基线 |

## 证据标签

- **[论文原文]**：来自正式论文正文、表格或补充材料。
- **[官方代码]**：来自作者仓库的固定 commit。
- **[SolarCHIP 适配]**：为当前数据、任务和 Lightning 训练入口作出的项目决定，不是论文设置。
- **[原文未报告]**：公开材料不足以唯一复现，绝不补造成论文数值。

## 统一 SolarCHIP 协议

- **[SolarCHIP 适配]** 复用现有 `FlareDataset` / `DataModuleFromConfig`、HMI 可用索引、标签和 split。
- **[SolarCHIP 适配]** 基础输入为 D 日 00:00 的单通道 HMI；标签为同一 UTC 日历日最高耀发等级，`0AB/C/M/X -> 0/1/2/3`。
- **[SolarCHIP 适配]** 保留 dataloader 的 signed-`log1p` + HMI z-score；模型内部只做结构需要的 resize、crop、patch 或序列组装。
- **[SolarCHIP 适配]** train 为日期 ID `[0,5000)`、validation 为 `[5000,5400)`，实际样本数由最新 HMI 可用索引过滤。
- **[SolarCHIP 适配]** validation 与未来 test 都统计最终四分类 accuracy 和 macro-F1，同时保留各论文指标。
- **[SolarCHIP 适配]** 当前没有 test split；代码只保留 `test_step` / `test_accuracy`，不创建日期、不报告测试结果。
- **[SolarCHIP 适配]** 正式实验只有一个 seed，固定为 `42`；checkpoint 由 validation macro-F1 选择。

## 输出合同

| 模型 | 输入 | 输出与解码 |
| --- | --- | --- |
| A1 | `[B,1,1024,1024]`，模型内 area resize 至 512 | `[B,3,2]` C+/M+/X+ Q 值；每头 argmax 后取最高阳性头 |
| A2 | `[B,1,1024,1024]`，模型内中心裁剪、resize、切 8 patch | `[B,3]` 累计 MIL logits；固定 0.5，取最高阳性头 |
| A7 | `[B,T,1,1024,1024]`，默认 `T=1`，模型内 resize 至 256 | 原生 `[B,4]` logits，直接 argmax 为 `0AB/C/M/X` |

A1/A2 的累计目标固定为 `C+=1[y>=1]`、`M+=1[y>=2]`、`X+=1[y>=3]`。A2 不做论文没有的单调投影，只记录概率/阈值不一致率。

## A7 连续日窗口合同

- 窗口 `D-(T-1),...,D` 的每一天 HMI 都必须存在，不能填洞。
- 所有日期必须位于同一 split；标签取最后一天 D。
- 随机翻转与旋转对整段序列保持一致。
- `T=1` 是首个正式设置，但此时 LT-SSM 不具有跨日依赖；它不等价于论文 672 小时历史。

## 固定来源

- **[论文原文]** Yi, Moon, & Jeong (2023), *Application of Deep Reinforcement Learning to Major Solar Flare Forecasting*, ApJS 265:34，[DOI](https://doi.org/10.3847/1538-4365/acb76d)。
- **[论文原文]** Francisco et al. (2025), *Limits of Solar Flare Forecasting Models and New Deep Learning Approach*, ApJ 985:108，[正式 PDF](https://eprints.whiterose.ac.uk/id/eprint/226838/1/Francisco_2025_ApJ_985_108.pdf)。
- **[官方代码]** P-CNN 固定 commit [`9a7ef5984fd3d714bb78da0edad3c5a10dd80641`](https://github.com/gfrancisco20/flare_limits_pcnn/tree/9a7ef5984fd3d714bb78da0edad3c5a10dd80641)，`sundl` 固定 commit [`48a79c9acffa26c86800d6c2f136d328104d318c`](https://github.com/gfrancisco20/sundl/tree/48a79c9acffa26c86800d6c2f136d328104d318c)。
- **[论文原文]** Nagashima & Sugiura (2025), *Deep Space Weather Model: Long-Range Solar Flare Prediction from Multi-Wavelength Images*, ICCV 2025，[CVF 正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Nagashima_Deep_Space_Weather_Model_Long-Range_Solar_Flare_Prediction_from_Multi-Wavelength_ICCV_2025_paper.html)。
- **[官方代码]** DeepSWM 固定 commit [`278779fb8fc99cab2c68b5ff583ccc76e68c9cc0`](https://github.com/keio-smilab25/DeepSWM/tree/278779fb8fc99cab2c68b5ff583ccc76e68c9cc0)。

## 验证边界

代码验收包含配置解析、网络构造、形状/解码单元测试、合成 batch 前后向和 Lightning 闭环。真实原图读取还依赖运行机器上的 `global_settings.DATA_ROOT`；这属于部署挂载检查，不以伪造数据代替。当前没有 test split，因此任何验证都不会被表述成 test 结果。
