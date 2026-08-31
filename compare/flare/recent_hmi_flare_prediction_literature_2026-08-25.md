# 近期 HMI 图像耀发预测论文调研与对比模型建议

检索日期：2026-08-25
目标任务：输入某日 00:00 的一张全日面 HMI 磁图，预测该日最高 GOES 耀发等级。
建议主任务：有序四分类 `0AB / C / M / X`。

## 1. 结论先行

严格按“近期、正式同行评议、使用 HMI/磁图图像、会议或期刊层级高”筛选后，最重要的结论是：

1. **协议最接近的是 Yi et al., ApJS 2023**：每日 00:00 UT 单张全日面 LOS 磁图，预测当天是否发生 M/X 耀发，几乎与当前任务同构，只是原论文做 M1+ 二分类且混用 MDI/HMI。
2. **最新且最值得工程落地的是 Francisco et al., ApJ 2025 的 P-CNN**：单时刻全日面 HMI、未来 24 小时、代码公开；patch-wise 共享 CNN 与全日面 multiple-instance aggregation 很适合“全日面最高等级”。
3. **数据接口最精确的近期多模型比较是 Yan et al., 2024**：每天 00:00 UT 的纯 HMI 全日面单帧，比较 CNN、注意力 CNN 和 ViT；但期刊层级弱于 ApJ/ApJS/A&A/MNRAS。
4. **标签目标最接近连续“最高强度”的是 van der Sande & Muñoz-Jaramillo, ApJ 2025**：每日单张全日面磁图，回归窗口内最大 GOES XRS-B 通量，并用简单 RF 质疑复杂 CNN 的必要性。
5. **真正属于近期一线 CV 顶会的只有 ICCV 2025 DeepSWM**。它预测未来 24 小时最大 O/C/M/X 等级，但原模型使用 HMI+AIA 十通道、多时刻和长历史，不能冒充当前“单张 HMI”协议下的直接复现。
6. **Flare Transformer（ACCV 2022）不是与 ICCV/CVPR 同档的顶会，但任务接近且代码公开**：全日面 HMI、未来 24 小时四分类，并有 image-only/单帧消融。

从“任务贴合、可复现、近期性、模型互补性”综合考虑，第一批建议实现：

1. `P-CNN / ordinal MIL`；
2. `Yi 2023 DenseNet-CNN`，DQN/Double-DQN 作为可选不平衡学习消融；
3. `historical full-disk CNN + magnetogram-statistics RF`；
4. `Yan 2024 CNN-ECA 或 ViT`；
5. `Flare Transformer, image-only, k=1`；
6. `DeepSWM, HMI-only, k=1` 作为顶会代表性适配模型。

## 2. 当前仓库的真实任务协议

以下是当前 `downstream/flare` 的实际协议；所有公平对比模型应在这一协议上重新训练，而不是照搬论文原始数据划分或指标。

- 输入：单通道全日面 `hmi.M_720s.YYYYMMDD_000000_TAI.pt`，即 D 日 `00:00:00 TAI` 的 HMI 720 s 磁图；当前模型输入分辨率为 `1 × 1024 × 1024`。
- 预处理：符号保持 `log1p` 后固定 HMI mean/std z-score；训练集还有 Resize、水平/垂直翻转和旋转，验证集仅 Resize。
- 标签：对 UTC 日期 D，取 `start_time ∈ [D 00:00, D+1 00:00)` 的 catalogued flare 中最高 GOES 字母等级。
- 类别顺序：`['0AB', 'C', 'M', 'X']`，其中 `0/A/B → 0AB`。
- 当前实际训练样本：4,876 日，2010-05-01—2024-01-07；四分类计数为 `2244 / 1801 / 748 / 83`。
- 当前实际验证样本：389 日，2024-01-08—2025-02-02；四分类计数为 `5 / 120 / 222 / 42`。
- 当前配置没有独立 test split。因此，现有 validation 不能在反复选择对比模型和超参数后继续作为最终无偏测试集。

仓库证据：

- [正式 HMI-only 配置](../../downstream/flare/solar_predictor_cnn.yaml)
- [标签生成与当天最高等级定义](../../downstream/flare/data/prepare_flare_labels.py)
- [统一类别分组定义](../../downstream/flare/data/class_groups.py)
- [当前标签来源与分布摘要](../../downstream/flare/data/flare_daily_labels.summary.json)
- [训练说明](../../downstream/flare/README.md)

还需要在最终论文中明确：输入是 `00:00 TAI`，而标签窗从同名日期的 `00:00 UTC` 开始，两者相差几十秒；应审计跨午夜开始或达到峰值的事件。

## 3. 一级候选：直接任务基线

### A1. Yi 2023：与当前日级协议最接近

**Yi, Moon & Jeong, “Application of Deep Reinforcement Learning to Major Solar Flare Forecasting,” The Astrophysical Journal Supplement Series, 265:34, 2023.**
[论文 DOI](https://doi.org/10.3847/1538-4365/acb76d)

- 输入：每天 **00:00 UT** 的单张全日面 LOS 磁图；1996–2010 使用 SOHO/MDI，2011–2019 使用 SDO/HMI，统一降采样为 512×512。
- 目标：预测当天是否出现 M/X 耀发，即 `M1+ / non-M1+` 日级二分类。
- 模型：DenseNet 风格 CNN，并比较 DQN 与 Double-DQN，用 TP/TN/FP/FN reward 处理类别失衡。
- 划分：1996–2008 训练，2009–2019 测试，属于严格跨时间/跨太阳周期评估。
- 与本任务匹配度：**最高**。全日面、单帧、00:00、当天标签都一致；主要差异是二分类与 MDI/HMI 混合训练。
- 代码：截至检索日未核验到作者公开仓库。

落地建议：

- 先实现普通 DenseNet-CNN 四分类；它是最干净的“论文架构重训”版本。
- DQN/Double-DQN 不是必须首发。当前任务不是序列决策 MDP，reward 分类本质上是代价敏感学习；应与 class-weighted CE/focal/ordinal loss 严格对照，避免仅因评价函数对齐而看似提升。

### A2. P-CNN：最推荐的现代可复现基线

**Francisco et al., “Limits of Solar Flare Forecasting Models and New Deep Learning Approach,” The Astrophysical Journal, 985:108, 2025.**
[论文 DOI](https://doi.org/10.3847/1538-4357/adc56d) · [作者公开代码](https://github.com/gfrancisco20/flare_limits_pcnn) · [数据](https://doi.org/10.5281/zenodo.10465436) · [冻结版代码](https://doi.org/10.5281/zenodo.14790146)

- 输入：每个样本一张全日面 HMI LOS 磁图；原数据每 2 小时取样一次。
- 目标：未来 24 小时是否发生 C+ 或 M+ 耀发，两个独立二分类任务。
- 模型：裁去高纬极区后把全日面分成 8 个 patch；共享权重的 EfficientNetV2-S 对每个 patch 输出概率，再以最大概率聚合为全日面预测。这是弱监督 multiple-instance learning，不需要事先知道哪个活动区会耀发。
- 评测：2010–2019 做五折训练/验证；用 81 天时间块和 27 天 buffer 减弱相邻样本及同一太阳自转泄漏；2020–2023 为独立时间测试。
- 与本任务匹配度：**很高**。都是单时刻、全日面 HMI 和 24 小时预测；分辨率链也从 1024 开始。
- 不一致：原论文是 C+/M+ 二分类、每 2 小时一个重叠窗口；当前是每天 00:00 一张、按 UTC `start_time` 日历日统计四分类。

落地建议：

- 第一版先忠实实现 `C+` 与 `M+` 两个二分类头，用现有四类标签派生二值目标，验证网络是否接通正确。
- 四分类主表建议改成三个有序阈值头：`P(y≥C)`、`P(y≥M)`、`P(y≥X)`；每个阈值在 patch 维度做 max/noisy-OR 聚合，并约束概率单调。这是面向当前任务的适配，不应写成原论文原样复现。
- 同时报告普通全图 EfficientNet 与 P-CNN，判断提升来自 patch-MIL 还是仅来自更强 backbone。

### A3. Yan 2024：纯 HMI、每天 00:00、多种常规模型

**Yan et al., “A Real-time Solar Flare Forecasting System with Deep Learning Methods,” Astrophysics and Space Science, 369:110, 2024.**
[论文 DOI](https://doi.org/10.1007/s10509-024-04374-8) · [Springer 正式页](https://link.springer.com/article/10.1007/s10509-024-04374-8)

- 输入：2010-05-01—2023-03-31，每天 **00:00 UT** 的 SDO/HMI 全日面单帧磁图。
- 目标：未来 24 小时 `C+` 与 `M+` 两个二分类任务。
- 模型：CNN、CNN-SE、CNN-CBAM、CNN-ECA、ViT。
- 划分：按时间顺序构造十折交叉验证，不是随机拆散相邻图片。
- 与本任务匹配度：**很高**；只需要统一为四分类输出和当前日期清单。
- 期刊边界：正式同行评议，但通常不与 ApJ/ApJS/A&A/MNRAS 同层描述。
- 代码：截至检索日未核验到作者公开仓库。

落地建议：无需一次实现全部五种。选普通 CNN、CNN-ECA、ViT 三个即可覆盖卷积、轻量通道注意力和 Transformer。

### A4. 每日全日面 XRS 通量回归：目标定义最接近

**van der Sande & Muñoz-Jaramillo, “The Struggles of Developing an Operational Machine Learning Model for Flare Forecasting: Recasting the Problem as a Regression onto X-Ray Flux,” The Astrophysical Journal, 984:87, 2025.**
[论文 DOI](https://doi.org/10.3847/1538-4357/ad8de5) · [官方代码仓库](https://github.com/SwRI-IDEA-Lab/idea-lab-flare-forecast)

- 输入：跨四个太阳活动周、来自五种仪器的每日单张全日面 LOS 磁图，其中包括 SDO/HMI。
- 目标：预测给定窗口内最大 GOES XRS-B 1–8 Å 通量，而不是先人为切成二分类。
- 比较：线性回归、MLP、随机森林、CNN，以及图像和标量特征混合模型；论文的核心结果是复杂 CNN 并未明显优于简单 RF。
- 与本任务匹配度：**很高**。每日单帧、全日面、预测最大强度几乎就是当前目标的连续版本。
- 不一致：论文标签是连续 XRS 最大通量；当前 `0` 表示“无 catalogued event”，不等价于 XRS 背景为零。论文还使用总有符号/无符号磁通、极值和耀发历史等标量特征。

落地建议：

- 在当前四分类协议下先实现 `CNN image-only` 和 `RF magnetogram-statistics-only` 两个公平基线。
- 若要忠实复现回归任务，需要另行从连续 GOES XRS 数据生成 `max_xrs_flux_24h`；不能用 `0AB/C/M/X` 的整数 ID 冒充连续通量。
- RF 是重要科学对照：若 SolarCHIP 只超过弱 CNN、却没有超过简单磁通统计 RF，结论说服力不足。

### A5. 历史全日面概率预测：单帧 HMI 的直接前作

**van der Sande, Muñoz-Jaramillo & Chatterjee, “Probabilistic Solar Flare Forecasting Using Historical Magnetogram Data,” The Astrophysical Journal, 955:148, 2023.**
[论文 DOI](https://doi.org/10.3847/1538-4357/acf49a) · [代码仓库](https://github.com/SwRI-IDEA-Lab/idea-lab-flare-forecast)

- 输入：每日单张全日面磁图，覆盖 1975–2022，包含 HMI/MDI 和更早历史仪器。
- 目标：未来 24 小时 M+ 耀发的校准概率。
- 模型：CNN 提取图像特征，logistic regression 使用磁图统计量和历史耀发信息，最后做概率集成。
- 主要价值：单帧磁图 CNN 提供的信息不显著多于少量标量磁图特征，而耀发历史更有预测力。
- 与本任务匹配度：**很高，但输出为 M+ 二分类概率**。

落地建议：与 A4 共享实现；把“只用图像”“只用磁图标量”“图像+标量”分栏。耀发历史超出当前 HMI-only 输入，只能作为额外上界，不能混入公平主表。

### A6. Flare Transformer：任务接近、现代且可复现

**Kaneda et al., “Flare Transformer: Solar Flare Prediction Using Magnetograms and Sunspot Physical Features,” ACCV 2022 main proceedings.**
[CVF 正式论文页](https://openaccess.thecvf.com/content/ACCV2022/html/Kaneda_Flare_Transformer_Solar_Flare_Prediction_using_Magnetograms_and_Sunspot_Physical_ACCV_2022_paper.html) · [DOI](https://doi.org/10.1007/978-3-031-26284-5_27) · [作者代码](https://github.com/keio-smilab21/flare_transformer)

- 输入：全日面 HMI LOS 磁图和 90 维太阳黑子物理特征；原完整模型使用四个逐小时观测。
- 目标：未来 24 小时最大 O/C/M/X 等级。
- 与本任务匹配度：**高**，尤其是 `without physical features`、`k=1` 单帧消融。
- 会议边界：ACCV 是正式且较强的计算机视觉会议，但不能写成与 CVPR/ICCV/ECCV 同档的一线顶会。
- 评测风险：公开配置中 validation 与 test 可落在同一年；接入时必须改为仓库统一且独立的时间切分。

落地建议：只实现 `image-only, k=1` 作为公平主比较；物理特征版另列，不与 HMI-only SolarCHIP 混在同一栏。

### A7. DeepSWM：近期真正的顶会代表

**Nagashima & Sugiura, “Deep Space Weather Model: Long-Range Solar Flare Prediction from Multi-Wavelength Images,” ICCV 2025 main conference, pp. 9396–9405.**
[ICCV/CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2025/html/Nagashima_Deep_Space_Weather_Model_Long-Range_Solar_Flare_Prediction_from_Multi-Wavelength_ICCV_2025_paper.html) · [DOI](https://doi.org/10.1109/ICCV51701.2025.00877) · [作者代码](https://github.com/keio-smilab25/DeepSWM) · [FlareBench](https://huggingface.co/datasets/sh237/FlareBench)

- 输入：每个时刻 10 通道全日面图像（1 个 HMI LOS + 9 个 AIA），近期四帧，并用最长 672 小时历史建模长期依赖。
- 目标：未来 24 小时最大 O/C/M/X 耀发等级。
- 方法：solar spatial encoder、长时状态空间模型和针对太阳图像的 sparse masked-autoencoder 预训练。
- 优点：ICCV 2025 主会，是本次筛选中唯一无歧义的一线 CV 顶会论文；代码和 FlareBench 已公开。
- 与本任务匹配度：**标签高、输入低**。原模型的主要贡献依赖多波长和时间历史。

落地建议：

- 公平主表只能实现 `HMI-only, k=1`，并明确称“DeepSWM-derived single-frame ablation”，不能声称复现原论文完整 DeepSWM。
- 完整模型适合另做“额外模态/时间信息上界”，不适合与单张 HMI 模型宣称公平比较。
- 当前约 4,876 个训练日远少于其逐小时 FlareBench 样本规模；应先评估参数量和过拟合风险。

## 4. 二级候选：近期领域期刊中的单帧 HMI 图像模型

| 年份 | 论文与正式来源 | 图像/任务 | 与当前任务的主要差异 | 建议用途 |
|---|---|---|---|---|
| 2024 | Zhang et al., [Causal Attention Deep-learning Model for Solar Flare Forecasting](https://doi.org/10.3847/1538-4365/ad7386), ApJS 274:38 | 单幅 HMI LOS 活动区图，ResNet-18 + Causal Attention；12/24/48 h M/X | 活动区裁块、二分类 | 近期、有[作者代码](https://github.com/deepsolar/CasualNet)，可做全日面适配；注意仓库名拼作 `CasualNet` |
| 2025 | Li et al., [Intelligent Forecasting for Solar Flares Using Magnetograms from SDO/SHARP, SDO/HMI, and ASO-S/FMG](https://doi.org/10.3847/1538-4365/add149), ApJS 278:63 | 比较 CNN、CNN-BiLSTM、ViT、MViT；24 h M+ | 活动区/裁剪磁图，含单/多 AR，且多为时序二分类 | `CNN / ViT / MViT` 架构消融；截至检索日未找到明确作者代码 |
| 2025 | Li et al., [Forecasting Major Flares Using Magnetograms and Knowledge-informed Features](https://doi.org/10.3847/1538-4365/ade687), ApJS 279:46 | 图像 CNN、CNN-BiLSTM、CNN-BiLSTM-Attention、ViT；另有 31 个知识特征模型；24 h M+ | SHARP 活动区为主，40 帧/AR；最佳模型是知识特征 iTransformer | 选择 CNN/ViT 结构；不能把 iTransformer 的论文分数当 HMI 图像分数 |
| 2025 | Vong et al., [Bypassing the Static Input Size ... Spatial Pyramid Pooling](https://doi.org/10.1051/0004-6361/202449671), A&A 695:A65 | SHARP LOS 单幅活动区图；SPP-CNN；24 h C+、M+ | 可变大小 AR patch、二分类；当前全日面已统一为 1024 | SPP 与 resize-CNN 对照，优先级低于 P-CNN |
| 2023 | Zheng et al., [Multiclass Solar Flare Forecasting Models with Different Deep Learning Algorithms](https://doi.org/10.1093/mnras/stad839), MNRAS 521:5384–5399 | SHARP LOS 图像 CNN/H-CNN/OAO-CNN；No/C/M/X，24 h | 活动区级，图像缩到 128；最佳 headline 模型使用物理参数序列 | 四分类 CNN/OAO-CNN 结构参考；[公开数据](https://github.com/FlarePrediction/Repository/tree/papers/paper12) |
| 2022 | Sun et al., [Predicting Solar Flares Using CNN and LSTM on Two Solar Cycles](https://doi.org/10.3847/1538-4357/ac64a6), ApJ 931:163 | 单幅活动区 LOS 图 CNN，与磁参数 LSTM 集成；24 h M/X | AR 级、二分类、跨仪器 | 可复现 CNN 与 LOS 极性伪影审计；[代码](https://github.com/ZeyuSun/flare-prediction-smarp) |
| 2022 | Liu et al., [Deep Learning Based Solar Flare Forecasting Model II: Influence of Image Resolution](https://doi.org/10.3847/1538-4357/ac99dc), ApJ 941:20 | 单幅 HMI LOS 活动区图；AlexNet、ResNet-18、SqueezeNet；48 h M/X | AR 级、48 h、二分类 | 常规 CNN/分辨率敏感性基线 |
| 2022 | Li et al., [Knowledge-Informed Deep Neural Networks](https://doi.org/10.1029/2021SW002985), Space Weather 20 | 单幅 SHARP LOS 活动区图和知识约束；48 h M+ | AR 级、48 h、融合版含额外先验 | 纯 CNN 分支可参考；融合分支只作额外输入上界 |

上述活动区论文不能直接用“后来实际耀发的 NOAA AR”来裁当前全日面图片，否则会把标签位置泄漏进输入。若适配为 AR 模型，必须用预测时刻已可获得的 AR 检测结果处理全部候选 AR，再聚合成日级全日面预测。

## 5. 时序、混合输入和 2026 年最新方向

这些论文正式且较新，但不应在当前“单帧 00:00”设定下原样放进公平主表。

| 年份 | 论文 | 原始输入和输出 | 为什么暂缓 |
|---|---|---|---|
| 2022 | Deshmukh et al., [Decreasing False-alarm Rates in CNN-based Solar Flare Prediction Using SDO/HMI Data](https://doi.org/10.3847/1538-4365/ac5b0c), ApJS 260:9 | 四幅 SHARP 径向磁图（0/−3/−6/−9 h），VGG-16；第二阶段加入 ERT、SHARP 和拓扑特征；未来 12 h M1+ | 同时改变 AR/full-disk、单帧/时序、12/24 h 和输入特征；[代码](https://github.com/vrd1243/solar-flare-hybrid-apj)可用于 false-alarm 研究 |
| 2022 | Guastavino et al., [Implementation Paradigm ... Video Data](https://doi.org/10.1051/0004-6361/202243617), A&A 662:A105 | 过去 24 h、40 帧 SHARP LOS 视频，LRCN 预测未来 24 h C+/M+ | 视频输入信息量远高于单帧；更适合借鉴 AR-disjoint 和重复划分方法 |
| 2024 | Grim & Gradvohl, [Magnetogram Sequences Learning with Multiscale Vision Transformers](https://doi.org/10.1007/s11207-024-02276-0), Solar Physics 299:33 | HMI 活动区磁图序列，MViT，未来 24/48 h M+ | AR 级、时序、二分类；[代码](https://github.com/lfgrim/SFF_MagSeq_MViTs)适合第二阶段时序扩展 |
| 2025 | Xu et al., [Solar Flare Forecasting Using Hybrid Neural Networks](https://doi.org/10.3847/1538-4365/ada281), ApJS | 24 h SHARP 图像序列 + 16 个磁场特征，CNN-TCN，未来 24 h C+/M+ | 多模态时序，不是 HMI 单图基线 |
| 2026 | Doria Rosales et al., [Advancing Solar Flare Forecasting with a Deep Learning Approach Using Multimodal Inputs](https://doi.org/10.3847/1538-4357/ae3827), ApJ 998:231 | 活动区短时序；LoS 磁图基线，再加入 continuum、AIA 193/304 和 SHARP 参数；未来 2 h C5+ | 时间窗、空间单位和模态均不同；适合 related work 最新进展 |
| 2026 | Alatoom et al., [Multi-Wavelength Transformer-Based 24-Hour Solar Flare Forecasting](https://doi.org/10.1029/2026SW005010), Space Weather | 每个 AR 四个观测、每时刻十个共配准波段，CNN + 冻结 DeiT-Tiny + temporal Transformer；24 h M+ | 多波长 AR 时序，只适合作为额外信息上界 |
| 2026 | Wu et al., [Prediction of Major Solar Flares Using Interpretable Class-dependent Reward](https://doi.org/10.1093/mnras/stag349), MNRAS 547 | SHARP LOS 磁图/物理特征的 CNN、CNN-BiLSTM、Transformer 与 CDR 版本；24 h M+ | CDR 对图像支路未稳定优于普通 CNN；不建议作为第一批核心模型 |
| 2026 | Lv et al., [FlareCast](https://doi.org/10.3847/1538-4365/ae43dc), ApJS 283:46 | SHARP CEA 活动区径向磁图时序，贝叶斯网络输出未来 12/24/48 h 最大 XRS 通量分布，可聚合 N/C/M/X | 输出形式很相关，但空间与时间输入不匹配；适合后续不确定性扩展 |

## 6. DeFN 和 FDDLM 之后的关系

- **DeFN-R（ApJ 2020）**和 Operational DeFN（Earth, Planets and Space 2021）是 DeFN 的后续，但主要输入是由 HMI/AIA 等观测计算出的活动区物理特征，不是 raw-HMI 图像端到端模型。因此不应把它们作为“近期 HMI 图像模型”核心答案。
- **FDDLM** 后续有 full-disk/AR ensemble 和归因分析版本，但近期最值得替代或扩展它的全日面图像方法是 **ApJ 2025 P-CNN**：它保留全日面预测，同时在弱监督下给出 patch 级风险和位置，不依赖事后指定会耀发的 AR。
- DeepSWM 的公开比较仍包含 DeFN/DeFN-R、Flare Transformer 等旧基线，说明它们仍是领域标准参照；但对当前仓库而言，Yi 2023、P-CNN 与每日全日面回归模型更接近真实输入协议。

## 7. 公平对比协议：落地前必须固定

### 7.1 所有主表模型共享

1. 完全相同的 `date_id` 样本列表，不允许各模型按自己的缺失规则重新筛选。
2. 同一张 D 日 `00:00:00 TAI` HMI `M_720s` 全日面图。
3. 同一 `log1p + z-score` 科学预处理；若论文模型需要不同输入尺度，把共享预处理与模型内 resize 分开记录。
4. 同一有序类别及 ID：`0AB/C/M/X → 0/1/2/3`。
5. 同一 train/validation/test 时间边界、随机种子集合、选模指标和训练预算。
6. 预训练版与从零训练版分栏；必须审计预训练图像日期是否覆盖 validation/test。
7. 类别权重、阈值和采样比例只能由 train 计算，不能读取 validation/test 分布。

### 7.2 标签口径不能偷换

当前标签按 UTC `start_time` 落入日历日聚合，而多数论文按观测时刻之后 24 小时的 GOES peak flux 聚合。虽然 00:00 观测使两个窗口看起来相近，它们仍可能因 TAI/UTC、跨午夜开始/峰值和正在持续的耀发而不同。接入论文架构时应统一使用当前标签，不应混用各论文原始标签再比较。

若开展 ApJ 2025 的连续回归实验，应另建连续 XRS 目标，并把它作为独立任务；catalog 中的 `0` 只表示当天无已编目事件，不表示背景 XRS 通量为零。

### 7.3 需要增加独立测试集

当前仅有 train 和 validation。继续用 2024–2025 validation 选择模型后再汇报它，会形成验证集泄漏。建议在开始实现前冻结一个独立、严格时间后置的 test，或重新规划 train/validation/test；具体边界应结合实际 HMI 存在日期确定。

对于全日面高时间自相关数据，可参考 P-CNN：在切分边界加入至少一个 Carrington rotation 量级的 gap，并检查同一持续活动区是否横跨 split。论文中的 AR-random split、重叠逐小时窗口或负样本下采样 test 都不能直接照搬。

### 7.4 统一指标

四分类主表至少报告：

- macro-F1、balanced accuracy、每类 precision/recall/F1、confusion matrix；
- 有序误差：class-index MAE 或 GMGS；
- 由四类概率导出的 `C+`、`M+`、`X+` TSS、BSS 和 reliability diagram；
- persistence baseline，以及“相对前一日是否发生等级变化”的 activity-change 子集表现。

不能只报告 accuracy。当前 validation 的 `0AB` 仅 5 个，而 `M` 有 222 个，单一 accuracy 或只看一个阈值会严重掩盖类别失衡。

## 8. 建议的第一轮实现矩阵

| 模型 | 公平主表版本 | 论文忠实度 | 工作量 | 推荐级别 |
|---|---|---:|---:|---:|
| 普通 CNN/ResNet18 | 单张 HMI，四类 softmax | 通用基线 | 低 | 必做 |
| 磁图统计 RF | 仅从同一张 HMI 计算 train-defined 标量 | 对应 ApJ 2025 简单基线 | 低 | 必做 |
| P-CNN | 8 patch，共享 CNN，ordinal MIL 三阈值头 | 中；输出头为任务适配 | 中 | **最高** |
| Yi DenseNet-CNN | 单张全日面 HMI，四类输出 | 高；去掉 MDI 与二分类差异 | 低—中 | **最高** |
| Yan CNN-ECA / ViT | 单张全日面 HMI，四类输出 | 高；输出头适配 | 低—中 | **高** |
| Historical full-disk CNN | 五层 CNN，image-only 四分类/ordinal | 高；去掉额外历史特征 | 低—中 | **高** |
| Flare Transformer | image-only、k=1、四分类 | 论文已有消融，较高 | 中 | **高** |
| CausalNet | ResNet18 + causal attention，全日面四分类 | 中低；原论文为 AR 二分类 | 中 | 可选 |
| DeepSWM-derived | HMI-only、k=1、无 AIA/长历史 | 低；必须明确称消融 | 高 | 顶会代表，第二阶段 |
| OAO/H-CNN | 全日面适配、四分类 | 低；原论文为 AR patch | 中 | 可选 |
| MViT/CNN-BiLSTM | 单帧后时序模块失去主要意义 | 很低 | 高 | 暂缓 |

最干净的论文叙事是：

> 在完全相同的单日全日面 HMI 输入、时间切分与 `0AB/C/M/X` 标签下，比较简单磁通统计、普通全图 CNN、弱监督 patch-MIL、注意力 CNN、Transformer 和太阳预训练编码器；另以多模态/时序模型作为非同输入上界，而不混入公平主表。

## 9. 顶会/顶刊与检索边界

- **ICCV 2025 DeepSWM**：一线 CV 顶会主会，可明确称顶会论文。
- **ACCV 2022 Flare Transformer**：正式主会且较强，但不应写成与 CVPR/ICCV/ECCV 同档。
- **AAAI 2021 Shape-based Feature Engineering** 属于 IAAI Emerging Applications technical track，不是 AAAI main research track，而且其输入是由 SHARP 图像提取的几何/拓扑特征，不是端到端 raw-image 模型。
- ApJ、ApJS、A&A、MNRAS、Space Weather 是本领域主流或旗舰同行评议期刊，但不是 ML 顶刊；“顶刊”应按天文/空间天气学科语境谨慎表述。
- 截至检索日，没有找到满足“2020–2026、raw-HMI 图像耀发预测、正式 CVPR/ECCV/NeurIPS/ICLR 主会或 TPAMI/TGRS/Pattern Recognition 正刊”的其他明确论文。
- IEEE Big Data、DSAA、AIKE、Frontiers、book chapter、workshop 和仅有 arXiv 的预印本不能包装成顶会。
- 检索以 2022–2026 为主，补充与 DeFN/FDDLM 直接衔接的早期工作；优先核对出版社正式页面、DOI、正式 PDF、作者代码和数据存档。
- 本报告是面向工程落地的重点检索，不是穷尽式 PRISMA 系统综述。

