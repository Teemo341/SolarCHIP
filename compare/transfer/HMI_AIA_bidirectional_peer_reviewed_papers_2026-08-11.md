# HMI↔AIA 双向图像转换：正式论文核验与基线选择

检索日期：2026-08-11

## 1. 直接结论

如果“同时能做”是指**同一篇正式发表、同行评议论文，在一个联合模型中训练并定量评估 HMI→AIA 和 AIA→HMI 两个方向**，本次检索**没有找到满足条件的论文**。

正式文献目前形成的是两条独立证据链：

- HMI→AIA：Park et al. (2019)、Galvez et al. (2019) 等；
- AIA→HMI：Kim et al. (2019)、Sun et al. (2023)、Dannehl et al. (2024)、Sayez et al. (2025) 等。

因此，最严谨的研究方案是用**两个方向的正式任务论文**证明任务可行性，再用 Jarolim et al. (2025) 证明双生成器、循环一致的太阳图像框架可行，并将自己的方法表述为 **coupled bidirectional HMI↔AIA translation**，而不是声称已有论文完成了完全相同的双向任务。

## 2. 最推荐的正式论文组合

### 组合 A：单波段、LOS 磁图，最接近严格对称任务

| 方向 | 推荐论文 | 实际任务 | 用途与边界 |
|---|---|---|---|
| HMI→AIA | Park et al. (2019), *ApJL* 884, L23, [DOI 10.3847/2041-8213/ab46bb](https://doi.org/10.3847/2041-8213/ab46bb) | HMI LOS→AIA 九个 UV/EUV 通道，其中包括 304 Å；比较 L1 与 L1+cGAN | 选择其 304 Å 输出即可构成 HMI LOS→AIA 304 分支；2011–2016 训练、2017 测试 |
| AIA→HMI | Sayez et al. (2025), *A&A* 702, A83, [DOI 10.1051/0004-6361/202555324](https://doi.org/10.1051/0004-6361/202555324) | AIA 304 Å→HMI LOS/Bz；比较 Pix2PixCC、非对抗 U-Net 与 I2IwFiLM | 对 GAN 幻觉和“全零磁图也能得到较高 SSIM”的问题有专门分析，适合作为科学可靠性基线 |

这套组合最适合把首个双向原型定义为：

`HMI LOS ↔ AIA 304 Å`

不过两篇论文的数据版本、裁剪范围和训练协议并不完全相同，不能直接横向比较论文中的指标；实际实验应在同一配准数据和同一时间隔离划分上重训两个方向。

### 组合 B：多波段 AIA、向量磁图，最适合完整系统

| 方向 | 推荐论文 | 实际任务 | 用途与边界 |
|---|---|---|---|
| HMI→AIA | Galvez et al. (2019), *ApJS* 242, 7, [DOI 10.3847/1538-4365/ab1005](https://doi.org/10.3847/1538-4365/ab1005) | 对齐的 HMI Bx/By/Bz→九通道 AIA，并提供 SDOML 数据与 CNN baseline | 最适合作为统一数据、配准方式和正向基线的正式依据 |
| AIA→HMI | Dannehl, Delouille & Barra (2024), *Earth and Space Science* 11, e2023EA002974, [DOI 10.1029/2023EA002974](https://doi.org/10.1029/2023EA002974) | 多种 AIA 通道组合→HMI Bz；附录还展示 Bx/By/Bz 向量磁图生成 | 使用 SDOML，对通道和网络配置做系统实验；是当前最直接的多波段/向量反向依据 |

这套组合共享 SDOML 数据体系，最适合构建：

`HMI (Bx, By, Bz) ↔ AIA (193, 304, 可选 1600 Å或更多通道)`

但 Dannehl et al. 明确显示：EUV 中缺少直接的磁极性信息，活动区磁足点的正负极性可能生成错误。向量输出的存在不等于 signed magnetic field 已被唯一、可靠地反演。

## 3. 各方向的正式论文清单

### 3.1 HMI→AIA

1. **Park et al. (2019)**, “Generation of Solar UV and EUV Images from SDO/HMI Magnetograms by Deep Learning,” *The Astrophysical Journal Letters*, 884, L23. [机构记录与摘要](https://khu.elsevierpure.com/en/publications/generation-of-solar-uv-and-euv-images-from-sdohmi-magnetograms-by-2/)；[DOI](https://doi.org/10.3847/2041-8213/ab46bb)。
   - HMI LOS→AIA 94、131、171、193、211、304、335、1600、1700 Å。
   - 任务最直接，是正向分支的首选正式依据。

2. **Galvez et al. (2019)**, “A Machine-learning Data Set Prepared from the NASA Solar Dynamics Observatory Mission,” *The Astrophysical Journal Supplement Series*, 242, 7. [同行评议版本与全文](https://eprints.gla.ac.uk/187270/)；[DOI](https://doi.org/10.3847/1538-4365/ab1005)。
   - 建立同步、配准的 SDOML 数据，并以 HMI 向量分量→AIA 多通道作为示例任务和 baseline。
   - 更偏数据集和基准论文，但可正式支撑多通道正向转换。

3. **Dash et al. (2024)**, “High Resolution Solar Image Generation Using Generative Adversarial Networks,” *Annals of Data Science*, 11, 1545–1561. [同行评议机构记录](https://researchwith.njit.edu/en/publications/high-resolution-solar-image-generation-using-generative-adversari/)；[DOI](https://doi.org/10.1007/s40745-022-00436-2)。
   - HMI LOS→AIA 304 Å；比较 Pix2Pix 与 Pix2PixHD，输出分辨率为 1024×1024。
   - 可作为高分辨率 GAN 正向基线；其论文内指标不能与采用不同时间划分的数据直接比较。

### 3.2 AIA→HMI：同一前盘、同时刻的直接转换

1. **Sun et al. (2023)**, “Solar Active Region Magnetogram Generation by Attention Generative Adversarial Networks,” *Research in Astronomy and Astrophysics*, 23, 025003. [期刊全文](https://www.raa-journal.org/issues/all/2023/v23n2/202302/P020230302526668355038.pdf)；[DOI](https://doi.org/10.1088/1674-4527/acaa92)。
   - 活动区 AIA 304 Å patch→HMI LOS；RHP-attention Pix2Pix。
   - 是局部活动区反向生成的正式基线，不是完整全日面方案。

2. **Dannehl, Delouille & Barra (2024)**, “An Experimental Study on EUV-To-Magnetogram Image Translation Using Conditional Generative Adversarial Networks,” *Earth and Space Science*, 11, e2023EA002974. [出版社全文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002974)；[DOI](https://doi.org/10.1029/2023EA002974)。
   - 系统比较单/多 AIA 波段→HMI Bz，并展示向量磁图输出。
   - 全局 SSIM 容易被大量近零背景抬高；局部极性错误必须单独评估。

3. **Sayez et al. (2025)**, “Mitigating hallucination with non-adversarial strategies for image-to-image translation in solar physics,” *Astronomy & Astrophysics*, 702, A83. [期刊全文](https://www.aanda.org/articles/aa/pdf/2025/10/aa55324-25.pdf)；[DOI](https://doi.org/10.1051/0004-6361/202555324)。
   - AIA 304 Å→HMI LOS/Bz，重点比较 GAN 与非对抗方法的物理幻觉。
   - 适合作为反向分支的可靠性与指标设计依据。

### 3.3 AIA/EUV→HMI：远面或带历史磁场先验

这些论文是正式发表，但与“纯 AIA 单帧→当前 HMI”的定义不同，引用时要注明附加信息和验证条件。

1. **Kim et al. (2019)**, “Solar farside magnetograms from deep learning analysis of STEREO/EUVI data,” *Nature Astronomy*, 3, 397–400. [出版社页面](https://www.nature.com/articles/s41550-019-0711-5)；[DOI](https://doi.org/10.1038/s41550-019-0711-5)。
   - 以配对 AIA 304 Å→HMI LOS 训练，再应用于 STEREO/EUVI 304 Å。
   - Liu et al. (2021) 对时间划分泄漏和磁极性可辨识性提出正式质疑，因此不宜作为唯一反向依据。

2. **Sun et al. (2022)**, “A Dynamic Deep-learning Model for Generating a Magnetogram Sequence from an SDO/AIA EUV Image Sequence,” *ApJS*, 262, 45. [DOI](https://doi.org/10.3847/1538-4365/ac85c0)。
   - AIA EUV 序列→HMI LOS 序列；适合研究时序模型，但尚未找到同等成熟的 HMI 序列→AIA 序列正式对称基线。

3. **Jeong et al. (2022)**, “Improved AI-generated Solar Farside Magnetograms by STEREO and SDO Data Sets and Their Release,” *ApJS*, 262, 50. [DOI](https://doi.org/10.3847/1538-4365/ac8d66)。
   - 当前 EUV 之外，还输入上一太阳自转的 AIA+真实 HMI reference；不能称为纯 EUV→HMI。

4. **Jeong et al. (2025)**, “Artificial-intelligence-based Reconstruction of Solar Farside Vector Magnetograms from Multispacecraft Extreme-ultraviolet Data,” *ApJS*, 281, 63. [机构记录](https://khu.elsevierpure.com/en/publications/artificial-intelligence-based-reconstruction-of-solar-farside-vec/)；[DOI](https://doi.org/10.3847/1538-4365/ae21b8)。
   - AIA/EUVI/EUI 171+304 Å 与 SFT 磁场先验→HMI 风格的向量磁图。
   - 适合需要可靠极性和远面应用的扩展路线，但不是纯跨模态反演。

## 4. 能否用正式论文支撑“一个双向模型”

可以，但应把**任务证据**和**联合架构证据**分开引用。

**Jarolim et al. (2025)**, “A deep learning framework for instrument-to-instrument translation of solar observation data,” *Nature Communications*, 16, 3157. [出版社全文](https://www.nature.com/articles/s41467-025-58391-4)；[DOI](https://doi.org/10.1038/s41467-025-58391-4)。

该论文正式验证了太阳观测中的两个生成器、A→B→A 和 B→A→B 循环一致训练，也支持两域通道数不同。因此它可以支撑“联合双向架构”设计。但其主要任务是仪器域/质量域转换；EUV→磁图应用只估计 unsigned magnetic field。**不能写成 Jarolim et al. 已完成 HMI↔AIA 双向转换。**

建议的正式依据链为：

1. Park (2019) 或 Galvez (2019)：HMI→AIA 任务可行；
2. Sayez (2025) 或 Dannehl (2024)：AIA→HMI 任务可行；
3. Jarolim (2025)：太阳图像的联合双生成器与 cycle-consistent 训练可行；
4. 本工作：首次在统一数据划分和统一评估下耦合两个跨层方向。

## 5. 没有纳入“正式核心论文”的最接近工作

### 5.1 最接近的早期双向先例

Barra & Delouille (2019), “Generation of magnetograms using image-to-image translation on EUV images,” [会议海报](https://ml-helio.github.io/2019/posters/barra.pdf)。

这张海报展示了修改后的 MUNIT 框架、AIA 304 Å→HMI Bz 和 HMI Bz→AIA 304 Å 两个方向，是本次检索中最接近“真正直接双向 HMI↔AIA”的先例。但它只是会议海报/communication，未核验到对应的同行评议正式论文，因此不能满足本报告的“正式论文”条件，只适合在 related work 中作为 preliminary precedent 披露。

### 5.2 明确展示两个方向、但不是正式出版物的工作

Shen et al. (2025), “Contrastive Heliophysical Image Pretraining for Solar Dynamics Observatory Records,” [arXiv:2511.22958](https://arxiv.org/abs/2511.22958)。

该稿以 SolarCHIP 表征和 ControlNet 展示 HMI→AIA 与 AIA→HMI，并针对不同目标模态训练专用的扩散先验/分支；因此在任务方向上确实覆盖两边。不过截至本次核验，arXiv API 中的 v2 记录带有管理员撤回说明，公开文本还含 `UNDER REVIEW` 和未完成占位内容，也没有核验到正式期刊或会议版本。它不能作为用户所要求的“正式论文”，但说明“双向都做”的研究空白已有非同行评议的公开稿件先例；正式投稿时的新颖性声明必须限定为“peer-reviewed literature”。

## 6. 科学定义与实验建议

HMI↔AIA 应称为**耦合的双向跨模态转换**，不应称为物理可逆映射：

- HMI→AIA：磁场不能唯一决定温度、密度、加热历史及视线积分后的辐射；
- AIA→HMI：EUV 形态不能唯一确定 signed polarity；
- cycle consistency 只能提供学习约束，不能证明物理可逆或输出真实。

推荐第一阶段采用 `HMI LOS ↔ AIA 304 Å`，再扩展到多波段/向量版本。训练和验证时至少应：

- 两个方向使用完全相同的时间隔离 train/validation/test split，避免太阳自转导致的活动区重复泄漏；
- 分别设置目标域专用的输出头/解码器和损失，不强迫两个方向共享同一输出分布；
- HMI 方向除 MAE/SSIM 外，单独报告 signed-polarity accuracy、磁通、PIL/活动区局部指标，并与全零磁图基线比较；
- AIA 方向报告辐射强度误差、结构相似性和通道间一致性；
- 将 cycle loss 作为非对称辅助项，并对关闭 cycle、关闭 adversarial loss 做消融。

## 7. 可用于论文的严谨表述

> 既有同行评议研究分别验证了磁图到多通道 UV/EUV 图像的生成（Galvez et al., 2019; Park et al., 2019），以及 EUV 图像到 LOS/向量磁图的重建（Dannehl et al., 2024; Sayez et al., 2025）。太阳观测的联合双生成器与循环一致训练也已在仪器域转换中得到验证（Jarolim et al., 2025）。基于这些互补工作，我们将两个单向任务统一为一个耦合的 HMI↔AIA 双向框架。

新颖性可谨慎写为：

> To the best of our knowledge, among the peer-reviewed literature identified, no prior study has jointly optimized and quantitatively evaluated both HMI→AIA and AIA→HMI within a single coupled architecture.

如果这样写，建议紧接一句承认 Barra & Delouille (2019) 的会议海报是最接近的初步先例。

## 8. 检索范围与局限

本次按题名、摘要、任务方向、输入输出、出版载体与 DOI 核验，优先使用出版社、期刊、机构知识库和论文全文。检索词包括 `HMI to AIA`、`magnetogram to EUV`、`AIA/EUV to HMI magnetogram`、`bidirectional solar image translation`、`cycle-consistent solar instrument translation` 等。

“未找到同一篇正式双向论文”是截至检索日期、在可公开核验文献范围内的结论，不是数学意义上的绝对不存在证明。若用于投稿中的“首次”声明，应在投稿前再对 ADS、Web of Science/Scopus 和最新预印本做一次更新检索。

## AI 辅助披露

本报告由 Codex 辅助检索、筛选、主源交叉核验和撰写。论文身份、DOI、任务方向和关键限制尽量回到期刊、出版社、机构记录或论文全文核查；正式研究结论仍应由研究者阅读原文并通过统一数据划分的复现实验确认。
