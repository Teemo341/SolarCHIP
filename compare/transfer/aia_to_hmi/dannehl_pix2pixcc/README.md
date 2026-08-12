# Dannehl Pix2PixCC：AIA → HMI LOS

这是 **Dannehl, Delouille & Barra (2024)** 的 Pix2PixCC 核心算法在
SolarCHIP 现有 dataloader 与 PyTorch Lightning 训练入口中的对比基线：

> *An Experimental Study on EUV-To-Magnetogram Image Translation Using
> Conditional Generative Adversarial Networks*, Earth and Space Science, 11,
> e2023EA002974.

一手来源：

- [出版社全文与 DOI](https://doi.org/10.1029/2023EA002974)
- [作者/机构论文记录](https://publi2-as.oma.be/record/6459?ln=en)
- [作者代码归档（Zenodo 10.5281/zenodo.10691500）](https://doi.org/10.5281/zenodo.10691500)
- [归档所链接的作者 GitHub](https://github.com/vbarra/pix2pixCC2)

本目录是独立实现，不依赖作者仓库。核验时，Zenodo 归档指向的代码快照
与 GitHub 提交 `7bdb494b35448a48b40590055c9e755fec467c74` 一致；该归档内没有
可直接加载的模型权重。因此这里复现训练逻辑，而不是加载原作者权重。

## 复现了什么

默认配置保留作者发布代码的主要网络与目标函数：

| 组件 | 本实现 |
| --- | --- |
| 生成器 | `Conv7 → 4 × Conv5/2 → 9 × ResBlock → 4 × ConvTranspose3/2 → Conv7` |
| 生成器激活/归一化 | Mish；InstanceNorm；replication padding |
| 生成器通道 | 首层 64，逐次下采样翻倍；输出单通道 |
| 判别器 | 单尺度 70×70 PatchGAN，首层 64，五个卷积块 |
| 条件判别 | `[AIA condition, HMI target/generated]` 按通道拼接 |
| GAN 损失 | LSGAN；判别器为 `0.5 × (real + fake)` |
| 特征匹配 | 判别器前四个中间特征的逐层 L1 之和 |
| 相关损失 | 原分辨率及三次 2× 平均池化，共四尺度 CCC，尺度间取平均 |
| 总生成器损失 | `2 × LSGAN + 10 × FM + 5 × CCC-loss` |
| 优化 | G/D 各自 Adam，lr `2e-4`，betas `(0.5, 0.999)`；无 scheduler |
| 输出层 | identity；不对 z-score 结果作 `tanh` 截断 |

训练采用 Lightning 手动优化，并用 `toggle_optimizer` 分开 G/D 更新。D 更新
时不保留生成器的 1024×1024 计算图；G 更新时重新前向一次，以计算换峰值显存。
所有训练/验证日志均设置 `sync_dist=True`，配置使用
`ddp_find_unused_parameters_true`，适配交替优化下的 DDP 参数使用方式。

这里没有把像素 L1 或 SSIM 加进生成器目标。L1 仅作为稳定、非对抗的
`val/loss`，用于选择 checkpoint；作者发布代码虽然提供可选 SSIM 权重，默认
为 0，论文主目标也不是 SSIM-loss 训练。

## Paper/code：论文正文、表格与作者代码的差异

“严格照正文”和“照作者归档”在若干处不能同时满足。本基线作如下明确选择：

| 差异 | 本实现选择与理由 |
| --- | --- |
| 正文结构叙述给出典型 1 个 residual block；发布代码默认 9 个 | 使用发布代码默认的 9 个 |
| 正文的解码器描述与发布代码激活不完全一致 | 使用每个上采样块的 InstanceNorm + Mish；最终 Conv7 后无激活 |
| 正文广义描述涉及 BatchNorm；发布代码默认 InstanceNorm | 使用发布代码的 InstanceNorm |
| 论文的特征匹配写作 `T=4` 个中间层；代码循环还把最终 logit 当作第 5 项 | 按论文语义只匹配前四层，不匹配 logit |
| 论文 DOE 表的参考设置是 `lambda_CC=1`；作者发布训练选项默认是 `2/10/5` | 使用作者发布选项的 `lambda_LSGAN/FM/CC=2/10/5`；`5` 也在论文试验范围内 |
| 论文统称 correlation coefficient；发布选项 `ccc=True` | 使用 Lin CCC，不退化为仅 PCC |
| 发布代码在整个 batch/channel/image 上一次性估计 CCC | 为 batch=1/多卡一致性，按样本估计后取 batch 均值；四尺度定义不变 |
| 发布初始化辅助函数只处理普通 Conv2d | 同一 normal(0, 0.02) 规则也用于 ConvTranspose2d，避免上采样层遗漏初始化 |

因此，它是“论文核心算法 + 作者代码结构”的工程复现，而不是逐行或数值完全一致
的复刻。作者代码中的通用 `ch_balance` 在本项目 1 输入/1 输出时比例恰为 1，故
条件与目标各拼接一次即可，无需额外复制通道。

## 数据与任务证据边界

十份配置均训练“一个 SolarCHIP AIA key → 一个 HMI channel”，但证据强度不同：

| 配置 | 口径 |
| --- | --- |
| `aia_0304_to_hmi.yaml` | 论文直接评估过的单通道 `[304] → magnetogram` 任务 |
| `aia_0094/0171/0211/0335/1700_to_hmi.yaml` | 对论文算法的单通道跨波段外推；原文没有这些单通道设置 |
| `aia_0131/0193/1600_to_hmi.yaml` | 原文只在某些多通道组合中用过这些通道；这里拆成单通道仍是外推 |
| `aia_4500_to_hmi.yaml` | SolarCHIP 项目扩展；4500 Å 是连续谱通道，不是 EUV |

论文考察的组合包括 `[304]`、`[193,304]`、`[171,193,304]`、
`[193,304,1600]` 与 `[131,1600]`。原文结果说明增加输入通道并不单调改善结果，
不能把这十个单通道实验解释为论文已有结论。

更重要的目标差异是：

- 论文同时讨论 LOS/Bz 与矢量磁场；作者数据加载代码通常堆叠 Bx/By/Bz 三个
  分量。
- 本项目 dataloader 只暴露单通道 `hmi.M_720s` 风格 LOS 磁图，所以这里输出
  一个 LOS channel。
- 矢量场的 Bz 只有在特定几何条件下才可近似 LOS；离开日面中心不能把
  `vector Bz → project LOS` 当作等价标签变换。本实现没有伪造这种转换。

论文也警告：全盘 SSIM/相关性很高仍可能在活动区出现局部结构或极性错误。因此
生成结果只能作为模型估计，不能替代 HMI 实测磁图。

## SolarCHIP 适配

这些差异是为了让比较遵守项目现有数据与训练口径，而不是论文事实：

- `multimodal_dataset` 要求 `modal_list[0] == 'hmi'`，所以即使方向是 AIA→HMI，
  YAML 仍写 `modal_list: ['hmi', '<AIA>']`；模型通过 `source_modal`/`target_modal`
  决定实际方向。
- 论文从原始 1024 数据下采样到 512 进行主要实验；本比较按项目要求直接 resize
  到 1024。
- 论文使用自己的截断、强度缩放、时间采样与划分。本项目沿用
  `signed-log1p → per-modal zscore`、索引区间 `[0,5000)`/`[5000,5400)` 以及成对
  几何增强。输出因此必须是 identity，而不能使用面向 `[-1,1]` 数据的 tanh。
- 论文/作者设置曾使用更大 batch；项目配置固定 batch=1、200 epochs、bf16。
  `generator_channels=64` 的 1024 模型参数量和显存需求都很高，这一点属于比较
  口径，不应误写成论文默认的计算成本。

## 验证指标

`module.py` 同时记录标准化空间和还原空间指标：

- `val/loss`、`val/l1`：标准化空间 L1；checkpoint 监控项。
- `val/mse`、`val/pcc`、`val/ccc`：标准化空间全盘指标，PCC/CCC 先逐图计算再
  对 batch 求均值。
- `val/physical_mae`：先精确撤销 HMI z-score，再撤销 signed-log1p 后的 MAE；
  单位跟项目存储的原始 HMI 数值一致（标准 HMI 磁图通常按 G 解释）。
- `val/mean_field_bias`：还原空间的有符号平均偏差。
- `val/strong_field_mae` 与 `val/strong_field_polarity_accuracy`：仅在目标
  `|HMI| >= 100` 的像素计算；阈值可用 `strong_field_threshold` 修改。
- `val/strong_field_fraction`：强场掩码覆盖率，用于识别“当前 batch 没有强场”
  时极性指标为 0 的情形。

逆变换在 `metric_max_log_value=20` 处只对异常预测作数值保护；训练输出本身不被
裁剪。测试阶段记录同名 `test/*` 指标。

SolarImageLogger 的三个键固定为：

```text
visualization/<AIA>/condition
visualization/hmi/target
visualization/hmi/generated
```

## 训练

从仓库根目录运行，例如直接证据最强的 304 Å：

```bash
python -m solarchip.main.train \
  -b configs/compare/aia_to_hmi/dannehl_pix2pixcc/aia_0304_to_hmi.yaml
```

可选配置：`0094`、`0131`、`0171`、`0193`、`0211`、`0304`、`0335`、
`1600`、`1700`、`4500`。每份 YAML 都是完整、可独立启动的配置，并保存
`last.ckpt` 与按 `val/loss` 排序的 top-3 **完整** checkpoint（含优化器状态），
不是 weights-only 文件。

当前准备环境没有 PyTorch，故只进行 Python AST、YAML 结构和文本契约检查；尚未
验证真实 tensor forward/backward、bf16、GPU 显存或多卡进程。首次真实运行建议
先用同一 YAML 临时覆盖很短的训练/验证区间做 smoke test，再启动完整 200 epochs。
