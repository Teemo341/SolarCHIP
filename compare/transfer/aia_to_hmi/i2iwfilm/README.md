# Sayez et al. I2IwFiLM：AIA → HMI 非对抗基线

本目录实现 Sayez et al. (2025), *Mitigating hallucination with
non-adversarial strategies for image-to-image translation in solar physics*,
A&A 702, A83 的 SolarCHIP 对比版本。论文全文见
[A&A](https://www.aanda.org/articles/aa/full_html/2025/10/aa55324-25/aa55324-25.html)，
[DOI](https://doi.org/10.1051/0004-6361/202555324)，作者代码见
[sayez/I2IwFiLM](https://github.com/sayez/I2IwFiLM)。

## 复现边界

- 论文直接实验只有 **AIA 0304 → HMI LOS/Bz**。因此
  `aia_0304_to_hmi.yaml` 是论文任务的直接适配。
- 0094、0131、0171、0193、0211、0335、1600、1700 是把论文核心算法迁移到
  SolarCHIP 其他 UV/EUV 单通道的项目对比，不应写成论文报告过的实验。
- 4500 是 SolarCHIP 的额外项目扩展；AIA 4500 是连续谱通道，不属于上述 EUV
  直接任务。
- 论文使用 SDOMLv2、256×256 中央裁剪、4 小时采样，并在其管线中把 AIA 304
  截断到 `[0, 1000] DN/s`、HMI 截断到 `[-1500, 1500] G` 后缩放到
  `[-1, 1]`。这里按用户要求复用 SolarCHIP 数据加载器，在 1024×1024 上使用
  项目现有的 signed-`log1p` + z-score。因数据、分割、分辨率和归一化不同，论文
  数值不能直接作为本实现的验收阈值。

## 保留的核心机制

实现类为 `SayezI2IwFiLM`，无判别器、无 GAN 损失、无扩散采样：

1. `PairGuidanceEncoder` 从对齐的 `(AIA, HMI)` 图像对提取 256 维指导向量。
2. `GuidedUNet` 是适配 1024 分辨率的卷积 U-Net（`base_channels=32`）；每个
   学习特征阶段都使用按通道的 additive FiLM，即只预测 `beta` 并执行
   `feature + beta`，不使用乘性 `gamma`。
3. `SourceGuidancePredictor` 只看 AIA，并以卷积编码器 + MLP 预测相同的 256 维
   向量；它是验证和部署时唯一允许使用的指导路径。
4. 输出层为 identity，不使用 `tanh`，因为 SolarCHIP 的 HMI target 是无界
   z-score。

论文描述的两个训练阶段被折叠进一次 `Trainer.fit`；默认配置使用 200 epoch，
0094 零场塌缩修复配置使用 400 epoch：

- 默认 Epoch 0–99（0094 配置为 0–199，Stage 1）：paired guidance + HMI 重建目标，
  同时让 source guidance predictor 拟合停止梯度的 paired guidance。这样部署路径不必
  等到 Stage 2 才开始学习。
- 之后的 Stage 2：固定 paired guidance teacher 与 Stage-1 generator，只训练
  source guidance predictor 以 L1 拟合教师向量。这与作者公开的 Stage-2 配置中
  图像重建损失权重全部为 0 的训练边界一致。

默认重建目标仍是全图 L1，以保持旧配置兼容。针对 0094→HMI 中观测到的零场
塌缩，`aia_hmi_i2iwfilm_0094.yaml` 改用强/弱场分组 SmoothL1：先分别计算
`|B| >= 100 G` 与其余像素的均值损失，再由 `strong_field_loss_fraction` 指定强场
区域占总损失的比例。0094 当前取 10%；它既避免强场贡献随约 0.56% 的像素频率
消失，也修正上一轮 50% 配置导致的约 179 倍单像素权重和大块饱和输出。该设置是
SolarCHIP 运行诊断后的项目适配，不是 Sayez 论文报告的超参数。

所有参数从开始到结束始终处于同一个 AdamW 参数组；代码不在阶段切换时动态
修改 `requires_grad`，以免破坏 DDP reducer。Stage 1 的 source predictor 只接收
停止梯度后的教师向量，Stage 2 不反传 paired encoder 和 generator，所以配置必须使用
`ddp_find_unused_parameters_true`。

## 训练参数的证据等级

论文没有给出可确定复现的 optimizer、learning rate、batch size、epoch 数或完整
AIA→HMI 配置。本项目采用可运行默认值：AdamW、`lr=1e-4`、`weight_decay=1e-4`、
batch 1、100+100 epochs，并让 cosine schedule 在 Stage 2 开始时重启。这样仅从
第 100 epoch 起获得梯度的 source predictor 仍能使用完整学习率周期，同时不重建
optimizer 或破坏 DDP 参数注册。这些是 **SolarCHIP 实验口径**，不是论文原文声称
的超参数。

截至核验的作者仓库提交 `fe4ec34da71e4cd745083b4d8c7df2157ad4a474`，公开
代码同样不能消除歧义：仓库主体实现名为 I2IFormer，和论文对 U-Net 主干的文字
描述不完全一致；通用 launcher 与 YAML 中还可见 600/800 epochs 的冲突，并且
没有完整的 AIA 0304→HMI 训练 YAML、论文 checkpoint 或可核验权重。因此本目录
选择论文的算法级机制，并用卷积 Guided U-Net 接入 SolarCHIP，而不是机械复制
仓库中无法闭环的脚本。

## 验证指标

`validation_step`、`test_step` 和 `log_images` 的主生成结果始终使用 source-only 路径：

`log_images` 的 `generated` 始终是 source-only 部署路径；另记录
`generated_paired_teacher`，用于判断 paired teacher 是否学到结构以及 Stage 2
是否完成 guidance 蒸馏，不用于 checkpoint 排序或最终结果。

- `val/loss`：配置选择的归一化空间重建目标；
- `val/reconstruction_l1`、`val/strong_field_l1`：全图与强场区域的原始 L1；
- `val/prediction_std`、`val/target_std`、`val/amplitude_ratio`、
  `val/prediction_abs_mean`：零场/幅度塌缩诊断量；
- `val/spatial_gradient_ratio`、`val/prediction_strong_field_fraction` 及对应的
  `paired_teacher_*`：分别检查过度平滑/大块结构和强场面积膨胀；
- `val/paired_teacher_l1`、`val/paired_teacher_pcc`、`val/paired_teacher_ccc`、
  `val/paired_teacher_amplitude_ratio`：paired-guidance 路径诊断量，用于诊断
  teacher 与 U-Net；它看过真实 HMI，因此绝不用于 checkpoint 排序或最终比较；
- `rmse_gauss`、`physical_mae_gauss`：反解 signed-log1p/z-score 后的物理量；
- PCC、CCC；
- `val/checkpoint_ccc`：始终等于 deployable `val/ccc`；Stage 1 已同步训练 source
  predictor，因此有效的早期最优模型也允许被保存；
- `strong_field_polarity`：`|HMI_target| >= 100 G` 像素上的符号准确率；
- `delta_ssim`：`SSIM(pred, target) - SSIM(0, target)`。预测和目标先反归一化，
  再截断至 ±1500 G 并缩放至 `[-1, 1]`。实现使用自包含的 11×11 局部 SSIM，
  不依赖 `torchmetrics`。论文没有公开足以保证逐位一致的 SSIM 窗函数细节，因此
  该量遵循论文定义，但不承诺和论文软件栈 bitwise 一致。

Stage 1 的 `val/loss` 坚持走 source-only predictor；应结合
`val/paired_teacher_l1` 判断 teacher/generator 是否在学习。最终比较使用保存的
source-only 最优 checkpoint，不把真实 HMI 泄漏到部署路径或最终选模指标。

HMI 的反变换是：

```text
signed_log = normalized * 1.4462468177923982 - 0.0033644122878536808
HMI[G] = sign(signed_log) * expm1(abs(signed_log))
```

## 配置与运行

每份 YAML 都是完整、独立的训练配置，并保持数据加载器要求的
`modal_list: ['hmi', '<AIA>']`；模型内部仍明确令 AIA 为 source、HMI 为 target。

| 配置 | 证据口径 |
| --- | --- |
| `aia_0304_to_hmi.yaml` | 论文直接任务适配 |
| `aia_0094_to_hmi.yaml` | 算法迁移 |
| `aia_0131_to_hmi.yaml` | 算法迁移 |
| `aia_0171_to_hmi.yaml` | 算法迁移 |
| `aia_0193_to_hmi.yaml` | 算法迁移 |
| `aia_0211_to_hmi.yaml` | 算法迁移 |
| `aia_0335_to_hmi.yaml` | 算法迁移 |
| `aia_1600_to_hmi.yaml` | 算法迁移 |
| `aia_1700_to_hmi.yaml` | 算法迁移 |
| `aia_4500_to_hmi.yaml` | SolarCHIP 项目扩展 |

从仓库根目录运行，例如：

```bash
python -m solarchip.main.train \
  -b configs/compare/aia_to_hmi/i2iwfilm/aia_0304_to_hmi.yaml
```

模型生成的是从 EUV/UV/连续谱条件推断出的合成磁图，而不是真实 HMI 测量。
论文也明确显示模型难以完整恢复活动区细尺度结构；这些结果不得替代磁场观测，
也不应在未做独立科学验证时用于物理结论。
