# AIA 0094 → HMI I2IwFiLM 白图诊断

诊断对象：`logs/aia_hmi_i2iwfilm_0094/2026-08-13T17-31-44`

## 结论

这次训练不是保存或配色出错，而是模型学成了接近零磁场的常数解。白色对应归一化 HMI 的 `0` 附近；它在当前 `RdBu_r`、`vmin=-3`、`vmax=3` 的固定显示范围内必然呈白色。

主因是：单像素 L1 面对以弱场、近零值为主且正负近似对称的 HMI 分布时，零/中位数预测是很强的局部最优；同时单通道 AIA 0094 强度对 HMI 磁极性本身缺少可辨识信息。第 100 epoch 切换到 source-only guidance 后，模型很快从带有少量结构的输出进一步收缩到零场解。

## 第二轮训练复查：`2026-08-14T11-40-36`

第二轮使用 5 倍强场加权 SmoothL1、400 epoch、确定性验证和 CCC 选模，日志完整读到 epoch 293。它避免了长期全白，但没有恢复可信磁结构：

| Epoch | 阶段 | prediction std | val PCC | val CCC | 强场极性 | 物理 MAE [G] |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 99 | Stage 1 末 | 0.5204 | -0.00113 | -0.00293 | 0.4840 | 8.8925 |
| 100 | Stage 2 首轮 | 0.00823 | 0.00034 | 3.70e-7 | 0.5062 | 7.6018 |
| 199 | Stage 2 | 0.3738 | 0.00173 | -0.00024 | 0.5002 | 8.1256 |
| 269 | 当前最佳 CCC | 0.4596 | 0.01901 | 0.01106 | 0.5105 | 8.4118 |
| 293 | 最新 | 0.4821 | 0.01006 | 0.00591 | 0.5090 | 8.5150 |

这说明“偏淡”只是表象。模型后期重新获得了约 0.48 的归一化幅度，但 PCC/CCC 仍几乎为零，强场极性仍约等于随机猜测，且 MAE 比 epoch 100 的近零解更差。继续从 293 跑到 400 不会修正训练目标。

本轮暴露了两个更具体的原因：

1. 训练集 `|B| >= 100 G` 像素只占约 0.565%。5 倍逐像素权重后，其在总权重中的占比仍只有 `5f / (1 + 4f) ≈ 2.76%`，安静区仍绝对主导目标。
2. 作者公开 Stage-2 配置将图像重建损失权重全部设为 0，仅优化 guidance-vector predictor；当前 SolarCHIP 版本却在 Stage 2 继续用 source guidance 更新 generator。第 100 epoch 的标准差从 0.5204 瞬间跌到 0.00823，随后 generator 重新长出高对比活动区块，但这些区块与 HMI 几乎不相关。

因此下一轮已改为：Stage 1 延长到 200 epoch，并把强场/其余像素分别取均值后按 1:1 合并；Stage 2 固定 paired teacher 和 generator，只训练 source guidance predictor。新增 `target_std`、`amplitude_ratio`、paired-teacher PCC/CCC/幅度比，并用只在 Stage 2 生效的 `val/checkpoint_ccc` 选模。

## 直接证据

### 1. 白图代表真实的近零输出

`SolarImageLogger` 对所有 HMI 图统一使用 `[-3, 3]`，并未对生成图单独自动缩放。相同尺度下 target 有清晰正负磁结构，而 generated 接近白色，因此不是显示范围不一致。

### 2. 塌缩发生在两阶段切换附近

- Epoch 99：source-only 日志仍能看到弱磁结构。
- Epoch 100：训练从 paired guidance 切换到 source-only guidance。
- Epoch 119：生成图已经基本变白。
- Epoch 139–799：持续接近白图。

TensorBoard 文件可读取部分的关键数值：

| Epoch | val/L1 | val/PCC | val/CCC | delta-SSIM vs zero | train recon L1 | train guidance L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.786535 | 0.000468 | 1.20e-6 | -2.85e-5 | 0.713360 | 0 |
| 99 | 0.781595 | -0.000278 | -4.20e-6 | -2.90e-5 | 0.701804 | 0 |
| 100 | 0.765258 | -0.000584 | -2.97e-7 | -5.33e-8 | 0.705055 | 0.246166 |
| 119 | 0.762969 | -0.000823 | -3.27e-7 | -8.38e-8 | 0.703268 | 0.036899 |
| 191 | 0.762462 | -0.001801 | -1.07e-6 | -5.13e-8 | 0.702987 | 0.036163 |

Stage 2 开始后，`val/L1` 看似明显变好，但 PCC、CCC 和相对零图的 delta-SSIM 同时趋近于零。这说明 L1 的下降不是磁结构恢复，而是输出更接近零基线。

`strong_field_polarity` 长期约为 0.50，也与模型未学到磁极性一致。

### 3. paired teacher 自身也没有学到有效重建

Stage 1 的 `val/paired_teacher_l1` 仅从 0.766025 降到 0.761570；即使 teacher 路径观察了真实 HMI，它也几乎停留在零场基线附近。因此 Stage 2 让 source predictor 拟合 teacher guidance 时，teacher 本身没有携带足够的磁结构信息。

## 原因排序

### A. 主要原因：L1 对近零像素的偏好

当前重建损失是对整幅 1024×1024 图等权平均的 L1。日面外背景、弱场和安静区像素数量远多于强场结构；HMI 又有正负两极。对这种分布，预测接近 0 可以获得不差的平均 L1，却完全不恢复活动区、极性和细结构。

### B. 主要原因：0094 单通道对磁极性存在信息缺口

AIA 0094 主要提供高温辐射强度，能提示活动区位置，但单幅无符号强度不能可靠决定 LOS 磁场的正负极性。当前结果在活动区位置偶尔留下很淡的轮廓，但 PCC/CCC 与极性准确率都接近无信息基线，符合这个可辨识性限制。

此外，Sayez 论文的直接任务是 AIA 0304 → HMI；0094 是本项目的算法迁移实验，不是论文验证过的通道。

### C. 主要原因：两阶段 guidance 退化

Stage 1 中 source predictor 完全不训练；日志和 checkpoint 指标却始终使用 source-only 路径。Stage 2 才首次训练 source predictor，并同时让 generator 用 source guidance 做重建。由于 paired teacher 已经接近退化，Stage 2 的联合目标很快选择了更容易的零输出。

当前 guidance 只有向量 L1 监督，没有机制保证 paired guidance 必须编码空间磁结构；全局池化后的 256 维向量也容易被 generator 忽略。

### D. 次要配置问题：模型 200 epoch，Trainer 跑 800 epoch

模型调度配置为 `stage1_epochs=100, max_epochs=200`，Trainer 却设置 `max_epochs=800`。学习率在前 100 epoch 做一次 cosine，在 100–199 epoch 重启并衰减；200 epoch 以后被固定在 `minimum_learning_rate=1e-7`。因此后 600 epoch 基本只是在已塌缩解附近微调，不能挽救结构。

这不是白图的起点，因为塌缩已在 epoch 100–119 发生，但它浪费了训练并让 epoch 707 的“最佳 L1 checkpoint”具有误导性。

### E. 次要配置问题：验证集仍使用随机增强

验证配置为 `[1024, 0.5, 360]`，每轮都会随机翻转、旋转。配对仍保持对齐，所以它不直接制造白图；但每个 epoch 的验证样本视图不同，会给 checkpoint 排序增加噪声。应改为 `[1024, 0.0, 0]`。

### F. 复现边界较大

当前实现将公开方法适配成 1024 分辨率卷积 Guided U-Net，并使用 signed-log1p + z-score；论文直接设置是 0304、256×256 中央裁剪、HMI/AIA 截断后缩放到 `[-1,1]`。因此当前 0094 训练更适合作为算法迁移实验，不能用论文效果直接预期。

## 建议的最小验证顺序

1. **先确认零基线**：在同一验证集计算全零预测的 L1、MAE、PCC、CCC、delta-SSIM 和强场极性。当前模型预计与它几乎相同。
2. **把 Trainer 改回 200 epoch**，验证增强改为 `[1024, 0.0, 0]`；不要继续用 800 epoch 判断是否会“慢慢学出来”。
3. **同时记录两条路径**：Stage 1 分别保存 paired-teacher output 和 source-only output；另记录生成图 `mean/std/p1/p99/abs>1` 比例、paired/source guidance 的均值、标准差和余弦相似度。
4. **做 plain source-only U-Net 对照**：去掉 guidance 两阶段，只用相同 U-Net 直接回归。若它同样为零，主因是任务/损失；若它明显更好，主因是 guidance 训练流程。
5. **处理目标不平衡**：对强场与活动区像素加权或分桶采样；在全图 L1 之外加入强场 masked L1、极性辅助损失、CCC/梯度结构损失。选模不能只看全图 L1，应优先看 PCC、CCC、delta-SSIM 和强场指标。
6. **先复现直接任务**：用 0304、论文更接近的裁剪和归一化完成闭环，再迁移到 0094。若必须从 0094 恢复极性，考虑多 AIA 通道、时间序列或其他物理先验；单通道单帧可能本来就不足。

## 日志完整性限制

当前目录中的部分大文件被截断：

- TensorBoard event 文件恰为 4 MiB，只能读到 epoch 192 左右；
- target PNG 恰为 3 MiB，部分图像文件尾不完整；
- 两个 checkpoint 均为 4.25 MiB，ZIP central directory 缺失，当前副本无法加载。

因此本报告对 epoch 0–191 的数值曲线是直接读取结果；对 epoch 199–799 主要依据仍可完整打开的 generated PNG 序列。若要检查 epoch 707/799 的真实权重和完整后期曲线，需要重新复制未截断的 event/checkpoint 文件。
