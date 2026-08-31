# SolarCHIP 日级太阳耀发分类数据

本目录把 SolarCHIP 每个 `date_id` 的 HMI/AIA 样本，与 NOAA/NCEI 的 GOES XRS 耀发事件目录按日历日期对齐。HMI 文件名中的观测时刻是 `00:00:00 TAI`，耀发目录使用 UTC；当前标签以相同的 `YYYY-MM-DD` 对齐，并不是把 TAI 时刻精确换算为 UTC。两者约几十秒的时间尺度差只可能影响极少数午夜边界事件。

## 数据源结论

首选源是 NOAA/NCEI 2026 年发布、每日更新的 **science-quality GOES XRS L2 Composite Flare Report**：

- [官方 CSV 目录](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/)
- [官方 ReadMe](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf)
- [CIRES/STP 下载示例](https://cires-stp.github.io/goesr-spwx-examples/examples/exis/flrpt_example_csv.html)
- [NCEI GOES-R EXIS 产品页](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance)

该产品按耀发一行记录 `start_time`、峰值时刻 `time`、`end_time`、`flare_class`、XRS-B 峰值辐照度、卫星、位置和活动区等字段；NOAA 已做主/备星选择和事件去重。元数据许可为可不受限制地使用和再分发。

截至本次下载（2026-08-20），官方 mission-length 文件名为：

```text
sci_xrsf-l2-flrpt_geo_s19950103_e20260816_v1-0-1.csv
```

下载器无论直接获取 mission-length 文件，还是通过 Jina Reader 获取逐年载荷后合并，最终都使用同一个稳定的本地文件名：

```text
data/noaa_goes_xrs/goes_xrs_flare_report.csv
```

`direct` 模式保留 NOAA mission-length CSV 的原始字节，只统一修改本地文件名；官方文件名和 URL 仍记录在 manifest。Jina 逐年模式把年度文件放进临时目录，逐个校验 22 列结构并合并；正常完成或代码异常退出时临时年度文件会自动删除，只保留合并 catalog、metadata 和 manifest。年度来源 URL、SHA256、字节数、行数及时间边界保存在 manifest 的 `source_yearly_catalogs` 中。

metadata、catalog 和 manifest 会先在同一文件系统的 staging 目录内全部下载、校验和生成，再作为一组提交；普通提交异常会尝试恢复上一组可验证文件，若回滚本身不完整则会在输出目录保留 `.goes_xrs_recovery_*` 供人工恢复。成功提交后，无论 direct 还是 Jina 都会受控清理旧版下载器留下的 `raw_yearly/`、旧格式 merged 文件名，以及 NAS 为这些受管文件生成的 `._*` AppleDouble 旁车；若 `raw_yearly/` 含有非 NOAA 年度目录文件，则为防止误删会直接报错并保留原内容。

`data/noaa_goes_xrs/download_manifest.json` 的 `files` 只列出实际保留的 catalog 和 metadata，并保存算法版本、NOAA 全局覆盖区间、本地实际下载覆盖区间和传输说明；代理合并文件不会被声称为 NOAA 字节级直连快照。转换前会再次核对 catalog 与 metadata 哈希。已经由旧版代码下载的数据，会在下一次成功运行下载器后迁移到上述布局。

用户提供的旧 `flare_data.csv` 只覆盖 2010-05-01 至 2017-06-27，共 4,529 条，并且只保留有经纬度的旧 GOES 年报记录；2011 年整年缺失，A 类为 0，不能用于当前 2024–2025 验证区间，也不能代表“每一次耀发”。另外，science-quality 产品把 GOES-8–15 事件统一重标定到 GOES-R 标度，因此同一事件与旧 operational 列表的等级不同不一定是转换错误，详见官方 ReadMe 的 caveat。

SDO 并非 2015 年开始工作：NASA 记录的发射日期是 2010-02-11，2010-05-01 进入 Phase E。SolarCHIP 的 `date_id=0` 正好对应 2010-05-01。[NASA SDO mission timeline](https://sdo.gsfc.nasa.gov/mission/project.php)

## 生成标签

```bash
# 下载。能直连 NOAA 时可改为 --transport direct。
python3 -m downstream.flare.data.download_goes_flare_report \
  --transport auto \
  --start-year 2010

# 生成 date_id 对齐的逐日标签。
python3 -m downstream.flare.data.prepare_flare_labels
```

默认规则是一个明确的建模选择：对日历日 D 的 HMI 输入，以耀发 UTC `start_time` 落入 `[D 00:00, D+1 00:00)` 作为“D 当天开始发生”，同日多次耀发取最高 GOES 字母等级。这对应“从 D 日开始后 24 小时内是否启动耀发”的目标，而 NOAA 文件本身按峰值 `time` 正式索引。

| label | 含义 |
|---:|---|
| 0 | 当天没有被目录收录的耀发 |
| 1 | A |
| 2 | B |
| 3 | C |
| 4 | M |
| 5 | X |

如需严格跟随 NOAA Flare Report 的正式峰值时间索引，可改用：

```bash
python3 -m downstream.flare.data.prepare_flare_labels --event-time-column time
```

输出 `data/flare_daily_labels.csv` 的字段为：

```text
date,date_id,label,label_name,max_flare_class,max_xrsb_irrad_w_m2,flare_count
```

下载器会把“NOAA 全任务范围”和“本地实际下载年份范围”分开写入 manifest；转换器只在本地范围内输出，显式越界会报错，避免把未下载年份静默填成 0。不过 Flare Report 是事件表，不是逐日观测质量掩码，无法排除范围内部的观测中断或检测失败；因此 0 始终只能表示“该日没有 catalogued event”。当前 start-time 版本覆盖 2010-05-01 至 2026-08-16，共 5,952 天；日标签分布为：

```text
0: 1166, A: 0, B: 1135, C: 2323, M: 1176, X: 152
```

原始事件中有 6 条 A 类，但这些日期都有更高等级事件，因此“日最高等级”标签没有 A 样本。另外，GOES-R 自动 flare-summary/detection routine 只检出部分 B 类且不以 A 类为常规检测目标；所以 label 0 不能解释为物理上绝对没有微小耀发。CSV 始终保留可审计的 0/A/B/C/M/X 原始标签；训练 Dataset 默认通过 `['0AB','C','M','X']` 把无目录事件、A 和 B 合并为一类。[Machol et al. (2026)](https://doi.org/10.1029/2026JA035181)

全目录有 279 个事件的开始日与峰值日跨 UTC 午夜，令 40 个日期的日最高标签在两种策略下不同。用户的 `[5000, 5400)` 区间只有 2024-11-19 受影响：默认 start-day 为 C，peak-day 为 M。标签 sidecar 会记录并校验 `event_time_column`，防止两种语义混用。

## Dataset

新类继承现有 `data.dataset.SolarDataset.multimodal_dataset`：

```yaml
validation:
  target: downstream.flare.data.dataset.FlareDataset
  params:
    modal_list: ['hmi', '0094']
    enhance_type: ['log1p', 'zscore']
    load_imgs: false
    torch_augment_type: [1024, 0.0, 0]
    time_interval: [5000, 5400]
    time_step: 1
    label_path: downstream/flare/data/flare_daily_labels.csv
    expected_event_time_column: start_time
    class_groups: ['0AB', 'C', 'M', 'X']
```

单样本返回：

```python
{
    "hmi": Tensor[1, H, W],
    "0094": Tensor[1, H, W],
    "label": LongTensor[],
}
```

`class_groups` 是一个有序分区：列表位置就是新的 label ID，因此默认映射为 `0/A/B/C/M/X -> 0/0/0/1/2/3`。六个原始符号必须各出现且只能出现一次；空组、缺失、重复或非法字符都会在 Dataset 构造时直接报错。组内字符顺序会规范化，但组的列表顺序具有语义，例如 `['C','0AB','M','X']` 会把 C 定义为新类 0。

默认 PyTorch collate 后，`batch["label"]` 为分组后的 `LongTensor[B]`。原始 CSV label 仍保存在 Dataset 的 `labels_by_date_id`，训练目标保存在 `grouped_labels_by_date_id`。标签按 `self.exist_idx[position]`（全局日期 ID）查询，不按过滤后的紧凑位置 `position` 查询。数据集初始化时还会核对 `flare_daily_labels.summary.json` 中的 CSV 哈希、dataset epoch 和 start/peak day 策略；继承的 `compute_modal_statistics()` 也已覆盖为只统计模态键。

不要启用项目现有的 `custom_collate_fn`：它只遍历 `modal_list`，会静默丢弃 `label`。`load_imgs=True` 也被显式拒绝，因为父类该分支返回堆叠 Tensor 而不是模态字典。

`config_example.yaml` 提供了完整的 `DataModuleFromConfig` train+validation 示例；所有需要分类标签的 split 都必须把 target 换成 `FlareDataset`。当前 `DataModuleFromConfig` 会在判空前读取 `train.params`，所以不能只留下 `validation` 段。

用户给出的 `[5000, 5400)` HMI+0094 区间实际有 386 个联合存在样本，日期 ID 为 5000–5388（2024-01-08 至 2025-01-30），标签全覆盖；分布为：

```text
0: 3, A: 0, B: 2, C: 120, M: 219, X: 42
```

使用默认分组后，Dataset 实际返回的四类分布为：

```text
0AB: 5, C: 120, M: 219, X: 42
```

请从仓库根目录启动训练，因为父类的模态索引路径是相对路径。也不要调用根项目的 `data.utils.transfer_date_to_id()` 来生成本任务 ID：该 helper 当前把日差额外乘了 1440；本转换器直接使用 `(date - 2010-05-01).days`，与 `self.exist_idx` 的日 ID 契约一致。

## SolarPredictor 分类模型

`SolarPredictor.py` 从 SolarCHIP checkpoint 中严格提取 HMI 分支，不在最终模型中注册非 HMI 模态或任何 decoder。checkpoint 可以是 Lightning 的 `{"state_dict": ...}` 或纯 Tensor state dict；HMI encoder、CNN `cls_proj` 和 contrastive projector 都要求键名及 shape 严格匹配，避免错误配置被随机初始化。

两类 backbone 被映射到相同的 256 维主特征：CNN 对 `[B,C,H',W']` latent 使用独立 `AttentionPool2d`，ViT 对投影前的原始 CLS token 使用 `Linear(D,256)`。可选的预训练 contrastive 全局向量保持原来的 32 维，再通过新 adapter 映射到 256 维并由可学习 gate 残差加入：

```text
feature = main_256 + tanh(gate) * adapter(contrastive_32)
logits  = MLP(LayerNorm(feature))
```

当前仓库有真实预训练权重的模型是 CNN。完整配置见 `solar_predictor_cnn.yaml`，其中使用 checkpoint 保存时的精确 `attn_resolutions: []`；不要换成当前通用配置中的 `[32,64]`，否则 encoder 无法严格匹配。训练命令：

```bash
python -m solarchip.main.train \
  -b downstream/flare/solar_predictor_cnn.yaml \
  -n flare_cnn_256
```

首次启动必须能读取 `pretrained_ckpt_path`，用于严格提取 HMI 预训练权重。之后保存的完整 Lightning checkpoint 已包含 HMI encoder、映射层、分类头、optimizer 和 scheduler 状态；即使原始 SolarCHIP checkpoint 被移动，也可以独立恢复。用本项目 `train.py` 恢复时必须同时显式传入本配置：

```bash
python -m solarchip.main.train \
  -r <run_dir>/checkpoints/last.ckpt \
  -b downstream/flare/solar_predictor_cnn.yaml
```

不能省略 `-b`：当前 `train.py` 会在 resume 时把 argparse 的默认 VQGAN 配置追加到已保存配置之后，进而覆盖分类模型配置。也不要把 Ctrl-C 时 `SetupCallback` 写出的 `last_state_dict.ckpt` 当作完整 resume checkpoint；它只有裸 `state_dict`，应使用 `ModelCheckpoint` 生成的 `last.ckpt`。模型参数里的 `max_epochs` 控制 cosine scheduler，trainer 的 `max_epochs` 控制训练终点，两处应保持一致。

配置只读取 `modal_list: ['hmi']`。SolarPredictor 接收与 Dataset 相同的 `class_groups`，并用组数自动确定分类头、confusion matrix、class-weight 长度及指标类别数；默认输出 4 类。训练启动前会逐个核对 train/validation/test Dataset 的规范化分组，连组顺序都必须完全一致，否则在首个 batch 前报错。checkpoint 同时保存并校验分组语义，因此即便两种方案恰好都是四分类，也不会静默错用 logits。旧的六分类 downstream checkpoint 不能完整 resume 到默认四分类，应从 SolarCHIP 预训练 checkpoint 开始新训练。checkpoint 仍以 `val_loss` 选择，且 `save_weights_only: false`、`save_last: true`。

分类目标通过模型超参数选择，默认仍是普通交叉熵：

```yaml
loss_type: cross_entropy  # 或 focal
focal_gamma: 2.0          # 仅 focal 使用，必须 >= 0
class_weights: null       # 可选：每个分组类别一个严格正权重
```

`focal` 使用多类 softmax focal loss：`-(1-p_t)^gamma log(p_t)`。设置 `class_weights` 后，它同时作为交叉熵的类别权重或 focal loss 的类别 alpha 权重；两种 loss 都按 batch 内真实类别权重之和归一化，因此 `focal_gamma: 0` 与相同权重下的 `cross_entropy` 完全退化一致。类别权重整体乘一个常数不会改变 loss。loss 类型、Focal gamma、归约方式和类别权重都会进入完整 Lightning checkpoint 的续训兼容性检查；旧 checkpoint 没有 loss metadata 时只按交叉熵解释，不能静默改成 Focal 续训。

`train_backbone` 控制预训练分支是否参与训练。设为 `false` 时，HMI encoder、可选的 VAE `quant_conv`、CNN `cls_proj` 和 contrastive projector 会永久冻结并从 optimizer 排除；新建的 mapper、adapter/gate 和分类 MLP 仍正常训练。设为 `true` 时保持完整微调，`freeze_encoder_epochs` 可选地让预训练分支先 warmup 冻结若干 epoch；若要从第 0 epoch 训练整个网络，把它设为 `0`。临时冻结期间预训练参数仍保留在 optimizer 中，forward 用 `no_grad` 跳过梯度，后续解冻不会改变参数组。完整 resume 不能切换 `train_backbone`，因为两种模式的 optimizer 参数组不同。若改为多 GPU，必须显式使用 `strategy: ddp_find_unused_parameters_true`。

## 统一测试入口

`downstream/flare/test.py` 使用训练目录中保存的 project YAML 重建模型与数据预处理，并严格加载完整 checkpoint。它同时支持 SolarPredictor、DeepSWM、P-CNN 和 Yi2023 DQN；统一指标都从各模型最终解码的类别预测计算，不直接混用三种对比模型含义不同的 logits/Q 值。

传入运行目录时默认使用 `checkpoints/last.ckpt`：

```bash
python -m downstream.flare.test \
  -r logs/compare_flare/deepswm/2026-08-30T22-48-11
```

也可以直接指定某个权重；需要测试验证指标最佳的 epoch 时应使用这种形式，因为 `last.ckpt` 表示最后一轮而不是最佳一轮：

```bash
python -m downstream.flare.test \
  -r logs/compare_flare/deepswm/2026-08-30T22-48-11/checkpoints/epoch=000029.ckpt
```

`--metrics` 支持 `overall_acc`，以及 `pod/csi/far/hss/tss/acc`。只写二分类指标简称会同时计算 C+ 和 M+；也可以用 `c_plus_tss`、`m_plus_far` 等完整名称只选择一个阈值：

```bash
python -m downstream.flare.test \
  -r <run-or-checkpoint> \
  --metrics overall_acc pod csi far hss tss acc
```

- `overall_acc`：把最终预测折叠成 `0AB / C / MX` 三分类后的 accuracy。
- C+：把 `C/M/X` 作为 positive，分别计算 POD、CSI、FAR、HSS、TSS 和 binary ACC。
- M+：把 `M/X` 作为 positive，计算相同指标。
- 二分类 confusion matrix 固定为 `[[TN, FP], [FN, TP]]`；所有零分母按仓库既有约定返回 `0.0`。

结果会打印到终端，并默认保存为 `logs/test_results/<model>_<run>_<checkpoint>_<split>_metrics.json`。文件名包含模型目录和训练时间戳，多个模型的 `last.ckpt` 不会互相覆盖。可用 `--output` 改路径，或用 `--no_save` 只打印。默认复用保存配置的 `validation` split；`--split`、`--time_interval`、`--time_step`、`--batch_size` 和 `--num_workers` 均可覆盖。P-CNN 在完整 checkpoint 恢复时会跳过冗余的 ImageNet 权重下载，但正式训练的默认初始化方式没有改变。

SolarDataset 完成模态存在性筛选后，FlareDataset 会继续过滤标签表中不存在的日期，而不是令测试中断。终端和结果 JSON 会同时记录请求的半开 `time_interval` 所对应的开始日期、包含的最后日期和不包含的结束边界，以及标签过滤后数据集实际保留的首尾日期、样本数和缺失标签丢弃数。

## 验证

```bash
python3 -m unittest \
  downstream.flare.tests.test_download_goes_flare_report \
  downstream.flare.tests.test_prepare_flare_labels
python3 -m py_compile \
  downstream/flare/data/download_goes_flare_report.py \
  downstream/flare/data/prepare_flare_labels.py \
  downstream/flare/data/dataset.py

# 模型测试需要先激活项目的 solargpt/PyTorch 环境。
python -m unittest \
  downstream.flare.tests.test_flare_dataset \
  downstream.flare.tests.test_solar_predictor \
  downstream.flare.tests.test_flare_test
```

已完成真实 NOAA 数据转换、CSV/日期 ID/manifest 全量哈希验证、下载器与转换器单元测试和静态编译。在本机 `solargpt` Conda 环境中还实际实例化了父类和 `FlareDataset`，读取真实 `exist_idx` 与 label sidecar，并用 PyTorch 2.6 验证默认 collate 得到 `LongTensor[B]`、统计函数不再把 label 当模态。由于仓库的 `global_settings.DATA_ROOT` 仍指向训练服务器上的 Linux `/mnt/...` 路径，本机测试用合成 Tensor 代替父类图像读取；真实 PT 图像 I/O 仍需在训练服务器的既有数据环境中跑一个 batch 验收。
