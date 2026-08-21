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
python3 downstream/flare/download_goes_flare_report.py \
  --transport auto \
  --start-year 2010

# 生成 date_id 对齐的逐日标签。
python3 downstream/flare/prepare_flare_labels.py
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
python3 downstream/flare/prepare_flare_labels.py --event-time-column time
```

输出 `data/flare_daily_labels.csv` 的字段为：

```text
date,date_id,label,label_name,max_flare_class,max_xrsb_irrad_w_m2,flare_count
```

下载器会把“NOAA 全任务范围”和“本地实际下载年份范围”分开写入 manifest；转换器只在本地范围内输出，显式越界会报错，避免把未下载年份静默填成 0。不过 Flare Report 是事件表，不是逐日观测质量掩码，无法排除范围内部的观测中断或检测失败；因此 0 始终只能表示“该日没有 catalogued event”。当前 start-time 版本覆盖 2010-05-01 至 2026-08-16，共 5,952 天；日标签分布为：

```text
0: 1166, A: 0, B: 1135, C: 2323, M: 1176, X: 152
```

原始事件中有 6 条 A 类，但这些日期都有更高等级事件，因此“日最高等级”标签没有 A 样本。另外，GOES-R 自动 flare-summary/detection routine 只检出部分 B 类且不以 A 类为常规检测目标；所以 label 0 不能解释为物理上绝对没有微小耀发。代码仍完整保留 0/A/B/C/M/X 映射，但直接训练六分类时 A 类为空，应先决定保留空输出、把 A 并入 0/B，或改用另一种可靠覆盖 A 类的标注定义。[Machol et al. (2026)](https://doi.org/10.1029/2026JA035181)

全目录有 279 个事件的开始日与峰值日跨 UTC 午夜，令 40 个日期的日最高标签在两种策略下不同。用户的 `[5000, 5400)` 区间只有 2024-11-19 受影响：默认 start-day 为 C，peak-day 为 M。标签 sidecar 会记录并校验 `event_time_column`，防止两种语义混用。

## Dataset

新类继承现有 `data.dataset.SolarDataset.multimodal_dataset`：

```yaml
validation:
  target: downstream.flare.dataset.FlareDataset
  params:
    modal_list: ['hmi', '0094']
    enhance_type: ['log1p', 'zscore']
    load_imgs: false
    torch_augment_type: [1024, 0.0, 0]
    time_interval: [5000, 5400]
    time_step: 1
    label_path: downstream/flare/data/flare_daily_labels.csv
    expected_event_time_column: start_time
```

单样本返回：

```python
{
    "hmi": Tensor[1, H, W],
    "0094": Tensor[1, H, W],
    "label": LongTensor[],
}
```

默认 PyTorch collate 后，`batch["label"]` 为 `LongTensor[B]`。标签按 `self.exist_idx[position]`（全局日期 ID）查询，不按过滤后的紧凑位置 `position` 查询。数据集初始化时还会核对 `flare_daily_labels.summary.json` 中的 CSV 哈希、dataset epoch 和 start/peak day 策略；继承的 `compute_modal_statistics()` 也已覆盖为只统计模态键。

不要启用项目现有的 `custom_collate_fn`：它只遍历 `modal_list`，会静默丢弃 `label`。`load_imgs=True` 也被显式拒绝，因为父类该分支返回堆叠 Tensor 而不是模态字典。

`config_example.yaml` 提供了完整的 `DataModuleFromConfig` train+validation 示例；所有需要分类标签的 split 都必须把 target 换成 `FlareDataset`。当前 `DataModuleFromConfig` 会在判空前读取 `train.params`，所以不能只留下 `validation` 段。

用户给出的 `[5000, 5400)` HMI+0094 区间实际有 386 个联合存在样本，日期 ID 为 5000–5388（2024-01-08 至 2025-01-30），标签全覆盖；分布为：

```text
0: 3, A: 0, B: 2, C: 120, M: 219, X: 42
```

请从仓库根目录启动训练，因为父类的模态索引路径是相对路径。也不要调用根项目的 `data.utils.transfer_date_to_id()` 来生成本任务 ID：该 helper 当前把日差额外乘了 1440；本转换器直接使用 `(date - 2010-05-01).days`，与 `self.exist_idx` 的日 ID 契约一致。

## 验证

```bash
python3 -m unittest \
  downstream.flare.tests.test_download_goes_flare_report \
  downstream.flare.tests.test_prepare_flare_labels
python3 -m py_compile \
  downstream/flare/download_goes_flare_report.py \
  downstream/flare/prepare_flare_labels.py \
  downstream/flare/dataset.py
```

已完成真实 NOAA 数据转换、CSV/日期 ID/manifest 全量哈希验证、下载器与转换器单元测试和静态编译。在本机 `solargpt` Conda 环境中还实际实例化了父类和 `FlareDataset`，读取真实 `exist_idx` 与 label sidecar，并用 PyTorch 2.6 验证默认 collate 得到 `LongTensor[B]`、统计函数不再把 label 当模态。由于仓库的 `global_settings.DATA_ROOT` 仍指向训练服务器上的 Linux `/mnt/...` 路径，本机测试用合成 Tensor 代替父类图像读取；真实 PT 图像 I/O 仍需在训练服务器的既有数据环境中跑一个 batch 验收。
