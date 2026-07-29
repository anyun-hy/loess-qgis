# 半自动矢量标注插件完整实施与验收方案（当前版）

> 文件名保留 `plugin_plan_v3.md`。本文是插件唯一的设计、实施和验收依据。`docs/design_inference_env_ui.md`、`.omo/plans/` 和外部粘贴方案只作为设计输入，不得覆盖本文决策。

## 1. 最终目标

插件面向一个完整的半自动标注闭环：

1. 对用户选择的同一影像和范围只切片一次。
2. 执行一个或多个已注册的语义分割模型。
3. 每个实际执行的模型都保存独立的 mask、confidence、mosaic 和矢量面结果，并在地图中形成独立图层。
4. 选择 `approved` 的 `fusion_profile.json` 时，按配置要求执行模型融合，保存独立的融合结果层。
5. 所有规模统一采用空间分区、Halo、Core 和有界工作包；正式目标支持 1 到 500,000 个 `512 x 512` Tile，不把全部 Tile、概率、状态或矢量线网一次性放入内存、单个 JSON 或单个全图几何操作。
6. 每个模型流和 Fusion 流先在分区概率 mosaic 上生成 raw coverage，再提取相邻 Polygon 的公共分界线；每条分界只执行一次 Cubic B-Spline 并由两侧共同复用，禁止构建旧共享边图、执行 RDP/直线/圆弧自适应拟合或整幅百万级亚像元线网。
7. 分区 Core、相邻 Seam corridor 和四分区 Junction patch 使用互斥空间所有权组装，保证跨分区没有缝隙、重叠和重复拟合；跨单元同类连通部件通过磁盘连接图分配统一 `object_id`，保留可流式写入的 `part_id`，禁止最后再执行一次整幅 dissolve。
8. 用户选择一个边界拟合和跨分区验收均通过、`ready` 且 `approved` 的 Fusion 结果作为不可变基准，插件按 14 个类别拆分出 14 个可独立显示、编辑和确认的类别工作层。
9. 用户在类别工作层中点击一个已有 Fusion 地物，SAM3 只对该位置生成临时候选边界；用户比较后决定保留 Fusion、采用 SAM3，或以任一几何为基础继续人工修边。
10. 14 个类别工作层全部确认后，插件组装完整矢量层并生成拓扑问题层；用户修正并确认后写入长期保存的 `accepted_labels.gpkg`。

SAM3 不是语义推理的必选步骤，也不参与类别判断。它不是独立结果来源，不生成单独的最终类别层，只为当前类别工作层中的当前地物提供可选几何。融合结果是多个模型分数的算法融合，不是多个矢量层的几何叠加。500,000 Tile 是本地单机正式设计上限；1,000,000 Tile 需要分布式调度和远程对象存储，不属于本方案验收范围。

### 1.1 QGIS4 迁移目标与运行边界

本轮正式桌面目标固定为 QGIS 4.2，不继续承担 QGIS3 或 Qt5 兼容成本。插件元数据必须为 `version=0.3.0`、`qgisMinimumVersion=4.2`，所有 Qt 枚举使用 PyQt6/Qt6 的 scoped enum 写法。

运行时严格分为两个进程边界：

| 边界 | 正式运行时 | 职责 |
|---|---|---|
| QGIS 插件进程 | QGIS 4.2 DMG 自带 Python 3.12.11、PyQt 6.11.0、Qt 6.11.1 | 地图、UI、范围捕获、图层管理和任务调度 |
| 推理子进程 | `/opt/anaconda3/envs/qgis`，Python 3.12.13、PyTorch 2.7.1 | TorchScript、融合、mosaic、polygonize 和 SAM3 CPU |

两套 Python 只能通过 Shell 命令、JSON/JSONL、GeoTIFF 和 GeoPackage 通信。禁止向 QGIS DMG Python 注入 Conda `site-packages`，也禁止向 `qgis` Conda 环境安装 Conda 版 QGIS。

正式安装目录默认为：

```text
/Users/example/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins
```

安装脚本必须支持 `--profile` 和 `--plugin-dir`，并默认使用 `/opt/anaconda3/envs/qgis/bin/python` 做仓库 Python 语法检查。推理环境由 `inference_scripts/environment-qgis.yml` 描述，`config.sh` 默认 `CONDA_ENV=qgis`。

## 2. 原方案输入输出核对

| 要求 | v3 原方案 | 当前代码 | 当前最终要求 |
|---|---|---|---|
| 影像与范围 | 当前视图或手绘矩形 | 已实现 | 保留；必须显示已捕获范围和扩展后的 tile 范围 |
| Tile | 512 x 512，允许重叠 | 已实现边缘扩展 | 保留；同一运行只提取一次，所有模型共享 |
| 单模型输出 | 单个 Swin 模型一套结果 | 已实现 | 升级为每个执行模型各自一套完整结果 |
| 多模型输出 | 无 | 未实现 | 每个模型必须保存 mask、confidence、mosaic、矢量面和运行记录 |
| 模型融合 | 无 | 未实现 | 读取 `approved` profile，在逐像素分数层融合并单独输出 |
| 跨 tile 矢量化 | 先 mosaic 后一次性矢量化 | 已实现旧整幅方式 | 每流按 Partition probability mosaic 后矢量化，再由 Seam/Junction 无缝组装；禁止 Tile 内和整幅全局操作 |
| SAM3 | 单语义层完成后自动处理所有对象 | 已实现旧整批运行 | 改为点击已有 Fusion 地物后生成单个临时候选；禁止自动整类处理和自动写回 |
| 14 类编辑 | 语义层/SAM 层整体编辑 | 旧分类工作区与新交互不一致 | 从一个 approved Fusion 拆出 14 个类别工作层；所有 SAM3 与人工修改都写回对应工作层 |
| 最终合成 | 直接接受 semantic 或 SAM | 部分实现 | 14 个类别工作层确认后组装 final_composite，再检查拓扑并入库 |
| 已确认区域 | 完全覆盖 tile 跳过，部分重叠 difference | 已实现 | 保留；作用于所有结果流的候选面 |
| UI | 单个狭长主面板 | 已实现但拥挤 | 主面板负责控制，三个专用弹窗负责复杂配置、监控和分类修整 |
| 结果路径 | 固定 `output/tmp` 与 `output/results` | 会覆盖旧运行 | 每次运行使用唯一目录，永久结果不覆盖，临时分数按策略清理 |
| 配置 | 旧单模型 `model.semantic_weight` | 当前仍使用 | Schema v2 模型注册表；不兼容旧格式 |

结论：v3 的“Tile 只用于推理、概率拼接先于矢量化、面级置信度、全局 ID、跳过已确认区域、SAM3 不判类、GeoPackage 主输出”继续有效；整幅一次性矢量化、单模型、自动/批量 SAM3、把 SAM3 当独立来源、固定五层和固定结果目录均已被当前方案取代。

## 3. 不可破坏的原则

1. 模型只有 14 个有效输出类别，不存在 background 模型通道。
2. 类别索引固定为 `0..13`，类别码顺序固定为 `[12,13,21,31,32,33,43,51,52,53,54,61,62,71]`。
3. `background_index=-1` 只代表没有推理结果的内部 nodata，不是模型类别。
4. 类别 mask 不做数值平均；融合发生在 14 通道分数层。
5. Tile 只用于模型输入；概率拼接、raw polygonize 和边界拟合都以带 Halo 的空间分区为单位，禁止在单个 `512 x 512` Tile 内独立矢量化后拼接。
6. 禁止分别平滑相邻 Polygon 的完整边界。两个区域之间的公共分界 `Polyline` 只提取、拟合和重采样一次；两侧 Polygon 必须复用同一组拟合坐标，方向可以相反。
7. 当前正式拟合只使用 Cubic B-Spline 消除像元台阶；每条拟合边只做左右面有效、正面积和总面积守恒的通过/回退判断，不叠加 RDP、直线/圆弧分类、单面面积约束、Gap/Overlap 检测或 Topology Repair。
8. 公共分界线在 raster 像素坐标空间拟合，再用完整 affine transform 转回地图 CRS；禁止把像素参数直接当成 EPSG:4490 的度。
9. 分类修整只允许从一个边界规则化验收通过、`ready` 且 `approved` 的 Fusion 结果初始化；正式 Fusion 保持不可变，模型结果只用于对照。
10. 14 个类别工作层是分类修整阶段唯一的可见编辑载体；SAM3 不创建独立类别结果层，也不改变类别、object_id 或 part_id。
11. SAM3 只生成当前点击地物的临时候选，任何候选都必须经过人工明确选择后才能替换工作层几何，失败和取消不得修改数据。
12. 人工可以保留 Fusion、采用 SAM3、以 Fusion 为基础编辑或以 SAM3 为基础编辑；最终保存的边界始终位于当前类别工作层。类别判断错误时，用户通过显式“更正类别”把选中对象移动到目标类别工作层，不需要重新勾画正确几何，也不得由 SAM3 自动改类。
13. 插件运行时只加载 TorchScript 部署模型，不复制训练工程中的模型结构。
14. 被拒绝的融合 profile 可显示用于追溯，但禁止启动正式融合或初始化正式分类工作区。
15. 每次运行以 SQLite/WAL 保存 Tile、分区、结果流、Seam、Artifact、重试和事件状态；JSON 只保存小型不可变配置快照与摘要，禁止把五十万 Tile 状态写成一个巨大 JSON。
16. 插件只在用户明确请求范围绘制或 SAM3 点选时临时接管地图工具，完成、取消、关闭或异常后必须恢复原地图工具。

## 4. 总体架构

```text
主停靠面板
  ├── 数据与范围
  ├── 当前推理方案摘要
  ├── 启动 / 停止 / 打开监控
  └── 打开分类修整工作区

推理配置弹窗
  ├── Schema v2 配置检查
  ├── 模型注册表与真实设备测试
  ├── fusion profile 选择与详情
  └── SAM3 可用性检查

异步运行器
  ├── 共享 Tile 提取
  ├── SQLite/WAL 任务库 + 有界工作包 + 背压
  ├── 单 MPS/CUDA 语义 worker，按工作包顺序执行所需模型
  ├── 分区 probability mosaic + raw polygonize
  ├── 相邻 Polygon 公共分界线提取 + 单次 Cubic B-Spline + 均匀重采样
  ├── Seam corridor + Junction patch 跨分区组装
  └── VRT/GeoPackage 结果、分页监控、停止/恢复/失败重试

分类修整弹窗
  ├── 选择 approved Fusion 基准并初始化 14 个类别工作层
  ├── 点击已有地物并生成单个 SAM3 临时候选
  ├── 保留 / 替换 / 人工修边，结果写回同一类别工作层
  ├── 逐类检查并确认工作层
  └── final_composite + topology_issues + accepted_labels
```

## 5. 配置输入：Schema v2

`inference_scripts/config.yaml` 只接受以下新结构，不解析旧 `model.semantic_weight`：

```yaml
schema_version: 2

runtime:
  device: auto
  model_artifacts_dir: ../weights
  keep_score_cache: false
  tile_batch_size: 1

scaling:
  partition_tile_rows: 8
  partition_tile_cols: 8
  partition_halo_px: auto
  seam_band_px: 64
  score_cache_budget_gb: 16
  min_free_disk_gb: 50
  max_cpu_partition_workers: 2
  max_open_frontier_units: 64
  max_partition_segments: 250000
  max_partition_features: 100000
  max_partition_runtime_sec: 900
  max_job_retries: 2
  tile_page_size: 500

semantic_models:
  - model_id: upernet_swin_b
    display_name: UPerNet Swin-B
    version: l2-20260630
    artifact: upernet_swin_b.torchscript.pt
    sha256: "<64位小写十六进制>"
    enabled: true

fusion_profiles:
  - profile_id: l2_fusion_v1
    file: ../weights/fusion_profile.json
    enabled: true

sam3:
  enabled: true
  checkpoint: ../weights/sam3.pt
  version: sam3-v1
  device: auto
  buffer_px: 32

boundary_fitting:
  enabled: true
  mode: divider_cubic_bspline_v1
  smoothing_factor: 1.0
  output_spacing_px: 0.5
  diagnostic_level: changed_and_failed

classes:
  background_index: -1
  index_to_code:
    0: 12
    1: 13
    2: 21
    3: 31
    4: 32
    5: 33
    6: 43
    7: 51
    8: 52
    9: 53
    10: 54
    11: 61
    12: 62
    13: 71
```

配置检查必须验证：

- `schema_version == 2`，旧格式直接报错并指出字段位置。
- `model_id` 唯一，artifact 存在，SHA256 匹配。
- 每个启用模型可以在有效设备执行一次 `[1,3,512,512]` 真实 dummy inference，并输出 `[1,14,512,512]` float32 logits。
- 模型注册表与 profile 中相同 `model_id` 的 artifact 文件名和 SHA256 一致。
- 类别顺序严格一致，不允许自动修正或猜测。
- explicit `cuda`、`mps` 或 `cpu` 不可用时阻止启动；`auto` 按 CUDA、MPS、CPU 选择并在 UI 显示最终设备。
- SAM3 设备独立检查；不稳定或不支持的 MPS 不强行使用，正式运行使用 CUDA 或 CPU。
- `scaling` 参数必须满足：配置的分区 Core 至少 `2 x 2` Tile、Halo 不小于 `max(tile_overlap,seam_band_px)`、缓存预算和剩余磁盘为正、CPU worker 不超过可用核心与内存预算、开放 frontier 和重试数为正。
- 启动前按 Tile 数、结果流数、像元覆盖和工作包预算估算峰值临时空间与永久空间；估算空间超过可用磁盘时阻止启动，禁止边运行边碰运气。
- `boundary_fitting.enabled` 必须为 true，当前只接受 `mode=divider_cubic_bspline_v1`。`smoothing_factor` 和 `output_spacing_px` 必须为正数；不接受旧 Shared Edge/RDP 参数控制正式行为。
- `diagnostic_level` 正式默认 `changed_and_failed`，保存实际发生拟合的公共分界线和偏移报告；`all` 只能在存储预检通过后使用。
- 推理环境必须提供 Shapely 的 Polygon 邻居查询能力和 SciPy `splprep/splev`；不要求独立拓扑修复 API。

## 6. 融合配置输入契约

插件只消费 `fusion_profile.json` 和它引用的 TorchScript 资产，不读取训练配置、checkpoint 或训练日志。

必须读取并校验：

- `schema_version`
- `profile_id`
- `status` 与 `approval.passed`
- `strategy`
- `class_order`
- `input`
- `models[].model_id/artifact/sha256/temperature`
- `weights`
- `fusion_head`（仅 `linear_1x1`）
- `metrics.baseline/fusion`
- `integrity`

支持的策略：

```text
equal_probability_average
calibrated_global_weighted
calibrated_class_weighted
linear_1x1
```

执行规则：

```text
模型输出 raw logits
  → log_softmax(logits, class_dimension)
  → 除以各模型 temperature
  → 按 profile 的 [14,M] 权重融合
  → linear_1x1 策略再执行 fusion_head
  → argmax 得到 mask
  → softmax 后最大概率得到 fusion confidence
```

不能直接对 raw logits 做分类权重融合，也不能用矢量面投票替代 profile 算法。

选择 profile 后，profile 要求的模型自动勾选并锁定。用户可以额外执行其他已注册模型，但额外模型不进入该 profile 的融合。

## 7. 单次运行输入

启动前插件生成小型不可变 `run_spec.json`，并创建 `run_state.sqlite`。`run_spec.json` 只保存配置、范围、模型与哈希；Tile、分区和事件明细进入 SQLite。

| 输入 | 来源 | 说明 |
|---|---|---|
| raster | 主面板 | 当前有效栅格层、CRS、完整 affine 和 nodata |
| requested_extent | 主面板 | 用户确认的当前视图或手绘矩形 |
| processing_extent | Tile 管理 | 扩展到完整 Tile 后的真实处理范围 |
| tile_width/height | 固定契约 | 正式模型固定 `512 x 512` |
| overlap | 主面板 | 默认 192 px；必须 `0 < overlap < 512` |
| partition_grid | scaling | 默认每个 Core 为 `8 x 8` Tile |
| partition_halo | scaling | `auto=max(overlap,seam_band_px)`，默认 192 px |
| work_package_budget | scaling | 以临时 score cache 预算限制同时在途 Tile 数 |
| boundary_fitting | 固定运行契约 | 公共分界 Polyline 提取一次、Cubic B-Spline 拟合一次、均匀重采样并同时重建两侧 Polygon；有效、正面积且总面积守恒才提交，否则共同回退原边 |
| output_root | 主面板 | 所有 run 的工作区，启动前验证可用空间 |
| accepted_gpkg | 主面板 | 长期保存的已确认标签库 |
| selected_model_ids | 配置弹窗 | 所有实际执行的模型 |
| fusion_profile_id | 配置弹窗 | 可为空；存在时必须 approved |
| effective_device | 环境检查 | 实际语义推理设备 |
| skip_accepted | 主面板 | 是否跳过已确认区域 |

每个执行模型都必须生成独立模型结果。运行计划必须同时记录 Tile 数、分区数、Seam 数、Junction 数、结果流数、预计临时空间、预计永久空间和最小可用空间。

## 8. 大规模流式推理、分区矢量化与边界拟合

所有规模只使用本节这一条链路。小范围只有一个分区；大范围增加分区，不切换为另一套算法。

### 8.1 Tile、Partition 与空间所有权

1. 对 requested extent 生成共享 `512 x 512` Tile 网格，边缘扩展到完整模型输入；原始影像和用户传入 Tile 永远只读。运行时物化副本固定写入 `output/cache/<run_id>/tile_cache/`，同一个缓存 Tile 被本 Run 的所有模型和依赖 Work Package 共享。
2. Tile 按空间顺序归入默认 `8 x 8` Tile 的 Partition。Partition Core 不是 Tile 外框 union，而是最终 mosaic 全局像素网格上的非重叠整数窗口；内部边界取相邻 Partition 末端/起始 Tile 有效中心线之间的确定性整数分割线。边缘 Partition 可以不足 8 x 8，Core 仍与同一全局像素网格对齐。
3. 每个 Partition 读取 Core 加 Halo。Halo 默认 `max(tile_overlap, seam_band_px)`，用于概率拼接、边界上下文和跨分区处理，不属于该 Partition 的最终所有权范围。
4. 矢量空间拆成互不重叠的三类所有权单元：`Core interior`、两分区之间的 `Seam corridor`、四分区交点的 `Junction patch`。每个有效像素必须且只能属于一个最终单元。
5. 完全落在 accepted 覆盖区的 Tile 跳过；部分重叠 Tile 照常推理，最终 formal coverage 再 difference。

`seam_band_px=64` 定义为分区内部边界每侧 64 px，因此完整 Seam corridor 为 128 px 宽；外部 processing extent 不创建 Seam。横纵 corridor 的交集归 Junction patch，Seam 再扣除 Junction，Core interior 最后扣除全部 Seam/Junction。所有窗口使用半开区间 `[x0,x1) x [y0,y1)`；所有权优先级固定为 `Junction -> Seam -> Core interior`，同级按 `unit_id` 排序只用于调度，不影响几何。规划器必须验证所有单元窗口 union 精确等于 processing extent 且 pairwise intersection 面积为 0；几何 coverage 的有效参考范围另扣除 raster nodata 和 accepted mask，gap/overlap 只能相对该有效范围计算。

禁止在单个模型 Tile 中 polygonize 或拟合后直接拼接。Tile 是推理单位，Partition/Seam/Junction 才是 raster-to-vector 处理单位。

### 8.2 有界工作包与执行并发

调度器把相邻 Partition 组成 Work Package。Package 必须包含完整 Partition 及其 Halo 依赖，大小由 `score_cache_budget_gb`、可用磁盘、结果流数和单 Tile 实测缓存字节数动态计算，至少 1 个 Partition，不得超过预算。工作包计划在启动前写入 SQLite；运行中只能因低磁盘或失败细分而缩小，不能静默扩大。

工作包按空间局部顺序执行；完成一个包后优先处理其未完成邻包，以尽快关闭 Seam/Junction 依赖。处于“本包完成、邻包未完成”的边界称为 open frontier；数量达到 `max_open_frontier_units` 时，调度器必须暂停远处新包并补齐邻包。这样需要长期保留的分区概率只与开放边界有关，不随累计完成 Tile 数增长。

Package 的 Tile 上限按实测值计算：

```text
package_tile_limit = floor(
  score_cache_budget_bytes
  / (current_model_probability_bytes
     + fusion_accumulator_bytes
     + mask_confidence_workspace_bytes
     + safety_margin_bytes)
)
```

最终取该上限、磁盘可用量和完整 Partition 边界共同允许的最小值。预检至少抽样一个真实 Tile 计算压缩前后字节，不得用固定经验数字代替。

```text
提取 Package 共享 Tile
  → 模型 A 加载一次并处理 Package Tile
  → 立即生成 A 的 Partition 结果并增量写 Fusion accumulator
  → 释放 A 的无依赖 Tile probability
  → 模型 B/C 依次执行同一过程
  → 完成 Fusion accumulator 并生成 Fusion Partition 结果
  → 各结果流 raw polygonize + 公共分界线单次 B-Spline 拟合
  → 邻接分区齐备后执行 Seam/Junction
  → 提交正式资产并释放无依赖临时缓存
```

- MPS/CUDA 正式只允许 1 个语义 worker，模型顺序执行，避免多模型争用统一内存；`tile_batch_size` 只控制单模型内部小批量。
- CPU 允许默认 2 个 Partition 后处理 worker，与下一批语义推理流水重叠；必须使用有界队列和背压，CPU 队列满时暂停继续产生 score。
- 每个几何 worker 是独立子进程。停止或超时只终止当前 Partition，不杀死 QGIS；父调度器根据 SQLite 状态决定重试、细分或失败。
- 不能为 500,000 Tile 创建 500,000 个常驻 Qt 行、Python Future 或内存对象。
- Fusion 使用磁盘分块增量 accumulator。`equal_probability_average`、`calibrated_global_weighted` 和 `calibrated_class_weighted` 按模型依次累加；`linear_1x1` 保存 profile 明确要求的最小中间通道。禁止为了 Fusion 同时保留整个 run 的全部模型概率。
- 单模型 probability 只有在其 Partition raster、raw/formal、Fusion accumulator、Seam/Junction 引用均提交后才删除。崩溃恢复按 Artifact 引用计数复用已完成缓存，不从头重跑整个工作包。
- Work Package 只有在自身模型流、Fusion 和正式 Partition 资产全部提交后，才可释放本包引用。仍被未完成邻包 Halo、重试或恢复引用的 Tile 缓存继续保留；最后一个依赖释放时同时删除 Tile 与其 metadata。失败、停止或未提交状态保留缓存用于安全恢复。
- Tile 清理只能删除经路径归属校验、直接位于本 Run `tile_cache` 下的文件；来源影像、用户 Tile、其他 Run cache 和 `runs/<run_id>/` 永久结果均不得成为清理目标。

### 8.3 分区概率拼接和永久栅格

每个结果流、每个 Partition 独立完成：

```text
Tile 14-class probabilities
  → Core+Halo 范围二维 cosine 权重累加
  → 逐像素归一化
  → argmax / max probability
  → Partition probability 临时栅格
  → Core mask/confidence 永久分块 GeoTIFF
```

概率拼接必须发生在 14 通道分数空间，禁止平均类别 mask。临时 probability 使用 `uint16 / 65535` 或等价有 scale 的分块格式，量化误差 `<1.6e-5`；单个 worker 只映射当前 Partition，禁止全图 14 通道常驻内存。

永久 raster 不强制合成一个巨型 TIFF。每个流保存 Core 分块 GeoTIFF，并生成 `mask_mosaic.vrt`、`confidence_mosaic.vrt` 作为统一地图入口；需要导出时再显式生成 BigTIFF/COG。VRT、所有分块及哈希共同构成正式资产。

### 8.4 模型流与 Fusion

每个实际模型都对相同 Partition 生成独立 mask、confidence、raw、formal 和报告。Fusion 在同一 Work Package 内按 profile 逐模型增量累加，不要求多个模型全量 probability 同时存在；最后生成 fused probabilities，再走与模型流完全相同的分区拼接和边界拟合。

当一个 Partition 的模型流、Fusion、相关 Seam/Junction 和哈希全部通过后，SQLite Artifact 引用计数归零的 Tile score 与 probability 分区才可删除。任何仍被 Fusion、重试、Seam 或报告引用的缓存不得提前清理。

### 8.5 公共分界 Polyline 单次曲线拟合（当前 E5S 唯一范围）

当前问题只按下面的直接流程实现，不建设独立 Shared Edge 或 GIS 拓扑系统：

```text
相邻 Polygon A / B
  -> A.boundary 与 B.boundary 的公共分界 Polyline
  -> 按弦长参数化
  -> 公共线只执行一次 Cubic B-Spline 拟合
  -> 0.5 px 均匀重采样
  -> 同一组拟合坐标写回 A 与 B（其中一侧可反向）
  -> 重建 Polygon A / B
  -> 检查两侧均有效、面积均大于 0、两侧总面积基本守恒
  -> 通过则提交；失败则两侧共同保留原公共分界
```

正式实现入口为：

```text
inference_scripts/polyline_smoother.py
inference_scripts/common_boundary_smoother.py
inference_scripts/boundary_fitting/unit_runtime.py
```

`polyline_smoother.py` 只负责一条开线或闭合线的 Cubic B-Spline 和重采样。`common_boundary_smoother.py` 只负责寻找相邻 Polygon 的公共线、调用一次拟合，并在原始环坐标序列中替换该分界段。它不对两个 Polygon 的完整边界分别平滑，也不通过 `difference`、`polygonize` 或 union 重新猜测覆盖关系。

公共分界线有两种合法形式：

- 开放分界线：固定首尾点，拟合后的点序列分别以正向和反向写入两侧环。
- 闭合分界线：用于岛状面与包围它的孔环；闭合线只拟合一次，内方面外环和外方面孔环复用同一闭合坐标序列。

同一个 Polygon 环上存在多条公共分界时，按当前已提交的 Polygon 逐条处理：每条公共分界拟合后同时构造左右两个候选面；在像素坐标和完整 affine 转换后的最终输出 CRS 中，只有左右面均有效、均为正面积，并且拟合前后左右总面积差不超过对应数值容差时，才同时提交两侧更新。像素面积容差为 `max(1e-6 px2, old_total_area * 1e-9)`，输出 CRS 容差按 affine 面积尺度换算。任一检查失败时，该公共分界两侧均保留拟合前坐标。该判断不分析凹角、窄颈、曲率或位移，也不调用几何修复。

默认配置固定为：

```text
mode = divider_cubic_bspline_v1
smoothing_factor = 1.0
output_spacing_px = 0.5
spline_degree = 3
max_deviation = null
```

默认不再设置 `1.5 px` 位移上限，也不按位移自动降档或回退；最大/平均偏移只写入报告供人工判断。短线或点数不足的分界保持原样。外部 processing extent 不属于两个 Polygon 的公共分界，不执行本次拟合。

公共分界至少 4 个点即可拟合。为避免 4–7 点稀疏折线退化成不稳定的精确三次插值，拟合前先沿原折线按 `1 px` 均匀加密样本，再执行 Cubic B-Spline；这一步不改变原折线，只改变拟合输入的采样密度。长度不足 `3 px` 或少于 4 点的微小分界保持原样。

当前 E5S 明确不实现：RDP、直线/圆弧自适应分类、独立 Shared Edge 图、Topology Repair、Gap/Overlap/Coverage 检测、单面面积约束、自交修复和 Polygon valid 修复。只执行上述拟合结果的简单通过/回退判断；`Seam/Junction` 只属于大规模分区调度，不参与本节曲线算法。

E5S 自动验收必须覆盖：两面共享同一开放拟合线、一个环依次处理多条分界、闭合岛状分界复用同一拟合环、无效/零面积/总面积不守恒时两侧共同回退、最终 formal 无无效或非正面积 Polygon、台阶方向显著下降，以及真实 Core 分区可完成处理。真实验收由用户在 QGIS 中对 raw/formal 图层叠加目视判断；任一代表区域仍明显呈台阶或形变不可接受，就调整 B-Spline 参数或停止该路线，不用拓扑系统补救视觉效果。

### 8.6 Difference

所有模型流和 Fusion 流在进入审核前都执行同一规则：

- accepted 完全覆盖的 tile 已在推理前跳过。
- 部分重叠 tile 的候选面在 polygonize 后与 accepted 做 difference。
- 一个对象被切成多个残片时保留 object_id，重新分配 part_id。
- 跨 accepted 边界的地物允许被截断，依赖人工修正。

模型流可以继续作为地图对照层，但“分类修整与组装”只能从一个 `ready` 且 `approved` 的 Fusion 流初始化。difference 后的 Fusion 矢量是类别工作层的不可变基准快照。

## 9. 单次运行输出目录

用户选择输出工作区，插件创建唯一 run 目录：

```text
output/
├── accepted_labels.gpkg
├── cache/
│   └── <run_id>/
│       └── tile_cache/
│           ├── tile_<row>_<col>.tif
│           └── tile_<row>_<col>_meta.json
└── runs/
    └── <run_id>/
        ├── run_spec.json
        ├── run_manifest.json
        ├── run_state.sqlite
        ├── config_snapshot.json
        ├── class_mapping_snapshot.json
        ├── models/
        │   └── <model_id>/
        │       ├── mask_mosaic.vrt
        │       ├── confidence_mosaic.vrt
        │       ├── raster_parts/
        │       │   └── <partition_id>_{mask|confidence}.tif
        │       ├── semantic_polygons_raw.gpkg
        │       ├── semantic_polygons.gpkg
        │       ├── boundary_fitting_report.json
        │       └── fitted_edges.gpkg
        ├── fusion/
        │   └── <profile_id>/
        │       ├── mask_mosaic.vrt
        │       ├── confidence_mosaic.vrt
        │       ├── raster_parts/
        │       ├── semantic_polygons_raw.gpkg
        │       ├── semantic_polygons.gpkg
        │       ├── boundary_fitting_report.json
        │       └── fitted_edges.gpkg
        ├── classes/
        │   ├── class_12.gpkg
        │   ├── ...
        │   ├── class_71.gpkg
        │   ├── workspace.json
        │   └── edit_history.jsonl
        ├── refinement/
        │   └── sam3/
        │       ├── sessions.jsonl
        │       └── candidates.gpkg
        ├── final/
        │   ├── final_composite.gpkg
        │   └── topology_issues.gpkg
        ├── logs/
        │   ├── pipeline.jsonl
        │   ├── run_report.json
        │   ├── seam_band_report.json
        │   └── failures.json
        └── tmp/
            ├── work_packages/
            ├── probability_parts/
            ├── unit_outputs/
            └── failed_jobs/
```

`run_id` 使用时间戳加随机短标识，跨进程、跨日期不撞车。禁止新运行覆盖旧 run 目录或同名 cache。`runs/<run_id>/` 只保存运行状态、正式结果和可恢复的非 Tile 工作资产；可删除的输入 Tile 物化副本只允许位于 `cache/<run_id>/tile_cache/`。Run 规范必须冻结 `cache_root` 和 `tile_cache_dir` 的绝对路径；运行时拒绝路径漂移和符号链接。`run_state.sqlite` 是运行明细真值源；`run_manifest.json` 是从数据库生成的小型结果摘要，不复制 Tile 明细。数据库启用 WAL、外键和事务，任何 Artifact 只有在临时文件 fsync、原子重命名、SHA256 写入成功后才可标记 ready。

`semantic_polygons_raw.gpkg`、`semantic_polygons.gpkg` 和诊断 GPKG 都按空间单元使用 GDAL/Fiona 批量事务流式追加并创建 RTree 索引；禁止逐要素提交事务，写入过程不得回读全部要素。`object_id` 必须按有界要素批次从 SQLite 查询，禁止每个要素单独建立数据库连接。每个空间单元完成时同时把报告标量摘要写入 SQLite，并把需要保留的诊断边写入单元诊断 GPKG；最终组装只从数据库聚合摘要并直接批量追加单元诊断 GPKG，不再保留重新解析全部 JSON 报告的旧组装路径。缺少摘要或诊断分片的旧 Run 必须明确拒绝组装，不能静默回退。最终文件只有在所有事务提交、`PRAGMA integrity_check`、图层字段/CRS/feature count 和 SHA256 通过后才进入 Artifact `ready`。

## 10. 结果流与图层命名

结果流分为：

- `model:<model_id>`
- `fusion:<profile_id>`

QGIS 显示名：

```text
<run_id> | Model | <model_id> | Mask
<run_id> | Model | <model_id> | Confidence
<run_id> | Model | <model_id> | Polygons
<run_id> | Fusion | <profile_id> | Mask
<run_id> | Fusion | <profile_id> | Confidence
<run_id> | Fusion | <profile_id> | Polygons
<run_id> | Class | <class_code> <class_name> | Working
<run_id> | Final | Composite
<run_id> | Final | Topology Issues
```

所有图层放入 `<run_id> 标注结果` 图层组；组内再分 `Models`、`Fusion`、`Classes`、`Final`。`Models` 和 `Fusion` 默认加载 VRT raster 与边界拟合通过的 `semantic_polygons.gpkg`，raw 和 fitted_edges 仅作为永久诊断资产，需要核对时再加载。`Classes` 下的 14 个工作层是分类修整阶段唯一可编辑图层。SAM3 候选只用临时预览覆盖物显示，`candidates.gpkg` 仅用于会话追溯，禁止作为独立结果层加载到图层树。图层内部 GPKG layer name 使用稳定短名，显示名可读且不依赖字符串包含关系判断来源。

### 10.1 唯一样式来源

14 类名称与颜色只允许由 `qgis_plugins/labeling_tool/core/style_manager.py` 的 `StyleManager.CLASS_COLORS` 提供。模型矢量、Fusion、14 个类别工作层和 final_composite 都按同一 `class_code` 调色板渲染；表格色块也从该常量读取。仓库不再读取或维护 `semantic_14class.qml`、`sam_refined.qml` 等旧 QML，防止两份颜色定义漂移。SAM3 临时候选的青色虚线只表示交互状态，不代表第 15 个类别。

## 11. 语义矢量字段

每模型流与融合流统一字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| run_id | TEXT | 本次运行 ID |
| result_stream_id | TEXT | `model:<id>` 或 `fusion:<id>` |
| result_kind | TEXT | `model` 或 `fusion` |
| model_id | TEXT | 模型流填写；融合流为空 |
| fusion_profile_id | TEXT | 融合流填写；模型流为空 |
| object_id | TEXT | `{run_id}_{stream_short_id}_{seq}` |
| part_id | TEXT | 默认 `000`，difference 后递增 |
| class_code | INT | 14 类 TDLYDM |
| class_name | TEXT | 对应 TDLYMC |
| confidence_mean | REAL | 面内 confidence 均值 |
| confidence_std | REAL | 面内 confidence 标准差 |
| model_version | TEXT | 模型版本或融合 profile 版本 |
| source | TEXT | `semantic_model` 或 `semantic_fusion` |
| fit_changed | INT | 最终对象是否包含已采用拟合边 |
| fit_methods | TEXT | `cubic_bspline_shared_divider` 或 `unchanged` |
| fit_version | TEXT | 当前固定 `divider_cubic_bspline_v1` |
| fit_status | TEXT | 当前面是否为 `changed` 或 `unchanged` |
| origin_unit_ids | TEXT | 来源 Core/Seam/Junction 单元 ID 列表摘要 |
| vertex_count_before | INT | raw 对应边界坐标点数统计 |
| vertex_count_after | INT | formal 坐标点数统计 |
| max_shift_px | REAL | 所有组成边的最大位移 |
| mean_shift_px | REAL | 所有组成边的平均位移 |
| area_change_ratio | REAL | 兼容旧字段；当前模式不计算、不作为门槛 |
| created_at | TEXT | ISO 时间 |

### 11.1 类别工作层字段

14 个类别工作层从同一个 Fusion 基准复制，保留语义字段，并增加以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| baseline_stream_id | TEXT | 初始化该工作层的 `fusion:<profile_id>`，之后不可改变 |
| geometry_source | TEXT | 当前几何来源：`fusion`、`sam3` 或 `manual_edited` |
| geometry_revision | INT | 初始为 0；每次实际保存几何修改后加 1 |
| edit_base | TEXT | 人工编辑前的来源：`fusion`、`sam3` 或 `manual_edited`；未人工编辑时为空 |
| sam_session_id | TEXT | 最近一次被采用的 SAM3 会话 ID；没有采用时为空 |
| sam_score | REAL | 最近一次被采用候选的模型分数；没有采用时为空 |
| sam_version | TEXT | 最近一次被采用候选的 SAM3 版本；没有采用时为空 |
| reviewed | INT | 类别确认前为 0，确认后为 1 |
| updated_at | TEXT | 最近一次保存或确认时间 |

Fusion 初始化时 `geometry_source=fusion`、`geometry_revision=0`。采用 SAM3 或普通人工保存时只修改当前工作层要素的 geometry 和上述追溯字段；`run_id`、`object_id`、`part_id`、`class_code`、`class_name`、`baseline_stream_id` 不得在原层内直接改变。只有用户显式执行“更正类别”时，插件才允许保持 geometry、`run_id`、`object_id`、`part_id` 和 `baseline_stream_id` 不变，将对象从原类别工作层移动到目标类别工作层，并把 `class_code/class_name` 改为目标层固定值。几何变化后，`confidence_mean/std` 以 Fusion confidence mosaic 对新几何重新做 zonal statistics；SAM3 自身分数只写 `sam_score`，两者不得混用。原始 Fusion GPKG 不保存编辑状态，也不得被写入。

类别工作层必须为 class_code/class_name 设置固定值约束，并为 object_id 设置唯一、非空约束。用户执行“新增人工面”时，插件必须在要素创建表单校验之前生成 `{run_id}_new_<uuid4>`，预填当前类别和追溯字段，并抑制完整属性表；用户只负责勾画 geometry，不得被要求填写 `object_id` 或其他运行字段。保存后设置 `geometry_source=manual_edited`、`geometry_revision=1`。修改、删除和更正类别都追加到 `classes/edit_history.jsonl`，记录操作、对象 ID、来源/目标类别、geometry hash、局部重叠提示和时间。这样即使要素被删除或移动，仍可追溯到原 Fusion 对象。

“更正类别”一次只处理当前类别工作层中唯一选中的一个面。插件弹出只包含其余 13 类的目标类别选择框；确认后先向目标工作层写入同一 geometry 和同一 `object_id/part_id`，更新 `class_code/class_name`、`geometry_source=manual_edited`、`edit_base`、`geometry_revision+1` 和 `reviewed=0`，再从原工作层删除。任一写入失败必须补偿回滚，禁止丢失对象或静默复制。来源类和目标类均退回 `editing`。与其他类别的重叠只计算、显示并写入历史，本阶段不自动裁剪，也不阻止移动或人工新增；统一留到 final topology 阶段处理。

## 12. SAM3 分类修整工作流

SAM3 从语义主流程中拆出。语义运行成功不依赖 SAM3 是否启用或是否成功。分类工作区依赖 approved Fusion，但不依赖 SAM3：SAM3 不可用时仍可拆分 Fusion、直接人工编辑和最终组装。SAM3 在本工作流中的唯一职责是：针对用户当前点击的地物，在同一位置提供一个可比较、可采用、可继续人工编辑的边界候选。

### 12.1 工作区初始化

1. 用户打开“分类修整与组装”弹窗，选择当前 run 中一个 `ready` 且 profile 为 `approved` 的 Fusion 结果。
2. 插件记录 Fusion GPKG、图层名、文件 SHA256 和 `baseline_stream_id`，并将该 Fusion 设为只读基准。
3. 插件按固定 14 类把 Fusion 面复制到 `classes/class_<class_code>.gpkg`，形成 14 个独立 QGIS 类别工作层；零要素类别也必须创建空工作层。
4. 拆分必须保持要素总数、几何、object_id、part_id、类别、置信度与基准一致，并将新增追溯字段初始化为 Fusion 状态。
5. 工作区一旦产生任何编辑，不允许切换 Fusion 基准；如需更换基准，必须显式放弃当前工作区并重新初始化，禁止把两个 Fusion 的面静默混入同一组工作层。
6. `classes/workspace.json` 持久化基准路径/SHA、14 类图层路径、状态、面数和最后修改时间。重新打开弹窗时先校验文件与基准 SHA，再恢复工作区；基准或工作层被外部修改且无法解释时阻止确认和组装。

### 12.2 默认操作：校正已有地物

这是 SAM3 的主入口，不是整类批处理：

1. 用户在表格中选择类别行并点击“校正边界”。
2. 插件激活该类别工作层，暂存当前 QGIS map tool，然后进入一次性点选状态。
3. 用户在地图上点击该工作层中的一个已有面；点击必须命中当前类别，不能跨层猜测对象，也不能仅按最近距离误选邻类。
4. 插件高亮当前 geometry，显示 class_code、object_id、part_id，并用“选中地物外包框 + `buffer_px`”从原始影像裁取局部窗口。地图点击点作为 SAM3 正点提示；Fusion 几何只用于确定当前对象、裁剪范围和对照，不作为自动正确答案强加给 SAM3。
5. 命中有效要素后立即恢复原 map tool；常驻 SAM3 worker 在局部影像上生成候选 mask，插件将候选转换为地图 CRS 几何并以临时预览覆盖物显示。推理和比较期间地图仍可平移、缩放。
6. 候选出现后，用户必须明确选择“保留当前”“采用 SAM3”“编辑当前”“编辑 SAM3”或“取消”。在用户作出选择前，类别工作层不得发生写入。
7. 选择完成、取消、失败、关闭弹窗或卸载插件时，清除临时高亮和候选预览；若一次性点选尚未结束，还必须恢复进入点选前的 map tool。

### 12.3 五种人工决定

| 决定 | 数据行为 | 追溯行为 |
|---|---|---|
| 保留当前 | 当前工作层几何不变；首次校正时当前几何就是 Fusion | 记录会话为 `kept_current`，不增加 revision |
| 采用 SAM3 | 用候选几何替换同一要素 geometry | `geometry_source=sam3`，revision + 1，写入 session ID、score、sam_version |
| 编辑当前 | 保持现有几何并进入 QGIS 原生编辑 | 保存后 `geometry_source=manual_edited`、`edit_base` 记录编辑前来源、revision + 1 |
| 以 SAM3 编辑 | 先把候选放入当前要素，再进入 QGIS 原生编辑 | 保存后 `geometry_source=manual_edited`、`edit_base=sam3`、revision + 1，并保留 session ID |
| 取消 | 不修改工作层 | 记录会话为 `cancelled` |

人工编辑必须使用 QGIS 原生顶点/重塑工具，允许用户继续移动、增加、删除节点以及修复 Fusion 或 SAM3 都不准确的边界。编辑过程中仍然是原类别工作层、原 object_id 和原类别；不得创建一个“人工层”或“SAM3 层”替代它。存在未保存编辑时，当前类别不能再次执行 SAM3、不能确认，也不能组装最终层。“编辑 SAM3”必须把候选放进 QGIS edit buffer：保存时才正式写入，回滚时恢复进入本次编辑前的工作层几何。

### 12.3.1 快速重画现有面

“人工操作”菜单固定为“快速重画现有面”“新增人工面”“节点精修（备用）”“更正类别”“保存当前编辑”和“取消当前编辑”。快速重画是修改明显错误边界的默认入口，节点精修只作为少量节点调整的备用入口；不得删除现有保存和回滚能力。

快速重画按以下固定交互执行：

1. 用户点击当前类别行的“快速重画现有面”；若当前层已唯一选中一个面则直接使用，否则插件进入一次性地图点选，点击必须唯一命中当前类别中的一个面。
2. 插件保留原 geometry 不变，以灰色轮廓作为参考，并通过 QGIS 4.2 原生 `QgsMapToolDigitizeFeature` 的 `Qgis.CaptureTechnique.PolyBezier` 捕获一个完整新 Polygon。用户右键结束捕获；捕获期间不向工作层新增要素，也不弹属性表。
3. 捕获完成后显示类别色候选预览和“平滑次数”整数控件，范围固定为 `0–5`、默认 `1`。`0` 保留分段化后的 Bézier 结果；`1–5` 使用 QGIS `QgsGeometry.smooth(iterations, 0.25)` 生成实时预览。平滑只作用于本次候选，不修改原面。
4. 候选操作固定为“采用”“重画”“取消”。“重画”清除候选并重新进入 PolyBezier 捕获，原面继续保留；“取消”清除全部临时覆盖物并保持工作层完全不变。
5. “采用”只把候选 geometry 替换到原要素的 QGIS edit buffer，保留 `object_id`、`part_id`、类别和其他不可变字段，不立即提交 GPKG。随后必须由“保存当前编辑”正式提交，或由“取消当前编辑”回滚到上次保存状态。
6. 正式保存沿用普通人工编辑追溯：`geometry_source=manual_edited`、`edit_base` 记录修改前来源、`geometry_revision+1`，重算 confidence 并写入 `edit_history.jsonl`。保存后取消 QGIS 选择高亮，恢复该类别统一颜色并刷新地图。

候选只做非空、Polygon 类型、正面积和 QGIS 几何有效性检查；失败时禁止采用并允许重画，不引入新的自动拓扑修复。与其他面的重叠继续只提示和记录，留到 final topology 阶段处理。

### 12.4 候选生成与坐标契约

- 地图点击点先用 `QgsCoordinateTransform` 从项目 CRS 转到原始 raster CRS，再用 raster 的完整逆 geotransform 转成像素坐标。禁止按地图 extent 比例估算像素位置。
- 裁剪窗口来自被点击要素在 raster CRS 下的 bounds，并按配置 `buffer_px` 扩展；窗口必须裁到 raster 有效范围，且保存窗口 transform、bounds 和尺寸。
- 点击点必须位于实际送入 SAM3 的局部窗口。若坐标转换失败、落在 raster 外或原图不可读，本次会话直接失败，不得调用错误位置。
- 返回多个 mask 时，只保留包含点击像素的有效候选，并在其中选择模型 score 最高者；没有候选包含点击点时判为失败，不使用“面积最大”或“最长轮廓”猜测。
- mask 转矢量采用 0.5 等值线/半像素轮廓，不用逐像元方块外轮廓。必须按轮廓包含关系恢复外环和洞；多部件结果只保留包含点击点的连通部件。
- 候选转换回地图坐标后必须做非空、有效性、点击点包含关系和裁剪边界检查。可按像元尺度做轻量简化，但不得改变洞结构或把候选扩展到窗口外。
- 候选预览使用醒目的青色虚线边界和透明填充；当前 Fusion/工作层几何继续使用该类别统一色。预览只是临时覆盖物，不进入图层树。

### 12.5 常驻 worker 与会话协议

打开分类修整工作区且第一次需要 SAM3 时，插件启动一个独立 worker。worker 加载 checkpoint 一次，后续地物复用同一进程和模型，禁止每点击一个面重新加载模型。插件与 worker 通过 `QProcess` 标准输入输出传输 JSON Lines：

```text
worker_ready
start_session
predict
cancel
close_session
shutdown
```

每个 `predict` 至少携带 `session_id`、run_id、raster、crop window、点击像素、class_code、object_id、part_id、checkpoint SHA、sam_version 和 device。返回事件至少包括 `started`、`candidate_ready`、`failed`、`cancelled`。事件必须带 session_id，迟到结果不得覆盖当前会话。

每次会话追加写入 `refinement/sam3/sessions.jsonl`，至少记录输入对象、坐标转换、裁剪范围、候选 score、用户决定、采用前后 geometry hash、耗时、设备、版本和错误。候选几何可追加到 `candidates.gpkg` 用于审计，但图层默认不加载，且候选记录必须标明 `previewed/kept_current/adopted/manual_base/cancelled/failed`。

SAM3 失败时不得生成 fallback 成功结果，也不得修改工作层。UI 应保留原 Fusion/工作层几何，显示可复制的具体错误，并允许对同一对象重试或直接人工编辑。

### 12.6 次要入口：新增漏标地物

当 Fusion 完全漏掉一个地物时，用户可在选定类别行的菜单中执行“新增漏标面”。此入口不是默认按钮：

1. 用户先明确类别，再在无当前类别面的目标位置点击。
2. SAM3 按点击点和局部上下文生成临时候选，仍需人工采用或取消。
3. 采用后只在当前类别工作层新增一个要素，object_id 使用 `{run_id}_new_<uuid4>`，`part_id=000`、`geometry_source=sam3`、`geometry_revision=1`。
4. 用户可以立即以 SAM3 为基础继续人工修边。

插件不得用 SAM3 自动判断漏标地物属于哪一类，也不得一次扫描整幅影像自动补面。

## 13. 14 类最终组装

分类工作区固定显示 14 行，每一行对应唯一的类别工作层，不再选择模型、Fusion、SAM3 或人工层作为“最终来源”。每个类别只允许处于以下状态之一：

- `editing`：工作层已创建，仍在校正或检查。
- `confirmed`：工作层已保存且人工确认。
- `confirmed_empty`：该类别工作层为零要素，人工明确确认本次范围无该类。

只有 14 行全部确认后，“组装最终图层”按钮才可用。

确认类别时，插件将该工作层现有要素 `reviewed` 统一写为 1，并把状态写入 `workspace.json`。此后任何新增、删除或几何修改都会把类别状态退回 `editing`，受影响要素设为 `reviewed=0`；重新确认后再统一写为 1。空类别只在 `workspace.json` 记录 `confirmed_empty`。

组装规则：

1. 只读取 14 个已确认的类别工作层；不得回读原始 Fusion 或隐藏的 SAM3 candidates 代替工作层。
2. 修复可安全修复的基础几何有效性，但不自动决定类别边界归属。
3. 追加为 `final_composite`，保留来源追溯字段。
4. 检查同类重复、异类重叠、目标范围内缝隙、无效几何和空几何。
5. 所有问题写入 `topology_issues`，不自动裁掉某一类别。
6. 用户在 QGIS 中修复 final_composite 后重新检查。
7. 拓扑检查通过或用户明确带问题确认后，才允许写入 accepted_labels。

`topology_issues` 字段至少包括：

```text
issue_id | issue_type | class_code_a | class_code_b
feature_id_a | feature_id_b | area | severity | resolved | message
```

缝隙检查目标范围为 `requested_extent - accepted_labels`，不把已确认区域误报为 gap。

## 14. accepted_labels

长期标签库继续使用 GeoPackage。字段至少包括：

```text
run_id | object_id | part_id
class_code | class_name
confidence_mean | confidence_std
source_stream_id | source | geometry_source | geometry_revision | edit_base
model_version | fusion_profile_id | sam_version
sam_session_id | sam_score | reviewed | created_at | updated_at
```

写入 accepted 前必须检查：

- class_code 与 class_name 一致。
- geometry 有效且非空。
- object_id 在库内不冲突。
- 同一 run 的 feature 不重复接受。
- source_stream_id 必须是初始化类别工作区的 Fusion 基准，并可追溯到 run_manifest。
- geometry_source 与 revision、edit_base、sam_session_id 的组合符合第 11.1 节契约。

写入后 `source=class_working`，`source_stream_id` 保留 Fusion 基准；真正的边界来源由 `geometry_source/edit_base/sam_session_id` 表达，禁止再用 `refined_id` 或独立 SAM3 图层推断。

## 15. UI：主面板与弹窗协作

### 15.1 主停靠面板

主面板保持紧凑，只承担高频控制：

```text
① 数据源与范围
  影像层
  当前视图 / 手绘矩形
  获取当前视图 / 绘制范围
  范围识别状态

② 切片
  512 x 512 / overlap
  自动扩展推理范围、完整 Tile 数量和步长
  Partition / Seam / Junction 数量

③ 输出位置
  输出工作区
  accepted_labels 路径
  跳过已确认区域

④ 推理环境
  推理脚本目录和 config.yaml 路径
  环境状态和首个问题
  [检查推理环境] [查看完整检查结果]

⑤ 推理方案
  已选模型数量和名称
  fusion profile / 无融合
  设备、配置状态、边界拟合: 公共分界线 Cubic B-Spline
  预计临时空间 / 永久空间 / 峰值工作包
  [选择模型与 Fusion]

⑥ 执行
  [开始] [停止] [推理监控]
  总阶段进度和当前任务

⑦ 结果
  模型流数量 / fusion 状态 / 失败数量
  [打开分类修整与组装]
```

环境检查大表不再长期占据主面板，主面板只显示“正常/警告/错误”和首个问题。

环境检查必须是用户明确点击的操作。推理脚本目录、输出工作区或 accepted 路径变化时，主面板只标记“配置已变化，请检查推理环境”，禁止自动启动耗时检查；环境检查完成前禁用“选择模型与 Fusion”和“开始标注”。环境检查通过后必须在推理方案弹窗中应用本次模型与 Fusion 选择，“开始标注”才可启用。“查看完整检查结果”是检查后的诊断入口，不属于正常必点步骤，复制命令放在详情窗口内部。

“打开分类修整与组装”只在当前 run 存在边界拟合与全部 Seam/Junction `passed`、`ready` 且 `approved` 的 Fusion 流时启用。未选择 Fusion、缺少 report、存在 failed unit 或 formal hash 不匹配时，按钮必须显示具体阻塞原因，不能退回 raw 或某个模型层偷偷初始化工作区。

### 15.2 推理配置弹窗

使用宽弹窗显示完整模型注册表：

| 运行 | 模型 | 版本 | 角色 | Artifact | 设备测试 | 状态 |
|---|---|---|---|---|---|---|

交互要求：

- 每个勾选模型都自动保存独立结果，无额外“保存结果”开关。
- 选择 fusion profile 后，所需模型自动勾选并锁定。
- 显示 profile 策略、模型数量、基线 mIoU、融合 mIoU、approval、路径和 SHA 状态。
- rejected profile 显示为不可运行。
- SAM3 只显示为后处理能力，不作为语义启动必选项。
- 显示只读摘要“边界拟合：公共分界线单次 Cubic B-Spline；两侧 Polygon 共用拟合线”；仅显示平滑因子、输出间距和诊断级别，不展示旧 RDP、面积、Gap/Overlap 或拓扑参数。
- “应用”后将有效方案摘要同步回主面板。

### 15.3 推理监控弹窗

监控弹窗非模态，关闭只隐藏，不能停止任务。主面板和监控弹窗的停止按钮调用同一安全停止入口。

主表按结果流显示：

| 结果流 | 阶段 | 当前/总数 | 状态 | 耗时 | 失败数 |
|---|---|---:|---|---:|---:|

次级详情默认显示当前结果流的 Partition/Seam/Junction 状态，不默认加载全部 Tile。Tile 明细通过筛选和分页从 SQLite 查询，每页最多 `tile_page_size`（默认 500），可按失败、Partition、Tile ID 搜索；标题必须显示完整结果流名称、当前明细类型和已记录总数。

结果流阶段固定为：

```text
queued -> inference -> fusion -> partition_mosaic -> raw_polygonize
-> edge_graph -> boundary_fitting -> seam_junction -> assembling
-> validating -> ready/failed/stopped
```

“成功”只允许表示整个结果流已经 ready；模型 Tile 推理结束只能显示“推理完成/等待后处理”，不得提前计入成功流。每个事件只更新对应结果流或当前可见页的一行，禁止清空重写全表。几何子进程心跳、细分、超时和回退必须实时显示；进度条在一个阶段完成后切换到下一阶段，不能停在 `N/N` 冒充整个任务完成。日志区继续提供 stdout、stderr、系统事件、保存和复制。

### 15.4 分类修整与组装弹窗

弹窗非模态，可与地图交叉操作，按照“固定基准 -> 逐类修整 -> 逐类确认 -> 组装检查 -> 入库”的顺序布局。

#### 15.4.1 顶部：固定 Fusion 基准

```text
Run: <run_id>
Fusion 基准: [fusion:<approved_profile> ▼]  [初始化 14 类工作层]
状态: 未初始化 / 已初始化 / 有未保存编辑 / 14 类已确认
```

- 下拉框只列出当前 run 中 boundary fitting report 为 `passed`、所有 Seam/Junction 通过、formal hash 匹配、`ready` 且 profile 为 `approved` 的 Fusion 流；raw、模型流和 rejected Fusion 不出现。
- 初始化后显示基准文件、formal SHA256、boundary fitting report SHA256、初始化时间和 14 类要素总数。
- 任一工作层产生修改后锁定 Fusion 下拉框。重新初始化必须经过明确确认，并说明会放弃现有工作层修改。
- 已存在合法工作区时再次打开弹窗应恢复原状态，不重复拆分或创建同名空层。

#### 15.4.2 中部：14 类工作表

| 可见 | 类别 | 色块 | 类别工作层状态 | 面数 | SAM3 校正 | 人工操作 | 确认 |
|---|---|---|---|---:|---|---|---|

“类别工作层状态”显示当前层内来源统计，例如 `Fusion 487 / SAM3 3 / 人工 2`，不是可切换的来源下拉框。14 个工作层可同时叠加显示；可见列控制单层显隐，点击类别行会激活该工作层但不强制隐藏其他类别。

表格使用可调整列宽：可见、色块、面数和三个操作列使用稳定窄宽度，“类别工作层状态”列拉伸占用剩余空间；窗口整体可缩放并提供纵向滚动，禁止把长路径、错误堆栈或来源下拉框塞进表格导致操作列被挤出。

- `SAM3 校正`：进入“一次点击一个已有面”的默认流程。按钮旁菜单提供次要入口“新增漏标面”，不提供“整类运行”。
- 类别工作层为空时禁用默认“校正边界”，但仍允许“新增漏标面”。SAM3 环境不可用时，两个 SAM3 入口显示具体阻塞原因，普通人工编辑仍可使用。
- `人工操作`：菜单固定提供“编辑选中面”“新增人工面”“更正类别”。编辑选中面要求当前层唯一选中一个面；新增人工面自动启动 QGIS 多边形捕获并预填全部属性，不弹出完整属性表；更正类别要求唯一选中一个面，再弹出目标类别选择框并移动该对象，不重新勾画 geometry。
- `确认`：把已保存的整个类别工作层标记为 `confirmed`；零要素层使用“确认本范围无该类”。活动 SAM3 会话或任意未保存编辑存在时禁用。
- 已确认类别再次发生几何修改或新增/删除要素后，自动退回 `editing`，必须重新确认。

#### 15.4.3 SAM3 会话区

仅在点选或候选比较期间展开，不长期占用表格宽度：

```text
当前类别: 21 果园       object_id: ...       当前来源: Fusion/SAM3/人工
SAM3 状态: 等待点选 / 推理中 / 候选可用 / 失败
候选 score: 0.923       局部拓扑提示: 无 / 与邻类重叠 ...

[保留当前] [采用 SAM3] [编辑当前] [编辑 SAM3] [取消]
```

- 等待点选时只启用“取消”；推理中允许“取消本次 SAM3”，不结束整个语义 run，也不关闭 worker。
- 候选可用后才启用四个决定按钮。按钮顺序固定为“保留当前 -> 采用 SAM3 -> 编辑当前 -> 编辑 SAM3 -> 取消”。
- 失败时显示完整可复制错误，保留“重试”“编辑当前”“取消”，不生成假候选。
- 候选采用或人工保存后，立即对当前要素与邻接类别做轻量 overlap/invalid 检查并显示提示，但不自动裁掉邻类；正式裁决仍由 final topology 阶段完成。

#### 15.4.4 底部：组装与入库

```text
[组装最终图层] [重新检查拓扑] [写入 accepted_labels]
14 类确认: 11/14    未解决问题: 3    未保存编辑: 0
```

- `组装最终图层` 只在 14 类全部 `confirmed/confirmed_empty` 且没有活动会话、没有未保存编辑时启用。
- `重新检查拓扑` 只针对当前 final_composite，并刷新 topology_issues。
- `写入 accepted_labels` 遵守第 13、14 节门槛，不因 SAM3 候选存在而直接可用。

关闭弹窗时若存在活动点选、推理、候选预览或未保存编辑，必须分别提示处理。关闭弹窗可以结束交互式 SAM3 worker，但不能停止已经完成的语义结果；无论何种退出路径，都不得永久替换地图平移工具或遗留预览覆盖物。

## 16. 异步状态机与 Bash 接口

运行级状态：

```text
idle -> preflight -> queued -> running -> assembling
-> validating -> semantic_ready / partial_failed / failed / stopped
```

Work Package、Partition stream、Seam 和 Junction 各自拥有独立状态；父级状态只能由 SQLite 聚合查询产生，不能靠 UI 猜测。结果流只有 formal、report、所有空间单元和哈希全部通过后才是 ready。SAM3 与 final assembly 是 semantic_ready 后的独立阶段。

### 16.1 SQLite/WAL 数据契约

`run_state.sqlite` 至少包含以下表；几何本体和大日志保存在文件资产中，数据库只保存窗口、关系、状态、摘要、路径和哈希：

| 表 | 主键与核心字段 | 用途 |
|---|---|---|
| `runs` | `run_id, schema_version, status, run_spec_sha256, created_at, updated_at` | 单次运行与总状态 |
| `streams` | `stream_id, kind, model_id/profile_id, version, status` | 模型流与 Fusion 流 |
| `tiles` | `tile_id, row, col, pixel_window, bounds, raster_path, status, sha256` | 共享 Tile 元数据和推理状态 |
| `partitions` | `partition_id, row, col, core_window, halo_window, package_id, status` | 空间分区和工作包归属 |
| `spatial_units` | `unit_id, unit_type, owner_key, pixel_window, status` | Core interior、Seam、Junction 所有权 |
| `work_packages` | `package_id, sequence, estimated_bytes, status, attempt` | 有界调度、停止和恢复 |
| `jobs` | `job_id, job_type, stream_id, tile_id/unit_id, status, attempt, progress, heartbeat_at, pid, error` | 最小可重试执行单元 |
| `artifacts` | `artifact_id, stream_id, unit_id, kind, path, bytes, sha256, status, ref_count` | 原子产物提交与清理 |
| `artifact_dependencies` | `consumer_job_id, artifact_id` | Fusion、Seam、重试和报告依赖 |
| `unit_report_summaries` | `stream_id, unit_id, status, count/max fields, diagnostic_count, report bytes/sha256` | 单元完成时固化报告标量，最终组装不重复解析全部 JSON |
| `object_links` | `stream_id, left_part_id, right_part_id, class_code` | 跨所有权单元连通分量 |
| `events` | `event_id, timestamp, level, event_type, stream_id, job_id, message` | 可复制日志和审计事件 |

必须为 `jobs(status, stream_id, unit_id)`、`tiles(row,col,status)`、`partitions(status,package_id)`、`spatial_units(status,unit_type)`、`artifacts(status,ref_count)` 和 `events(timestamp)` 建索引。`events` 按配置保留最近明细并滚动归档 JSONL，禁止无限增大 WAL。

状态转换使用 `BEGIN IMMEDIATE` 短事务和 compare-and-set；只有持有 lease 的 worker 可以把 job 从 `queued/interrupted` 改为 `running`。心跳超时后 lease 才可回收。Artifact 必须按“临时文件写完并 fsync -> 原子 rename -> 计算 SHA -> 数据库事务标记 ready”提交。恢复时所有 `running` job 先改为 `interrupted`，不能直接判成功或失败。

### 16.2 正式 Shell 接口

正式 Shell 入口：

```text
run_work_package.sh --run-spec <json> --package-id <id>
run_partition_mosaic.sh --run-spec <json> --stream-id <id> --partition-id <id>
run_partition_polygonize.sh --run-spec <json> --stream-id <id> --unit-id <id>
run_boundary_fit.sh --run-spec <json> --stream-id <id> --unit-id <id>
run_seam_job.sh --run-spec <json> --stream-id <id> --unit-id <id>
run_stream_assemble.sh --run-spec <json> --stream-id <id>
run_sam3_interactive.sh --session-root <run/refinement/sam3>
```

`run_work_package.sh` 负责有界模型 score 和 Fusion；`run_unit_fit.sh` 只处理一个可重试空间单元并调用当前公共分界线 Cubic B-Spline；`run_assemble_stream.sh` 只组装已完成单元。正式队列不包含旧亚像元、边界规则化或 Shared Edge/RDP 入口。所有脚本输出 JSON Lines 事件和心跳：

```json
{"event":"fit_progress","stream_id":"model:upernet_swin_b","unit_id":"core_00042","current":8000,"total":12631}
```

### 16.3 失败与重试

失败与重试：

- 单 Tile 失败只重试该 Tile；单空间单元失败只重试或细分该单元，不重新执行已验证邻区。
- Artifact 状态由 SQLite 事务和 SHA 决定；resume 比较 run_spec、模型/profile SHA、Tile 元数据、Partition raster、raw/formal/report、Seam/Junction 与 VRT 引用。
- QGIS 或 worker 非正常退出后，`running` job 在恢复时变成 `interrupted`；验证临时文件后选择复用、重跑或清理，不把它直接当 failed/ready。
- 磁盘低于预留阈值时停止产生新 score，完成可安全提交的在途单元后进入 `paused_low_disk`，不得写坏已有结果。
- 停止时只终止当前子进程组，不结束 QGIS，不破坏地图工具。
- SAM3 的 `cancel` 只取消当前 session；弹窗关闭、插件卸载和 QGIS 关闭时发送 `shutdown`，超时后再终止 worker 进程组。
- worker 迟到事件必须按 session_id 丢弃，不能改写已经取消或已切换对象的会话。

## 17. 代码模块规划

### 插件新增或重构

```text
qgis_plugins/labeling_tool/core/model_registry.py
qgis_plugins/labeling_tool/core/fusion_profile.py
qgis_plugins/labeling_tool/core/run_spec.py
qgis_plugins/labeling_tool/core/result_catalog.py
qgis_plugins/labeling_tool/core/run_state_db.py
qgis_plugins/labeling_tool/core/spatial_partition.py
qgis_plugins/labeling_tool/core/work_package_scheduler.py
qgis_plugins/labeling_tool/core/storage_budget.py
qgis_plugins/labeling_tool/core/class_workspace.py
qgis_plugins/labeling_tool/core/sam3_interactive_controller.py
qgis_plugins/labeling_tool/core/sam3_prompt_map_tool.py
qgis_plugins/labeling_tool/core/final_assembler.py
qgis_plugins/labeling_tool/core/topology_validator.py
qgis_plugins/labeling_tool/gui/inference_config_dialog.py
qgis_plugins/labeling_tool/gui/class_refinement_dialog.py
```

### 当前推理与边界链

```text
inference_scripts/semantic_batch.py
inference_scripts/torchscript_runtime.py
inference_scripts/work_package_runtime.py
inference_scripts/partition_mosaic.py
inference_scripts/incremental_fusion.py
inference_scripts/finalize_partition_rasters.py
inference_scripts/polyline_smoother.py
inference_scripts/common_boundary_smoother.py
inference_scripts/boundary_fitting/unit_runtime.py
inference_scripts/assemble_stream.py
inference_scripts/scale_acceptance.py
inference_scripts/sam3_interactive_worker.py
inference_scripts/run_work_package.sh
inference_scripts/run_finalize_partition_rasters.sh
inference_scripts/run_unit_fit.sh
inference_scripts/run_assemble_stream.sh
inference_scripts/run_scale_acceptance.sh
inference_scripts/run_sam3_interactive_worker.sh
```

### 测试新增

```text
tests/test_run_state_db.py
tests/test_spatial_partition.py
tests/test_work_package_scheduler.py
tests/test_polyline_smoother.py
tests/test_common_boundary_smoother.py
tests/test_partition_seams.py
tests/test_large_tile_monitor.py
```

### 必须重构

```text
inference_scripts/config.yaml
inference_scripts/check_environment.py
inference_scripts/mosaic_builder.py
inference_scripts/polygonize_mosaic.py
inference_scripts/run_polygonize.sh
qgis_plugins/labeling_tool/core/inference_config.py
qgis_plugins/labeling_tool/core/async_runner.py
qgis_plugins/labeling_tool/core/pipeline_plan.py
qgis_plugins/labeling_tool/core/run_spec.py
qgis_plugins/labeling_tool/core/result_catalog.py
qgis_plugins/labeling_tool/core/layer_manager.py
qgis_plugins/labeling_tool/core/layer_names.py
qgis_plugins/labeling_tool/gui/main_dock.py
qgis_plugins/labeling_tool/gui/inference_monitor.py
```

旧 `boundary_fitting/edge_graph.py`、`adaptive_fit.py`、`unit_fitter.py`、`map_precision.py` 及 `tests/test_shared_edge_fitting.py` 已删除，禁止以兼容或回退分支恢复。

### 必须删除的旧运行链

```text
inference_scripts/predict_semantic.py
inference_scripts/run_semantic.sh
inference_scripts/semantic_model.py
inference_scripts/run_sam3.sh
inference_scripts/sam3_class_batch.py
inference_scripts/run_sam3_class.sh
qgis_plugins/labeling_tool/core/inference_runner.py
qgis_plugins/labeling_tool/core/sam3_job_runner.py
```

旧单模型/state_dict 运行链已经删除，并由自动测试禁止恢复；旧 `sam3_class_batch/run_sam3_class/sam3_job_runner` 必须在交互式 worker 通过验收后删除。本项目不维护 Schema v1、整类批量 SAM3 或新旧 SAM3 双轨兼容。

## 18. 实施顺序

| 阶段 | 工作 | 阶段验收 |
|---|---|---|
| E0 | 冻结当前契约并清除生产双轨 | 主 pipeline 中没有整幅亚像元线网、整幅 dissolve 或 Tile 内矢量化入口 |
| E1 | `run_state.sqlite`、WAL、迁移、索引和 Artifact 事务 | 500,000 Tile 明细可写入、分页、恢复；JSON 不承载明细 |
| E2 | Partition/Halo/Core、Seam/Junction 和磁盘预检 | 任意范围得到确定性互斥所有权；空间与缓存不足时启动前阻止 |
| E3 | 有界 Work Package、单加速器 worker、模型独立结果和增量 Fusion | 模型每包只加载一次；缓存、队列和进程数受限；失败可从最小单元恢复 |
| E4 | 分区概率 mosaic、Core raster 分块/VRT、raw coverage | cosine 结果与小范围整图参考一致；不生成单 Tile formal polygon |
| E5S | 公共分界 Polyline 单次 Cubic B-Spline 拟合与两侧重建 | 两侧复用同一拟合坐标、锯齿显著下降；不包含拓扑、Gap/Overlap 或面积验收 |
| E6 | Seam/Junction 重建、磁盘连通图和流式组装 | 空间所有权完整且互斥；跨分区无 gap/overlap；对象 ID 跨单元一致 |
| E7 | 主面板、配置、监控、分页 Tile 查询和真实状态聚合 | 50 万 Tile 不创建 50 万 UI 行；进度不会停在 N/N 假运行 |
| E8 | 现有 Fusion 类别工作区、交互式 SAM3、final/topology/accepted 回归 | 新 formal 流仍能完成已确认的分类修整闭环 |
| E9 | 正式模型、QSDK 边界 A/B、QGIS4 安装与真实运行 | 三模型/Fusion 的精度、边界、拓扑、停止恢复和地图交互通过 |
| E10 | 1k、10k、500k 分级规模验收 | 达到第 19 节规模门槛后才能声明相应规模已支持 |

E1 到 E8 先使用确定性 fixture 完成软件契约，E9 必须使用正式资产和真实影像。E10 中的调度与状态压力测试可以使用 fixture，但不能代替同规模正式端到端运行。

### 18.1 一次性落地顺序

本次改造是一个完整 Goal，不再把“大 Tile”和“边界光滑”拆成两套方案。按以下顺序实施；前一项硬门槛未通过时，不接入后一项正式链路：

1. 冻结现有成功 run 为只读回归资产；从 production pipeline 移除 `subpixel_vectorizer.py`、整幅 probability mosaic 和整幅 geometry union/dissolve 调用；A/B 只使用已冻结输出资产，不保留第二套正式执行分支。
2. 创建 `run_state.sqlite` schema、版本迁移和索引，导入小型 `run_spec.json`，把 Tile、Partition、Work Package、Stream、Seam、Junction、Artifact、Dependency、Retry、Event 全部改为数据库记录。
3. 实现确定性空间规划器，生成 Tile -> Partition -> Core interior/Seam/Junction 所有权；把计划、像素窗口、affine、依赖和预估字节写入数据库。
4. 实现存储预检和低磁盘保护。启动前用真实单 Tile 缓存字节数估计峰值，运行中低于 `min_free_disk_gb` 时完成当前原子写入后进入 `paused_low_disk`，不得继续产生新概率。
5. 重构 semantic 调度为有界 Work Package：单个 MPS/CUDA worker 顺序加载模型，每个模型在一个 Package 内只加载一次；CPU 后处理通过有界队列与推理重叠。
6. 实现逐模型独立分区结果和 Fusion 增量 accumulator。每个模型完成一个 Package 后先提交自身 Partition 资产并更新 accumulator，再按引用计数释放缓存；禁止等待全 run 三模型概率同时齐备。
7. 实现 Partition `Core+Halo` probability cosine mosaic、Core mask/confidence 分块 GeoTIFF 和 VRT；在小范围 fixture 上与整幅参考逐像素对比。
8. 对每个 Partition 生成永久 raw coverage；超过 segment/feature/时间门槛时递归细分，所有几何子进程都有心跳、超时和单元级重试。
9. 实现 `相邻 Polygon -> 公共分界 Polyline -> 单次 Cubic B-Spline -> Resample -> 同时重建两侧 Polygon -> 简单通过/回退检查`；禁止整环分别平滑，不接入旧 Shared Edge、RDP、拓扑修复或 Gap/Overlap 检查。
10. 实现 Seam corridor 和 Junction patch 独立任务；用互斥所有权裁切后流式追加最终 GPKG，通过磁盘连接图统一跨单元 `object_id`，不执行整幅 dissolve。
11. 生成每流 aggregate report、VRT、formal GPKG、诊断边和 Artifact 哈希；只有所有空间单元通过才把结果流标记为 `ready`。
12. 重构监控为数据库聚合和分页查询，默认显示 Partition/Seam/Junction；Tile 每页最多 500 条。所有阶段必须发出可计算进度和心跳，结束、失败、停止后进度条退出忙碌状态。
13. 用新 formal Fusion 回归 E8：14 类拆分、SAM3 单对象候选、人工修边、确认、final、topology 和 accepted 全部保持原契约。
14. 依次完成 238 Tile 故障 run 回归、1k、10k、500k 分级验收。每一级失败都先修复同一级，不通过跳过小规模直接宣称 500k。
15. 每次修改插件源码后执行 `./qgis_plugins/install_qgis_plugin.sh --profile default` 并在 QGIS4 重载；只有实际安装副本与仓库一致，才记录 QGIS 证据。

## 19. 自动测试与实际验收

### 自动测试

- Python 语法与 shell 语法检查。
- Schema v2 正常、旧格式拒绝、路径和 SHA 错误测试。
- 14 类顺序和无 background 测试。
- TorchScript 输入输出契约和设备测试。
- 四种 fusion 策略固定向量测试，特别验证 `log_softmax/temperature/weights` 顺序。
- SQLite migration、外键、WAL、事务、Artifact 原子提交和损坏恢复测试。
- 500,000 Tile、对应 Partition/Seam/Junction 元数据压力测试；分页查询每页不超过 500，禁止把全部行实例化到 UI。
- 任意行列数的 Partition/Halo/Core 规划测试；Core interior、Seam 和 Junction 的 union 等于 processing extent，pairwise intersection 面积为 0。
- Work Package 预算、队列背压、低磁盘暂停、引用计数清理、单元失败重试和停止/resume 测试。
- `output/cache/<run_id>/tile_cache/` 路径冻结、来源影像/用户 Tile 不可删除、跨 Package 共享 Tile 最后引用释放及 cache 符号链接拒绝测试。
- 两个模拟模型、多 Work Package 的端到端测试；每个模型独立输出，Fusion 增量 accumulator 与一次性数学参考逐像素一致。
- 峰值 Tile probability 数不超过计划预算；已提交且无依赖的缓存被清理，仍被 Fusion/Seam/retry 引用的缓存不会提前删除。
- mosaic 重叠区按二维 cosine window 在 14 类概率空间加权，归一化后再统一生成 mask/confidence；Partition 结果与小范围整幅参考一致，禁止回退到中心线硬切。
- Core raster 分块、VRT、scale、CRS、完整 affine、类别顺序、nodata、哈希和量化误差测试。
- raw/formal/report 齐全性、哈希和 resume 失效测试；SQLite 报告摘要与 JSON Artifact 哈希必须一致，摘要完整时最终组装不得重复解析无诊断 JSON；任一拟合或 Seam/Junction 失败时不得留下 ready formal。
- 组装性能回归：`object_id` 使用有界批量查询，raw/formal/诊断 GPKG 使用 `writerecords` 批量事务，报告/诊断分片校验并发数和在途任务数有硬上限；12,635 单元不得产生等量内存对象或逐要素 SQLite 连接。
- raster affine 像素坐标往返测试，覆盖 EPSG:4490、负像元高、旋转/剪切 transform，禁止直接把 px 当地图单位。
- Polyline B-Spline 测试：开线端点严格不动，闭合线首尾一致，输出间距稳定。
- 两个相邻面的台阶分界只产生一份拟合坐标；两侧重建后提取到的公共线与该坐标正向或反向完全一致。
- 同一 Polygon 环同时连接两个邻面时，每条公共分界必须基于当前已提交几何依次尝试；前一条通过后，后一条仍能正确定位并独立通过或回退。
- 岛状面闭合外环和包围面的孔环复用同一份 periodic B-Spline 坐标。
- 每条拟合分界写回左右面后，在像素坐标和最终输出 CRS 中检查 `valid + positive area + pair total area`；失败时两侧共同保留原分界，最终 formal 的 invalid/empty/non-positive 数量必须均为 0。
- 默认使用完整拟合强度，实际偏移只报告，不设置 `1.5 px` 或其他默认硬门槛。
- 自适应细分、心跳、watchdog、子进程退出和二次失败隔离测试；禁止回退到整幅几何操作。
- 静态生产链检查：不得引用 `subpixel_vectorizer.py`、全图 probability mosaic、全图 `unary_union` 或全图 dissolve。
- polygonize 后 confidence zonal statistics 测试。
- object_id/part_id 跨 run 唯一性测试；新增漏标面的 object_id 不与 Fusion 对象冲突。
- Fusion 拆分测试：14 个工作层均创建，几何、ID、类别、置信度和总面数与不可变基准严格一致，初始化前后 Fusion SHA256 不变。
- 工作区恢复测试：重复打开弹窗不重复复制；有修改后禁止静默切换 Fusion 基准。
- 点选测试：只允许命中当前类别工作层已有面，不按最近距离选择邻类；有效命中后立即恢复原 map tool。
- 坐标测试：项目 CRS -> raster CRS -> crop 像素 -> 候选地图几何可逆，覆盖旋转/负像元高/非同 CRS，禁止 extent 比例换算。
- SAM3 worker 测试：checkpoint 只加载一次，连续多个 session 复用进程；session_id 隔离迟到事件；cancel 只取消当前预测。
- 候选测试：只接受包含点击像素的最高分 mask；半像素轮廓保留洞和点击连通部件；无合法候选明确失败。
- 预览隔离测试：候选显示、失败和取消均不修改类别工作层，也不在图层树产生独立 SAM3 结果层。
- 决定测试：保留当前不改 geometry/revision；采用 SAM3 只替换同一 ID 的 geometry；编辑当前和编辑 SAM3 保存后正确写 geometry_source/edit_base/revision，回滚恢复原几何。
- 直接 QGIS 编辑测试：新增人工面不会弹出完整属性表，`object_id` 在创建前自动生成且不冲突；新增、删除或修改工作层要素后类别自动退回 `editing`；活动会话或未保存编辑阻止确认和组装。
- 更正类别测试：唯一选中对象可移动到其余 13 类之一，geometry/object_id/part_id 保持不变，类别与追溯字段更新，来源/目标类均退回 `editing`；目标写入或来源删除失败时不得丢失或复制对象。局部重叠写入提示和历史，但不阻止移动。
- 14 类工作层确认与空类别确认测试。
- overlap/gap/invalid geometry 拓扑问题测试。
- stop/resume/retry、低磁盘暂停和旧 run 不覆盖测试。

### Seam-band 对照标签验收

概率拼接修复必须在同一批真实 tile、同一对照标签和同一有效覆盖范围上，与旧的中心优先 mask mosaic 做成对验收。seam-band 由相邻 tile 重叠区域的中线向两侧各 32 px 构成，参考标签按最终 mosaic 网格栅格化，类别码必须使用固定 14 类映射。

报告至少保存：整体与 seam-band 的 accuracy、mIoU、有效像素数、逐类 IoU、seam-band 与非 seam-band 的差值，以及新旧方案的逐项差值。正式通过条件为：新方案 seam-band accuracy 和 seam-band mIoU 均不低于旧中心优先基线，且 seam-band 相对非 seam-band 的 accuracy 差距不扩大。参考覆盖不足、类别映射失败或 seam-band 无有效像素时必须判为不可验收，不能以成功退出掩盖。

### 公共分界线拟合真实验收

从现有真实语义分割 Core 中选择至少三个包含明显台阶的相邻类别区域。A 为 raw Polygon，B 为 `divider_cubic_bspline_v1` formal Polygon；保存 raw/formal 叠加截图、公共线输入/输出坐标、点数、最大/平均偏移和耗时。

正式通过条件为：公共线只拟合一次，两侧 Polygon 提取出的分界与该拟合线坐标一致；默认拟合强度为 1，开放线端点位移为 0，闭合线保持闭合；实际偏移只供人工判断，不设置统一像素硬门槛。每条拟合线必须通过左右面有效、正面积和总面积守恒判断，否则共同回退原分界；最终 formal 不得包含 invalid、empty 或非正面积 Polygon。本验收不运行 topology、gap、overlap、coverage 或自交修复。人工必须目视确认台阶显著减少、整体轮廓基本一致；否则调整曲线参数或停止，不用复杂 GIS 系统补救。

### 大规模分级验收

规模能力按以下等级独立记录，未通过高一级不能沿用低一级结论：

| 等级 | 数据规模 | 必须执行的链路 | 通过条件 |
|---|---:|---|---|
| L0 故障回归 | 238 Tile | 正式三模型 + approved Fusion + 四流分区矢量与拟合 | 原 `20260717_180420_d9d642` 规模完成；不再生成百万级整幅 linework；四流 ready |
| L1 基础规模 | >=1,000 Tile | 正式三模型 + Fusion 全链路 | 停止/恢复、单元失败重试、VRT/GPKG/QGIS 加载通过 |
| L2 耐久规模 | >=10,000 Tile | 正式三模型 + Fusion 全链路 | 连续运行、磁盘预算、缓存回收、RSS 平台化、监控分页通过 |
| L3 正式上限 | 500,000 Tile | 正式三模型 + Fusion 全链路 | 全部流 ready、无未处理 failed unit、产物哈希完整、恢复与 QGIS 抽查通过 |

此外必须先做 500,000 Tile 元数据/调度 fixture 压力测试，证明 SQLite、分页、计划生成和 UI 不随 Tile 数线性占用常驻内存。该压力测试只能证明控制面可扩展，不能代替 L3 的正式模型运行。只有 L3 实际执行通过后，才能写“已验收支持 500,000 Tile”；在此之前只能写“按 500,000 Tile 设计，已通过 Lx”。

每一级保存 `scale_acceptance_report.json`，至少包括 Tile/Partition/Seam/Junction/Package 数、每流完成数、失败和重试、模型加载次数、峰值缓存、峰值 RSS、磁盘预估与实测、清理字节数、吞吐、总耗时、最长无心跳时间、停止/恢复位置和 Artifact 校验。运行期间内存与临时空间必须在预检上限内形成平台，不得随已完成 Tile 数持续单调增长；磁盘实测偏离预估超过 20% 必须标记 warning 并在下一级前修正估算器。

### QGIS 实际交互验收

1. 执行 `./qgis_plugins/install_qgis_plugin.sh --profile default`，不手动复制；默认目标必须是 QGIS4 profile。
2. 在 QGIS 4.2 中重新加载插件，确认 `labeling_tool 0.3.0` 已启用且源码同步。
3. 当前视图按钮能捕获范围并显示；手绘完成后恢复原平移工具。
4. 配置弹窗能显示 QGIS/Python/PyQt/Qt 版本、Conda Python 路径、模型、profile、真实路径、哈希、设备和错误修改位置。
5. 先使用两模型小范围运行，再执行 L0/L1；地图 UI 不冻结。
6. 主面板、监控弹窗同时显示一致进度；隐藏监控不停止任务。
7. 地图中出现两个独立模型结果组和一个 fusion 结果组。
8. 每个结果流都能打开 mask/confidence VRT 和 formal polygons；诊断时能定位具体 Partition/Seam/Junction 资产。
9. 每个结果流都保存 raw/formal/report，QGIS 默认只加载 formal；监控显示拟合、Seam/Junction、组装阶段和失败原因，Tile 明细分页且标题含完整结果流名称。
10. 选择 boundary fitting report passed 的 approved Fusion 初始化工作区，地图 `Classes` 组准确出现 14 个工作层，Fusion formal SHA 不变，图层树没有独立 SAM3 结果层。
11. 在一个非空类别点击已有 Fusion 地物：点选后立即恢复地图工具，只出现该对象的一个候选预览，推理期间地图仍可平移和缩放。
12. 分别实测保留当前、采用 SAM3、编辑当前、编辑 SAM3 和取消；核对 geometry、object_id、class_code、revision、来源字段与 sessions.jsonl。
13. 实测一次 SAM3 失败：工作层无变化、错误可复制、可以重试或直接编辑；取消后没有迟到结果写回。
14. 在一个 Fusion 漏标位置使用“新增漏标面”，采用后只向当前类别工作层增加一个可继续编辑的唯一对象。
15. 实测“新增人工面”：画完不出现完整属性表，插件自动生成 ID 和当前类别；实测“更正类别”：不重画 geometry 即可把一个错分类对象移动到正确类别层，重叠只记录不阻止。
16. 修改后的类别重新确认；14 类确认后只从工作层生成 final_composite 和 topology_issues。
17. 修正后写入 accepted_labels，再次重叠运行能正确跳过或 difference。
18. 测试语义停止、Partition/Seam 失败重试、低磁盘暂停、SAM3 会话取消、弹窗关闭、QGIS 关闭和插件卸载，不出现崩溃、残留 map tool、预览覆盖物或子进程。

## 20. 完成定义

任务二只有同时满足以下条件才算完成：

- Schema v2 成为唯一配置格式，旧单模型格式不再运行。
- `run_state.sqlite` 是明细真值源，Tile/Partition/Seam/Junction/Artifact 可以事务恢复和分页查询；50 万 Tile 不进入单个 JSON 或常驻 UI 表。
- 所有规模使用同一套有界 Work Package、Partition/Halo/Core、Seam/Junction 链路；不存在单 Tile 矢量化、小图整幅矢量化或大图另一套算法。
- 每个执行模型和 Fusion 都有独立且永久保存的 Core mask/confidence 分块、VRT、raw polygons、formal polygons 和 boundary fitting report；临时概率只在全部依赖提交后按引用计数清理。
- 每个模型流和 Fusion 流都完成 14 类概率空间 cosine 加权 Partition mosaic；Fusion 使用增量 accumulator，不能要求全 run 多模型概率同时常驻。
- 每条 formal 流都对相邻 Polygon 的公共分界线执行一次 Cubic B-Spline，并把同一拟合坐标写回两侧；禁止两个 Polygon 整环分别平滑。
- 曲线拟合报告记录实际偏移、点数、逐边回退数和耗时；除左右面有效、正面积和总面积守恒的简单提交门槛外，不执行 RDP、单面面积约束、Gap/Overlap/Coverage 或 Topology Repair；最终视觉效果由真实 QGIS raw/formal A/B 验收。
- approved profile 能生成独立 fusion 结果，数学实现与 profile 契约一致。
- approved Fusion 能无损初始化 14 个独立类别工作层，原始 Fusion 始终只读且哈希不变。
- SAM3 只对用户点击的当前类别地物生成一个临时候选，不批量扫整类、不自动采用、不创建独立最终层；失败和取消不修改数据。
- 用户可以保留当前边界、采用 SAM3、编辑当前边界或编辑 SAM3 候选；也可以无属性表地新增人工面，或在 geometry 不变的情况下把错分类对象移动到正确类别层，并具有完整 revision/来源追溯。
- 14 类工作层可以叠加显示、独立编辑和确认，并且只从这 14 层组装 final_composite。
- topology_issues 能标记重叠、缝隙和无效几何。
- accepted_labels 写入、skip accepted 和 difference 闭环有效。
- 主面板与三个弹窗状态同步，关闭弹窗不破坏运行或地图操作。
- 停止、重试、resume 和 QGIS 关闭没有残留进程或主程序崩溃。
- 安装脚本同步后通过自动测试和 QGIS 实际交互验收。
- QGIS 插件只使用 QGIS4 自带 PyQt6/Qt6；推理只使用独立 `qgis` Conda 环境，两边没有 `site-packages` 混用。
- 三份正式语义模型在 `qgis` 环境使用 MPS，SAM3 0.1.4 通过现有 compatibility 路径使用 CPU。
- L0、L1、L2、L3 均取得第 19 节规定的正式证据后，才能声明完整的 500,000 Tile 目标完成；未完成 L3 时必须明确当前只通过到哪个等级。

任何仅能运行单模型、只保存 Fusion、不保存各模型结果、把五十万 Tile 明细写入 JSON/Qt 表、保存全 run 多模型概率、在 Tile 内矢量化、构建整幅亚像元线网、执行整幅 dissolve、只看顶点减少率就宣称边界成功、逐 Polygon 独立平滑、使用 raw/失败 Fusion 初始化工作区、自动对整个类别执行 SAM3、把 SAM3 结果另建为最终类别层、未经人工决定就覆盖几何、不能在原类别工作层继续人工修边、或仍会覆盖旧 run 的实现，都不符合本文目标。

## 21. 实施与验收记录（截至 2026-07-17）

本节只记录迁移时的历史状态，不改变前述当前设计和完成定义；最新事实、测试和证据统一记录在 `docs/IMPLEMENTATION_STATUS.md`。旧 E5R、整幅亚像元和旧 mosaic 文件证据不能自动换算为当前 E4-E6 或规模验收。

| 阶段 | 当前状态 | 已取得证据 |
|---|---|---|
| E0 | 方案已冻结 | 本文已明确唯一当前 production pipeline；旧边界算法源码、回退分支和测试已删除 |
| E1-E2 | 待实施 | 当前 run 仍使用 JSON manifest 和整图路径；尚无 SQLite/WAL、空间所有权计划和 100k 预检证据 |
| E3-E4 | 待实施 | 正式多模型与概率 cosine 已有历史能力，但尚未改为有界 Work Package、增量 Fusion、Partition raster 和 VRT |
| E5-E6 | 待实机验收 | 单 Tile 亚像元 A/B 曾通过；238 Tile 运行证明整幅线网不可用；当前改为公共分界线单次 B-Spline，仍需新 run 目视验收 |
| E7 | 部分历史能力可复用 | 监控已有结果流标题和增量 Tile 刷新，但尚未改为 SQLite 的 Partition/Seam/Junction 聚合和分页查询 |
| E8 | 历史闭环已通过，需对新 formal 回归 | QGIS4 的 14 类、交互式 SAM3、final/topology/accepted 已有证据；当前 Fusion formal 尚需复验 |
| E9 | 部分通过 | 正式模型、Fusion、MPS、SAM3 CPU 和 QGIS4 已有证据；公共分界线 B-Spline 当前全链路和安装后实机尚未通过 |
| E10 | 未通过 | L0 的 238 Tile 旧链路未完成；L1/L2/L3 尚未执行，不能声明 1k/10k/100k 已支持 |

旧架构的直接失败证据是正式 run `output/runs/20260717_180420_d9d642/`：14 x 17 共 238 Tile，Swin-B 与 SETR 流到达旧 `ready`，MambaOut-B 在整幅 5632 x 4672 probability mosaic 上生成 6,751 个 raw 面后进入全图亚像元线网。日志在 `4671/4671` 时已经累计 `1,033,510` 个 segments，但没有后续 `step_finished`；`run_manifest.json` 仍为 run `running`、MambaOut `regularizing`、Fusion `pending`。用户随后停止运行。该记录证明旧方法即使完成逐行建线，也可能在全图 polygonize/重建阶段长时间无心跳，不能扩展到 10 万 Tile；它是 L0 失败证据，不是算法完成证据。

变更前自动测试历史结果为 `38 passed, 2 skipped`；它覆盖旧 SAM3 class 批处理，不覆盖第 12、15.4、16、19 节新增的交互式契约。跳过项所需的 Fiona/GDAL 能力曾在实际 `geoai` 环境通过 polygonize 与旧 SAM3 class smoke 脚本。Python 编译与 Bash 语法检查当时通过；当时实际 QGIS 环境为 QGIS 3.40.3、Python 3.12.12、Qt 5.15.8。

上述 QGIS 3.40.3/Qt5 内容是迁移前历史证据。2026-07-15 起正式目标切换为 QGIS 4.2 DMG、Python 3.12.11、PyQt 6.11.0、Qt 6.11.1 和独立 `qgis` Conda 推理环境；QGIS3 历史验收不能替代 QGIS4 最终验收。

测试资产完整运行记录：

- 成功 run：`output/e9_acceptance/runs/20260714_011539_d8dbb5/`。
- `run_manifest.json` 状态为 `ready`，结果流为 `model:fixture_forward`、`model:fixture_reverse`、`fusion:e9_equal_average`。
- 三条流各自保存有效的 `mask_mosaic.tif`、`confidence_mosaic.tif`、`semantic_polygons.gpkg`，并记录 SHA256。
- QGIS 实际加载 9 个图层，分别进入 `Models` 和 `Fusion` 组；三条矢量层均可读取。
- `logs/e9_mask_canvas.png` 验证索引值 `0..13` 使用 14 类调色板可见渲染；索引 0 标签为 `12 水浇地`，索引 13 标签为 `71 河湖库塘`。
- 停止/恢复复验 run：`output/e9_stop_recheck/runs/20260714_020427_98c320/`。运行中停止后 manifest 持久化为 `stopped`，当时流状态为 `stopped/pending/pending`；QGIS 仍可响应且地图工具保持 `QgsMapToolPan`。随后对同一 `run_spec` 执行 resume，三条流均到达 `ready`，没有创建替代 run。
- 失败重试 run：`output/e9_acceptance/runs/20260714_013648_af2c22/`。首轮故意损坏 `tile_0_1` 后两个模型流均准确记录单 tile 失败；恢复原始 tile 并调用 `retry_failed()` 后，每个模型复用 1 个成功 tile、只补跑 1 个失败 tile，最终两个模型流和 fusion 流全部 `ready`。
- resume 校验会重新计算模型 artifact、tile、fusion profile snapshot、mask mosaic、confidence mosaic、semantic polygons 和 difference review layer 的 SHA256；任一输入或输出被修改后，ready 流都会失效并重建。
- QGIS 实际拓扑复验同时检出 `invalid_geometry=1`、`cross_class_overlap=1`、`gap=1`，且检查后 QGIS 保持可响应。
- `object_id` 已在两个不同 run 的实际 GeoAI polygonize smoke 中验证互不相交；旧 SAM3 class job 的 `refined_id` 曾验证唯一且带 run 前缀。新工作流不再使用 `refined_id`，该历史证据不能替代新 object/revision/session 契约测试。
- E1-E8 测试资产阶段的 QGIS 进程内环境检查确认语义设备为 `mps`，SAM3 设备为 `cpu`，14 类契约和官方 SAM3 checkpoint 均为 `ready`；该结果不替代正式三模型的 MPS 复验。
- 安装脚本完成同步后，排除 `.DS_Store/__pycache__` 的 rsync dry-run 无差异；插件连续两次热重载成功，QGIS MCP 仍返回 `pong`。
- 正式资产部署前的配置弹窗曾显示 5 个候选模型登记项和 1 个未就绪 profile，并正确阻止启动；这只是缺失资产提示能力的历史证据，不是正式模型验收结果。
- 当前视图按钮能立即给出范围/影像重叠状态；手绘矩形工具激活后可恢复 `QgsMapToolPan`。关闭监控弹窗只隐藏窗口，没有发出停止信号。
- 停止、恢复和再次热重载后，QGIS 主进程内检查没有发现 semantic/fusion/mosaic/polygonize/SAM3 残留子进程。

正式 E9 已取得证据：

- 正式来源只读目录为 `/Users/example/Desktop/remotesensing/Loess/work_dirs/fusion/l2_fusion_v1/`。批准的 `l2_fusion_v1` profile 实际引用 `upernet_swin_b`、`setr_vit`、`upernet_mambaout_b` 三个模型；该部署包没有 ConvNeXt 或 SegFormer TorchScript，二者不能作为正式资产登记或冒充验收。
- 三份正式 TorchScript 与 `fusion_profile.json` 已部署到 `weights/`。模型 SHA256 分别为 `e1b28d88821f0a35e17e399c89d01ef010c8a73dd1dce797f4db2f2b17425214`、`7e47de54db003a802a36f78bb26295954201467234fbb084e00db4925a74a12f`、`76819fa558ce4261033fc6b0d65353778ec2a86cb4b0a0ce4a5ba8fd03ae1054`；profile 文件 SHA256 为 `1e4a3086d016b26c074499a2035fea28ad0a4056e1cf3539fb04eb5e2f8c615d`。`config.yaml` 不再含占位 SHA。
- profile 为 `approved`，策略为 `equal_probability_average`，类别顺序为固定 14 类，三模型引用、artifact、SHA、temperature 和 14 x 3 权重均通过 Schema v2 校验；validation/test overlap 为 0，profile 记录的测试 mIoU 为 `68.45298388928819`，高于 baseline `65.72565908650316`。
- 正式 CPU 基线 run 为 `output/e9_formal_cpu/runs/20260714_215037_071adb/`，输入来自真实影像 `data/raw/Google_loess/loess_1m_clip_RGB_4490.tif` 的中心 512 x 512 窗口，CRS 为 EPSG:4490。三个模型分别生成并永久保存 mask、confidence、mosaic 和 `semantic_polygons.gpkg`，面数为 65、74、113；四条结果流的最终文件 SHA 已写入 `run_manifest.json`。
- 正式 fusion 独立保存于 `fusion/l2_fusion_v1/`，生成 108 个矢量面；mask、confidence 与三个模型结果范围完全一致。对同一 run 执行 resume 时，三个模型和 fusion 均复用 1/1 tile，0 failed，没有创建替代 run。
- 历史旧链路曾从正式 Fusion 层选择 `21 果园` 执行官方 SAM3 CPU 批量修整，输出 `refinement/fusion_l2_fusion_v1/class_21.gpkg`；1 个对象 success，0 fallback，0 failed，checkpoint SHA 为 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`，父流、父对象、类别和版本关联完整。该记录只证明 checkpoint 与旧批处理可运行，不满足当前交互式 E7 验收。
- 受限终端内环境检查为 warning 的唯一设备原因是该终端看不到 MPS；三份正式模型的 CPU TorchScript 契约、approved profile、14 类映射、官方 SAM3 0.1.4、tokenizer、checkpoint 与 SAM3 CPU 设备均为 ready。
- 正式冻结 TorchScript 在 MPS 上不能仅依赖 `module.to('mps')`。当前运行时保留 artifact 和 SHA 不变，将非标量 float32 冻结常量迁移到 MPS，并仅把 MPS 不支持的 UPerNet adaptive average pool 桥接到 CPU，报告 `mps_frozen_hybrid`。Swin-B 的 QGIS 启动子进程实测输出为 `[1,14,512,512]` float32 MPS tensor；与 CPU 的 argmax 一致率为 `0.9999961853`，confidence MAE 为 `1.60866e-6`。SETR 与 MambaOut 尚需同样的 QGIS 进程内正式复验。

2026-07-17 正式 E7/E8 与 QGIS4 复验结果：

- 旧整类批处理 `sam3_class_batch.py/run_sam3_class.sh` 已删除；旧名 `sam3_job_runner.py` 已由职责明确的 `sam3_worker_runner.py` 取代。自动测试禁止旧入口恢复，环境检查和运行时指纹只引用交互式 worker。
- QGIS4 实际进程使用正式 run `output/runs/20260716_100240_8aafa0/` 恢复 14 个类别工作层、132 个面。交互式 SAM3 CPU 对真实 13 旱地对象返回 score `0.7264912724` 的候选；地图工具按 `QgsMapToolPan -> QgsMapToolEmitPoint -> QgsMapToolPan` 恢复，保留当前、采用 SAM3、编辑当前、编辑 SAM3、取消和新增漏标面均取得实际进程证据。
- SAM3 候选提交后会重新读取 GPKG 实际落盘几何。候选 WKB 哈希与 OGR 规范化后的落盘哈希可以不同，但 `edit_history.jsonl` 和 `sessions.jsonl` 的 `after_geometry_hash` 均以落盘值为准；本次实测二者都等于 `428e449d26a79965e3b632a7fd972aedb5f2e7e8be09014157283f8d8cf01bb6`。
- 14 类均已确认，其中空类别明确记录为 `confirmed_empty`；所有非空要素 `reviewed=1`，没有未保存编辑或活动 SAM3 会话。组装得到 132 面 `final_composite`，拓扑检查为 0 项，随后写入 132 面 `accepted_labels`。
- 关闭分类修整窗口后 worker 为 `None`，QGIS 进程子进程清单中没有 semantic、fusion、mosaic、polygonize、boundary regularizer 或 SAM3 worker；地图工具保持 `QgsMapToolPan`。
- 最终安装后，排除 `.DS_Store/__pycache__` 的安装目录 rsync dry-run 无差异；QGIS 已加载 `labeling_tool 0.3.0` 和新 `sam3_worker_runner`，没有加载旧控制器。通过真实“检查推理环境”按钮后，按钮恢复可用、状态为 ready、正式 run 自动恢复。

正式 Goal 尚不能标记完成：历史单 Tile 亚像元 A/B 和 E8 人工闭环证据继续保留，但 E9 尚未完成公共分界线 Cubic B-Spline 与新 pipeline 的 QGIS 实机验收，E10 只取得 L0 失败证据。下一步按第 18.1 节继续，不得恢复旧整幅线网或复杂 Shared Edge 路线。
