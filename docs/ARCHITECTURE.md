# 当前架构与数据合同

## 1. 文档地位

本文只描述当前仍有效的架构和不可破坏合同。实际完成程度与部署状态见
[CURRENT_STATUS.md](CURRENT_STATUS.md)。

交互式架构视图集中维护在 [visualizations/](../visualizations/README.md)：

- [高层运行时架构](../visualizations/loess-qgis-runtime-architecture.html) 展示 QGIS、
  Conda Worker、PostgreSQL、Artifact 和 macOS/Ubuntu 双平台运行边界；
- [数据库连接修改前后](../visualizations/loess-qgis-database-connection-before-after.html)
  展示 PostgreSQL-only 连接合同以及与 GeoPackage 数据层的区别。

交互图用于帮助阅读，架构真值仍由当前代码和本文书面合同共同约束。图表必须由
同目录 JSON 源规格生成，不得直接编辑生成后的 HTML。

## 2. 权威源码与平台

GitHub `anyun-hy/loess-qgis` 的受保护 `main` 是 macOS 与 Ubuntu 的权威源码：

- Ubuntu：QGIS 3.44、Qt5/PyQt5、独立 `qgis` Conda、CUDA/RTX 3090；
- macOS：QGIS 4.2、Qt6/PyQt6、独立 `qgis` Conda、MPS；
- 两个平台安装相同插件、推理运行时和 Bash 部署入口；
- `loess-project` 是生成的部署项目，不是源码仓库；
- 权重、输入、QGIS 工程、人工标签和输出由用户控制，不随源码部署覆盖。

## 3. 代码边界

| 区域 | 职责 |
|---|---|
| `qgis_plugins/labeling_tool/` | QGIS UI、地图交互、运行编排、监控、人工修整 |
| `inference_scripts/` | 环境检查、Tile 推理、Fusion、Partition、V3/V3.3、组装和验收 |
| `bash/` | 插件安装、部署项目初始化、可选 SSH 入口 |
| `tests/` | 契约、恢复、故障、规模和平台兼容测试 |
| `docs/` | 当前架构、状态、操作和长期决策 |

QGIS 插件进程只使用宿主 QGIS 的 Python/Qt。TorchScript、Fusion 和其他推理
任务只使用当前平台的 `qgis` Conda 子进程，禁止混用两边 `site-packages`。

## 4. 正式运行数据流

详细函数、进程、Artifact、并发和恢复关系见
[diagrams/PRODUCTION_CALL_GRAPH.md](diagrams/PRODUCTION_CALL_GRAPH.md)。

```text
影像 + 完整研究范围
  -> Run Spec + PostgreSQL 状态图
  -> 每个位置只物化一份共享 Tile 缓存
  -> 三个模型依次读取同一 Tile 缓存并独立推理
  -> Partition Halo cosine 概率拼接
  -> 各模型独立 Core mask/confidence + 增量 Fusion
  -> Fragmentation V3 冻结基线
  -> V3.3 Partition 与后续 Work Package 流水并行
  -> Fusion 单元 lossless max-confidence 压缩并提前释放 14 波段概率
  -> Fragmentation V3.3 全局 Finalize 权威 Core
  -> 四条 Stream 的 Core/Seam/Junction 空间单元矢量化
  -> GeoParquet 分片 + 边界签名
  -> 四条 Stream 并行组装为最终 GPKG
  -> 完整范围 gap/overlap/outside 硬验收
  -> QGIS 结果层与人工修整
  -> final/topology/accepted_labels
```

## 5. 空间和规模合同

- `512 x 512` 是模型 Tile 尺寸，不代表地面分辨率；
- 物理面积和距离必须由当前影像 Affine 与 CRS 计算；
- Halo 只提供上下文，互不重叠的 Core 拥有正式像元；
- 冻结矢量研究范围栅格化时保留所有 touched 边界像元，保证 raster support
  不在非像元对齐边界内缩；正式 GPKG 再以同一冻结矢量精确裁剪；
- 中间 touched Core 可以部分位于矢量范围外，但正式结果仍必须通过
  gap=0、overlap=0、outside=0，不能把中间像元计数当成精确矢量面积；
- 大图使用有界 Work Package 和 Partition，不分配整幅概率或整幅线网；
- Tile、Partition、Stream、Job 和 Artifact 明细以 PostgreSQL 为状态真值；
- Run JSON 只保存冻结配置和摘要，不保存几十万 Tile 明细；
- 临时 Tile/probability Artifact 只有在依赖提交后才按引用关系清理；V3.3 Run
  中，Fusion 空间单元先把归一化 14 类概率的逐像元最大值无损保存为 float32
  `unit_confidence`，V3.3 与该压缩任务都释放依赖后即可清理原概率 Halo；
- 磁盘准入冻结永久结果、安全底线、原子写开销、紧凑 `unit_confidence` 全额和
  一个有界 Work Package。所有概率 Halo 的理论全量只作为运行时托管 Artifact
  上限，不是必须同时存在的准入需求；实际增长由引用清理和运行时 backpressure
  约束。
- 最终成品大小预测是观察指标：新 Run 冻结单一预测值，结束后回报实际 ready
  Artifact、带符号差额和比例；不设置上下限、不生成 warning，也不参与任何
  验收或磁盘准入决策。

## 6. 模型与 Fusion 合同

- 每个实际执行模型保留独立结果身份；
- approved Fusion 是独立结果流，不覆盖模型结果；
- Fusion 按冻结 profile 对 14 类概率执行算法融合，不做矢量几何叠加；
- 当前正式 profile 引用 Swin-B、SETR 和 MambaOut-B 三个模型；
- 模型、profile、输入和关键输出必须记录 SHA-256；
- 单个 Run 的可恢复性不能依赖另一个 Run 的临时缓存。

## 7. 权威栅格与碎片治理

- Fragmentation V3.3 是当前生产碎片治理方案；
- V3 先从 Fusion 概率生成冻结基线和上下文，V3.3 再统一裁决所有 Core，
  V3.3 输出才是矢量化使用的唯一权威 Core；
- V3.3 必须保持单标签，不得产生 gap、overlap 或范围外发布；
- 每个 V3.3 Partition 在自己的全部依赖 Work Package ready 后即可领取，并可与
  GPU 后续 Work Package 并行；全局 Finalize 仍必须等待所有 V3.3 Partition；
- 每个 Fusion Core/Seam/Junction 在自己的概率依赖 ready 后先生成 lossless
  `unit_confidence`；正式 Fusion 空间单元必须同时等待该置信度 Artifact 和 V3.3
  全局 Finalize，类别仍只读取 V3.3 权威 Core；
- V3 保留为新 Run 的明确回滚选项，但同一个 Run 内不允许 V3.3 失败后静默退回
  V3；
- Generate 只作研究参考，不是生产入口；
- 失败 RAG 与空间联合解码已归档，不得从 archive 导入生产；
- V3.1—V3.4 的同域实验最终选择 V3.3；
- V3.3 规则存放在可查询、严格校验的版本化配置中，执行器不得使用隐藏默认值
  覆盖类别权限、面积阈值、关系优先级或冲突顺序；
- V3.1、V3.2、V3.4 的源码和测试只保存在实验分支历史，不属于主干当前文件。

### V3.3 approved Fusion 唯一性硬门

单模型概率实验中保持 zero-gap、zero-overlap、zero-outside，不能自动证明
approved Fusion 迁移后仍满足相同合同。迁移或接入前必须独立验证：

- strict-valid 范围内每个像元都有有限、非负且和为 1 的 14 类 Fusion 概率，
  Fusion coverage weight 不得为 0；
- hard label 使用冻结 `CLASS_ORDER` 的稳定 argmax，每个有效像元只发布一个
  类别；最高概率并列及 near-tie 必须单独计数和审计，但并列概率本身不允许
  变成多标签或跨类别 overlap；
- Partition Core 对完整范围精确覆盖一次，Halo 不发布；proposal 只能由一个
  冻结 owner 发布，不能重复改写同一像元；任何 owner 都无法完整裁决的跨 Core
  footprint 不执行 V3.3 改写；
- V3.3 后的权威 Core mosaic 必须逐像元验证 single-label、gap=0、overlap=0、
  outside=0、invalid 保持和完整范围 coverage；
- 矢量化及全流组装后必须再次执行 gap、overlap、outside 硬验收，不能用栅格
  阶段通过替代矢量阶段通过。

## 8. 矢量与边界合同

- 相邻 Polygon 的公共分界只拟合一次，两侧复用同一坐标；
- 只允许误差受限的公共分界处理，禁止逐 Polygon 独立平滑；
- 不在最后执行整幅 dissolve；跨单元同类对象通过连接关系取得统一身份；
- 正式组装必须验证完整范围、无效几何、gap、overlap 和 outside coverage；
- coverage 验收失败的 Stream 不得标记为 ready。

## 9. 人工修整和长期标签

- 正式 Fusion 是只读基准；
- 用户按 14 类工作层比较、编辑和确认；
- SAM3 只提供当前对象的临时候选边界，不负责类别判断；
- 未经用户决定不得覆盖正式几何；
- 所有类别确认后才生成 final、topology issues 和 accepted labels；
- accepted labels、对象来源和 revision 必须可追溯。

## 10. 状态、恢复与部署

- Run 控制面只使用 PostgreSQL；历史文件状态库不再读取或恢复，需用当前部署创建新 Run；
- Job 领取、Artifact 引用、失败重试和清理必须事务化；
- 新 Run 的完整控制图创建成功后，旧 `failed/stopped` Run 的数据库过程明细
  在同一事务中归档清理；保留不可恢复的 Run 墓碑、spec 哈希、规模计数和有界
  错误摘要，不删除任何输出文件；
- `ready`、当前新 Run、非终态 Run、`resetting` Run，以及仍有 running Job 或
  有效 lease 的 Run 不参与自动归档；归档或记录失败只产生可见 warning，不阻止
  新 Run；
- live PID 不代表 Run 成功，必须以 Job、Stream、Artifact 和 hard gate 收口；
- `bash/install_plugin.sh` 安装共享插件；
- `bash/init_project.sh` 初始化或更新部署项目；
- 远程 Ubuntu 操作可通过 `bash/ssh_tencent.sh`，主机别名可由环境变量覆盖；
- 部署成功、自动测试通过和 QGIS 实机验收是三个不同结论。

## 11. 变更原则

任何新实现都必须先说明它影响的数据合同、状态图、空间所有权、永久 Artifact
和跨平台行为。研究候选不得直接接入正式入口；通过自动测试但没有真实资产或
QGIS 证据时，不得标记为生产完成。
