# 当前生产调用关系图

## 1. 文档范围

本文描述当前 QGIS 自动推理主链，以及自动推理完成后的人工整理链。它说明：

- 谁调用谁；
- 输入何时读取、缓存、复用和清理；
- GPU、CPU、V3.3 和四流组装如何并行；
- PostgreSQL、文件 Artifact 和恢复机制如何配合；
- 哪些脚本不属于当前自动生产入口。

类别规则、面积阈值和 V3.3 冲突裁决见
[../operations/FRAGMENTATION_V3.md](../operations/FRAGMENTATION_V3.md)，本文不重复定义。

## 2. 自动推理总图

```text
用户在 QGIS 选择影像、范围、模型、Fusion、overlap、Accepted
                              │
                              ▼
main_dock.py
  ├─ 环境、权重、SHA、设备和 Batch 探测
  ├─ 生成 Tile 网格和范围选择
  ├─ 用正式物化路径测量一个真实 Tile
  ├─ 冻结范围矢量与 Accepted 快照
  └─ 解析空间、存储和并发参数
                              │
                              ▼
run_builder_v5.create_v5_run
  ├─ run_spec.json
  ├─ PostgreSQL Run / Stream / Job / Artifact 依赖
  ├─ Work Package
  ├─ Partition / Core / Seam / Junction
  ├─ V3.3 Partition + Finalize Job
  ├─ Fusion unit_confidence Job
  └─ 四条 Stream 的 unit_fit Job
                              │
                              ▼
V5AsyncInferenceRunner.run_from_spec
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
GPU Work Package       CPU V3.3 / confidence / unit_fit    PostgreSQL监控
三模型 + Fusion        碎片裁决、概率压缩、矢量化           租约/进度/恢复
       │                      │                      │
       └──────────────┬───────┴──────────────────────┘
                      ▼
          V3.3 Partition × 最多4
                      │
                      ▼
              V3.3 全局 Finalize
                      │
                      ▼
        finalize_partition_rasters：四流 VRT
                      │
                      ▼
       assemble_stream：四条 Stream 最多4个并行
                      │
                      ▼
        range clip + coverage + Accepted差分
                      │
                      ▼
               scale_acceptance
                      │
                      ▼
           QGIS加载四条正式结果流
```

## 3. QGIS 到任务图

```text
main_dock._start_inference_after_tile_cache_probe
  │
  ├─ ModelRegistry：冻结三个模型与 approved Fusion
  ├─ plan_spatial_units：规划 Partition/Core/Seam/Junction
  ├─ storage_preflight：冻结缓存和永久输出预算
  ├─ reserve_run_directory：创建 Run 专属目录
  ├─ _freeze_pending_range_snapshot：冻结范围
  ├─ _freeze_pending_accepted_snapshot：冻结 Accepted
  └─ create_v5_run
       ├─ 写快照与 run_spec.json
       ├─ 创建 PostgreSQL 任务图
       └─ 返回 spec_path 和 state database
              │
              ├─ InferenceMonitorDialog.bind_state_database
              └─ V5AsyncInferenceRunner.run_from_spec
```

`run_spec.json` 冻结本次 Run 的输入、模型、Fusion、范围、空间计划、资源、规则和
部署身份。PostgreSQL 保存可变运行状态，Run JSON 不承担几十万条任务明细。

## 4. Tile、三个模型和 Fusion

### 4.1 每个位置只切一份 Tile

```text
原始大影像
    │
    │ tile_materializer._materialize_one
    │ 每个 Tile 位置只物化一次
    ▼
Run专属 Tile 缓存：512×512 GeoTIFF
    ├─ 模型1重新读取同一缓存
    ├─ 模型2重新读取同一缓存
    └─ 模型3重新读取同一缓存
```

因此：

- 原始影像切割：每个 Tile 位置一次；
- Tile 缓存读取：每个模型各读取一次；
- Fusion：只读取三个模型的概率，不重新读取原始影像或 Tile；
- 缓存有效时恢复运行直接复用，不重新物化。

### 4.2 单 GPU 上的模型顺序

```text
同一个 Work Package
  │
  ├─ 模型1 Batch 推理
  │    ├─ 概率 checkpoint
  │    ├─ 模型1 Partition mask/confidence
  │    └─ 写入 Fusion accumulator
  │
  ├─ 模型2 Batch 推理
  │    ├─ 概率 checkpoint
  │    ├─ 模型2 Partition mask/confidence
  │    └─ 写入 Fusion accumulator
  │
  ├─ 模型3 Batch 推理
  │    ├─ 概率 checkpoint
  │    ├─ 模型3 Partition mask/confidence
  │    └─ 写入 Fusion accumulator
  │
  └─ FusionAccumulator.finalize
       ├─ Fusion概率
       ├─ Fusion confidence
       └─ Fusion原始类别
```

三个模型在一张 GPU 上依次运行，不同时占用 GPU。每个模型内部并行读取 Tile，
并让概率 checkpoint 写入和 Partition 构建形成有界流水线。模型对象由
`PersistentModelProvider` 跨 Work Package 保持驻留。

## 5. V3 与 V3.3

```text
Fusion概率
    │
    ▼
V3规则处理
    ├─ v3_baseline_core
    ├─ v3_context_core
    ├─ partition_probability
    └─ core_confidence
            │
            ▼
等待该 owner 的依赖 Work Package ready
            │
            ▼
V3.3 Partition Job，最多4个并行
    ├─ 读取冻结 V3 baseline
    ├─ 读取邻接 V3 context
    ├─ 读取 Fusion probability
    ├─ 执行冻结 V3.3 规则
    └─ 发布 staged mask + audit
            │
            ▼
同一批 probability 还供 unit_confidence
    ├─ 计算归一化 14 类概率的逐像元 max
    ├─ 无损写入 float32 单波段空间单元 Artifact
    └─ 与 V3.3 消费者都完成后释放 probability Halo
            │
            ▼
V3.3 Finalize 单一发布屏障
    ├─ 等待全部 V3.3 Partition ready
    ├─ 检查完整 Core 所有权
    ├─ 审计全域4连通对象和碎片
    ├─ gap/overlap/outside/invalid = 0
    ├─ 保护类损失 = 0
    └─ 原子发布唯一权威 Fusion core_mask
```

模型 Stream 不依赖 V3.3，可在 GPU 继续处理后续 Work Package 时由 CPU 提前执行
空间单元任务。V3.3 Partition 和 `unit_confidence` 也按依赖就绪流式执行。
Fusion Stream 的正式空间单元任务必须等 V3.3 Finalize 发布权威 Core，并等本单元
`unit_confidence` ready 后才能领取。

## 6. Core、Seam、Junction 空间单元

每条模型 Stream 和 Fusion Stream 都执行同样的单元任务：

```text
模型：Partition probability Halo + 权威 Core mask
Fusion V3.3：unit_confidence + 权威 Core mask
                  │
                  ▼
boundary_fitting.unit_runtime.run_unit_fit
  ├─ 模型流拼接概率上下文；Fusion 读取紧凑 confidence
  ├─ 拼接不重叠的权威类别
  ├─ 像元转 Polygon
  ├─ 公共分界拟合或保持原边界
  ├─ 概率只用于 confidence，不重新判类
  └─ 发布单元 Artifact
       ├─ raw.parquet
       ├─ formal.parquet
       ├─ fitted_edges.parquet（存在拟合边时）
       ├─ boundary_signatures.json
       └─ report.json
```

CPU `unit_fit` 可以多进程并行。每个进程只写自己 Stream 和 Unit 的文件，不共享
可写 GPKG。

## 7. 栅格收口与四流组装

```text
全部任务 ready
    │
    ▼
finalize_partition_rasters
  ├─ 每条 Stream 的 mask_mosaic.vrt
  ├─ 每条 Stream 的 confidence_mosaic.vrt
  └─ Stream状态改为 raster_ready
    │
    ▼
四条 Stream 最多4个独立 assemble_stream 进程
  ├─ 模型1组装
  ├─ 模型2组装
  ├─ 模型3组装
  └─ Fusion组装
```

每条 Stream 内部：

```text
GeoParquet + boundary signatures
    │
    ├─ 校验字段、CRS、数量和SHA
    ├─ 相邻单元按边界区间连接
    ├─ 生成稳定 object_id
    │
    ▼
Pyogrio Arrow批量写最终GPKG
    ├─ semantic_polygons_raw.gpkg
    ├─ semantic_polygons.gpkg
    └─ fitted_edges.gpkg
    │
    ▼
精确范围裁剪
    │
    ▼
矢量覆盖硬验收
    ├─ gap = 0
    ├─ overlap = 0
    └─ outside = 0
    │
    ▼
Accepted差分 + 正式Artifact提交
    │
    ▼
清理该Stream的单元GeoParquet和签名
```

磁盘空间锁只用于短时登记空间预算，四个独立文件的实际写入可以并行。

## 8. 整体验收与监控

```text
scale_acceptance
  ├─ Work Package、V3.3、unit_confidence、unit_fit和Stream是否收口
  ├─ 报告、Artifact路径、大小和SHA
  ├─ GeoParquet中间文件是否按合同清理
  ├─ 模型加载、驻留和复用次数
  ├─ 缓存、内存、磁盘和耗时
  └─ 所有硬门是否通过
          │
          ▼
run_manifest.json + run_report.json + scale_acceptance_report.json
          │
          ▼
QGIS LayerManager加载四条结果流
```

监控界面从 PostgreSQL 的一个只读快照读取任务、阶段和 Artifact 摘要：

```text
PostgreSQL
  ├─ Job lease / heartbeat / retry / resume
  ├─ Artifact path / SHA / ref_count / cleanup
  ├─ Stream和空间单元状态
  └─ Work Package、V3.3、unit_confidence、unit_fit阶段进度
          │
          ▼
InferenceMonitorDialog
  ├─ 当前阶段进度
  └─ 整体任务完成度
       Work Package + V3.3 + unit_confidence + unit_fit + 栅格收口 + 四流组装 + 验收
```

整体进度表示任务完成度，不是剩余时间预测。

## 9. 停止、失败与恢复

```text
停止或进程失败
  ├─ 租约任务变为 interrupted/failed
  ├─ 已提交且SHA匹配的Artifact保留
  ├─ Tile缓存有效则复用
  ├─ 模型概率checkpoint有效则复用
  ├─ Partition、V3.3 staged和单元结果有效则复用
  └─ 仅重跑未完成、失效或明确重置的任务
```

Artifact 只有在所有消费者释放引用后才能清理。运行中 PID 不是完成证据，必须以
PostgreSQL Job、Stream、Artifact 和硬验收共同收口。

## 10. 自动推理后的人工整理

```text
选择通过验收的 Fusion
  │
  ▼
初始化14个类别工作层
  ├─ 手工编辑
  ├─ QGIS平滑
  └─ 可选 SAM3 边界候选
  │
  ▼
14类分别保存并确认
  │
  ▼
final_assembler.assemble_final
  │
  ▼
final/final_composite.gpkg
  │
  ▼
topology_validator.validate_topology
  │
  ▼
用户明确确认
  │
  ▼
accepted_writer.append_final_to_accepted
```

人工整理不会在自动推理完成时自动执行；SAM3 只产生边界候选，不判断类别。

## 11. 当前自动入口与独立工具边界

`V5AsyncInferenceRunner` 当前自动调用六个 Bash 入口：

```text
run_work_package.sh
run_fragmentation_v33_work_package.sh
run_unit_fit.sh
run_finalize_partition_rasters.sh
run_assemble_stream.sh
run_scale_acceptance.sh
```

`*_experiment.py`、`*_ab_validate.py`、V3.3 replay、独立 mosaic/polygonize、历史
fragmentation postprocess 等脚本属于实验、诊断、回放或显式工具，不会被当前 QGIS
自动主链调用。

## 12. 最终验收的 Job 计数合同

`scale_acceptance.py` 分别核对四类 Job，最后再核对总数：

```text
Work Package Job = Work Package 数量
V3.3 Job         = Partition 数量 + 1个全局 Finalize
unit_confidence   = V3.3 Fusion 的 Core/Seam/Junction 数量之和
unit_fit Job     = 每条 Stream 的 Core/Seam/Junction 数量之和
总 Job           = 上述四类之和
```

V3.3 未启用时，其预期 Job 数必须为 0；V3.3 启用时，Partition 和 Finalize 必须
全部 ready，不能只凭总 ready 数通过。基础空间单元数量来自 Run Spec 冻结的
Core/Seam/Junction 空间计划，并与 PostgreSQL 的每条 Stream `stream_units`
逐一核对，不把 V3.3 内部控制单元误算成矢量化单元。
