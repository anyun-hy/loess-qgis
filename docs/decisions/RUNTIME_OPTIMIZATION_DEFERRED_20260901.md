# 暂缓低收益或不满足一致性的运行时优化（2026-09-01）

## 决定

当前不实施以下四项运行时优化：

1. 跨 Work Package score cache；
2. 持久 geometry worker；
3. 默认启用 CUDA expandable segments；
4. 推理监控日志刷新降频。

这四项不得仅依据此前的理论收益估算重新进入生产代码。后续只有取得本文列出的
新证据，并形成新的项目决策，才重新评估。当前安全基线为 Git 提交
`9f198d4`；batch、Partition 尺寸、V3.3 policy、coverage 硬门和四流组装并发
继续保持不变。

## 跨 Work Package score cache

### 当前结论

不采用。当前设计虽然减少重复推理，但改变剩余 Tile 的 CUDA batch 组成，未保持
最终数据一致性。

### 实机证据

Tencent RTX 3090 使用真实五堡影像和 MambaOut 模型执行两个 Package A/B：

- 第二个 Package 的推理 Tile 从 6 降至 2；
- 第二个 Partition 有 334,569 个量化 probability 元素不同，占 2.90%；
- confidence 有 269,234 个像素不同；
- 模型 Core mask 改变 13 个像素；
- Fusion Core mask 改变 27 个像素。

因此，按 Tile 内容寻址不足以证明可复用；模型输出还受到 batch 组成和顺序影响。
相关候选已从运行时、规划器、状态库和测试中撤回。

### 重新评估条件

只有新方案同时满足以下条件才重新评估：

- 不改变每次模型前向的 batch 组成、顺序和有效 batch size，或能正式证明模型输出
  与这些因素无关；
- 三个正式模型、模型流和 Fusion 流的 probability、confidence、mask 全部逐数组
  严格一致；
- 并发发布、恢复、引用计数、磁盘预算和最后消费者清理均通过实机故障测试；
- 节省的墙钟时间足以抵消新增缓存和状态合同的复杂度。

## 持久 geometry worker

### 当前结论

不在当前生产架构中实施。方向具有潜在收益，但当前规模收益有限，长生命周期内存
和控制面合同尚未闭环。

### 实机证据

Tencent 对真实 0830 Core、Seam 和 Junction mask 执行隔离 A/B：

- 逐进程与持久进程的逻辑几何、报告统计，以及 raw、formal、fitted edges、
  boundary signatures 四类制品哈希一致；
- `conda run` 完整启动与 import 平均约 2.30 秒；
- 持久进程启动 RSS 约 467 MiB，连续 6 个混合任务后约 810 MiB；
- 峰值 RSS 约 1.25 GiB；
- 单次 Core 后执行 `malloc_trim`，RSS 仍比启动高约 177 MiB；
- 连续 20 个小任务后 RSS 约 651 MiB。

隔离测试覆盖 polygonize、边界平滑、置信度统计和 GeoParquet 制品，但没有写正式
Run 数据库，因而没有覆盖 lease、heartbeat、失败重试、StorageGuard 和 Artifact
原子提交全链。0830 同规模的保守墙钟收益约 40—80 秒，不足以支持本轮扩大运行时
架构。

### 重新评估条件

- 出现类似 0811 的大规模 Run，且启动开销重新成为可测量的关键路径；
- worker 具有最大任务数、RSS 上限、`malloc_trim` 和安全自动重启合同；
- 崩溃、超时、lease 丢失、重试和原子提交均通过 PostgreSQL 实机故障注入；
- 新旧执行方式对正式概率输入和所有最终制品严格一致；
- 端到端墙钟收益通过新 Run A/B，而不是由 CPU 秒数直接推算。

## CUDA expandable segments

### 当前结论

不写入默认生产配置。它已证明输出一致，但尚未证明能消除正式 Run 的 OOM 或产生
稳定且足够的端到端收益。

### 实机证据

PyTorch 2.6.0+cu124 的有效变量是：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

此前建议的 `PYTORCH_ALLOC_CONF` 会被该版本静默忽略。Tencent 三臂测试使用三个
正式 TorchScript 模型、冻结的 64 个真实 Tile 和 `64 → 32 → 64` 压力序列：

- 错误别名与默认配置行为相同，allocator snapshot 不包含 expandable segment；
- 正确变量生效后，三个模型 probability 与默认配置逐元素相等，最大差异为 0；
- 64-Tile 三模型总推理时间约从 41.58 秒降至 40.85 秒，提升约 1.8%；
- 本次默认与正确配置均未复现 OOM 或自动 batch 降档；
- 32 batch 阶段部分保留显存下降，但 Swin/Mamba 首次 64 batch 的保留显存略高。

这些结果只能证明变量生效和输出一致，不能证明正式 Run 的稳定性改善。

### 重新评估条件

- 在相同输入、相同部署身份和相同模型常驻顺序下稳定复现默认配置 OOM 或降档；
- 正确配置显著减少 OOM、降档或运行方差，且不增加硬磁盘或显存风险；
- 三个模型及最终模型/Fusion 制品逐像元一致；
- 至少一个完整新 QGIS Run 的端到端耗时和阶段计时支持采用结论。

## 推理监控日志刷新降频

### 当前结论

不修改。保留完整 JSONL 和当前推理监控呈现路径。

### 实机证据

在 macOS QGIS 4.2 / Qt6 的真实 `LogPanel` 中输入 15,889 条事件：

- 当前逐条呈现约 4.05 秒；
- 极端缓存后一次呈现约 3.46 秒；
- 总节省约 0.58 秒。

正式 Run 中这些事件分散在约 57.6 分钟内，墙钟收益可以忽略。修改刷新机制反而会
增加日志延迟、缓存状态和界面一致性成本。

### 重新评估条件

只有 QGIS 实机分析证明事件呈现造成可见卡顿、主线程长任务或稳定的运行时损失，
并且降频方案保持 warning/error 即时显示、完整 JSONL 和可恢复监控语义，才重新
评估。

## 使用边界

- 本文记录的是“当前不实施”的正式决定，不是永久否定研究方向；
- 理论速度、微基准或单一模型结果不能单独推翻该决定；
- 后续分析可以引用本文证据，但不得把隔离 A/B 描述成完整 QGIS Run 验收；
- 任何重新启用都必须形成新的决策文档，并明确替代本文的哪一项结论。
