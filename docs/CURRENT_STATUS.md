# 当前实施状态

更新日期：2026-08-30。

## 当前结论

- macOS 与 Ubuntu 使用同一源码仓库和统一部署入口；
- PostgreSQL 是新正式 Run 的控制面状态库；
- Fragmentation V3.3 已接管生产权威栅格；V3 保留为冻结的第一阶段基线和新
  Run 的明确回滚选项；
- V3.1—V3.4 已在独立实验分支完成同域全量比较，最终选择 V3.3；
- 主干候选只保留可配置规则系统和 V3.3 实现，V3.1、V3.2、V3.4 的
  源码、测试和实验脚本仅保存在实验分支历史；
- V3.3 已接入 approved Fusion 正式 Work Package、状态库依赖和权威 Core
  发布流程；
- 最终 `scale_acceptance` 已按 Work Package、V3.3 和 `unit_fit` 三类 Job 分别
  核对冻结任务数与 ready 状态，V3.3 Partition/Finalize 不再被遗漏；
- 推理监控已把来源和严重程度分开，提供 Warning/Error 快捷定位、可读中文
  摘要、技术详情和按任务/尝试去重；结果流表格固定约五行并支持横纵滚动；
- 非像元对齐研究范围的权威 Core 掩膜已改为 all-touched raster support，最终
  仍按冻结精确矢量裁剪并执行 gap/overlap/outside 硬验收；
- Generate 仍只作研究参考，失败 RAG 与空间联合解码仍保持归档状态。

## V3.3 生产代码

- 规则系统：`inference_scripts/fragmentation_policy/`；
- 冻结生产配置：`inference_scripts/fragmentation_policy/policies/v33.yaml`；
- V3.3 执行器：`inference_scripts/fragmentation_v33_candidate/`；
- 正式任务：`inference_scripts/fragmentation_v33_work_package.py`；
- 查询能力覆盖类别权限、源吸收、目标增长、面积阈值、单类包围、多类别
  包围、同类桥接、关系优先级和 proposal 冲突顺序；
- `inference_scripts/config.yaml` 默认选择 V3.3；V3 仍生成冻结基线，但不再直接
  发布为生产 Fusion 权威 Core；
- 全部 Work Package ready 后，V3.3 以每 Partition 一个可恢复任务、最多四个并行
  执行，随后由唯一全局 Finalize 统一验收和发布；其完成前，同一 Fusion 的矢量
  空间单元不能领取。

## 全域实验结果

Tencent 同域测试覆盖 140 个 Core、831,531,565 个 strict-valid 像元：

| 方案 | 动态碎片数 | 相对上一主要方案 | 结论 |
|---|---:|---:|---|
| V3 | 30,239 | 基线 | 冻结基线/回滚方案 |
| V3.1-B | 25,983 | 比 V3 少 14.08% | 有效但提升有限 |
| V3.2 | 21,649 | 比 V3.1-B 少 16.68% | 继续改善 |
| V3.3 | 7,545 | 比 V3.2 少 65.15% | 当前生产方案 |
| V3.4 | 7,530 | 比 V3.3 只少 0.20% | 不采用 |

V3.3 相对 V3.2 减少 14,104 个动态碎片，动态碎片面积减少 69.65%，
组件减少 13,429，内部边界边减少 291,981。它改变 524,513 个像元，约占
完整有效范围 0.0631%。gap、overlap、outside、保护源类别损失均为 0。

V3.4 只比 V3.3 再减少 15 个动态碎片和 202 个像元，提升不足以支持采用。

## 生产接管验收

Tencent 使用冻结的 140-Core V3 基线和完整概率输入，执行正式 Work Package
生产路径，Run 为 `20260826_234000_v33production`：

- 140/140 个 Core 均发布权威 mask 和独立审计；
- 与此前选中的历史 V3.3 逐像元完全一致，差异像元为 0；
- V3 到 V3.3 改变 524,513 个像元；
- gap、overlap、outside、invalid、保护源类别损失均为 0；
- 概率非有限、负值、零和、错误归一化均为 0；
- 按分区审计口径，动态碎片从 45,188 降至 22,496。该口径不与上表的全域
  合并组件口径混算。

## 本地验证

当前主干的生产任务、V3/V3.3、PostgreSQL、GeoParquet、边界签名、四流组装、
范围转换、恢复和监控回归：

```text
522 passed, 5 skipped
```

独立真实 PostgreSQL 集成测试：

```text
3 passed
```

macOS 与 Tencent 当前部署项目和 QGIS 插件来自同一 `f6176dd` 源码与发布包，
source bundle 为
`2180b3c35a157a5761f5f42ce78bdc3c68c3032e48a67ebbd797b047b6eca1ad`。自动测试、
部署一致性和用户在 QGIS 中完成完整真实输入验收仍是三个不同结论。

## 尚未取得的证据

- 最新提交从原始影像重新推理到最终矢量组装的 140-Core 全链复跑；当前生产
  接管验收复用了已冻结、已校验的 V3 基线和概率输入；
- 最新提交对应的双平台 QGIS 现场验收。
- `f6176dd` 部署后尚未创建新 Run 验证非像元对齐范围的四流最终 coverage；旧
  Run `20260829_111032_087cae` 使用旧 Core mask，不能通过恢复来验证修复。

详细选择依据见
[decisions/FRAGMENTATION_V33_SELECTION_20260826.md](decisions/FRAGMENTATION_V33_SELECTION_20260826.md)。

## 历史证据

2026-07-15 至 2026-08-11 的完整测试、运行、部署和失败记录保存在：

- [archive/IMPLEMENTATION_HISTORY_20260715_20260811.md](archive/IMPLEMENTATION_HISTORY_20260715_20260811.md)
- [archive/PLUGIN_PLAN_V3_LEGACY.md](archive/PLUGIN_PLAN_V3_LEGACY.md)

历史文档不继续追加，也不能自动代表当前提交已通过相同验收。

## 更新规则

本文件只在已验证能力、部署状态或阻塞发生实质变化时更新。普通代码修改、
单次测试和会话交接不追加到这里；详细命令和日志由 commit、CI、Run Artifact
或历史归档承担。
