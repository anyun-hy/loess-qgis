# 当前实施状态

更新日期：2026-08-23。

## 当前结论

- macOS 与 Ubuntu 使用同一源码仓库和统一部署入口；
- PostgreSQL 是新正式 Run 的控制面状态库；
- 三个正式语义模型及 approved Fusion profile 已有历史资产与实机证据；
- Fragmentation V3 是当前生产碎片治理方案；
- 当前 RAG 与空间联合解码效果验收失败，源码和实验已移入 ignored archive；
- Generate 仅作研究参考；
- V3.1 尚未开始编码。

## 本地最近验证

当前本地 `main` 已提交 exact-range coverage、权威 Core 零缺口硬门、coverage
事件持久化和 QGIS 监控显示。

定向回归命令：

```text
conda run -n qgis pytest -q \
  tests/test_fragmentation_postprocess.py \
  tests/test_authoritative_raster.py \
  tests/test_stream_coverage_validation.py \
  tests/test_work_package_runtime.py \
  tests/test_run_builder_v5.py \
  tests/test_inference_monitor_v5.py \
  tests/test_assemble_stream_performance.py
```

结果：`72 passed in 28.38s`。

该结果证明上述本地代码路径，不等同于最新提交已完成双平台部署或 QGIS 现场
验收。

## 后处理方法状态

| 方法 | 当前地位 | 结论 |
|---|---|---|
| V3 | 生产 | 继续作为主方案 |
| Generate | ignored 研究参考 | 碎片少，但已有 gap/overlap 硬合同失败证据 |
| RAG v1 / full quick | ignored 失败归档 | 方法效果验收失败 |
| 单 Swin-B 空间联合解码 B | 腾讯实验归档 | 同域组件仅减少 2.963% |
| V3.1 | 未开始 | 先完成文档和工作树基线整理 |

详细决策见
[decisions/POSTPROCESSING_METHOD_20260823.md](decisions/POSTPROCESSING_METHOD_20260823.md)。

## 尚未取得的当前证据

- coverage 新硬门在最新本地提交上的双平台部署回读；
- coverage 状态在真实 QGIS 监控窗口中的现场验收；
- V3.1 的设计冻结、实现和同域 A/B；
- 使用同一模型、同一输入、同一完整 Mask 的 V3/Generate/新候选语义比较；
- 最近本地提交对应的正式 Tencent CUDA Run 和 macOS QGIS 现场验收。

## 历史证据

2026-07-15 至 2026-08-11 的完整测试、运行、部署和失败记录保存在：

- [archive/IMPLEMENTATION_HISTORY_20260715_20260811.md](archive/IMPLEMENTATION_HISTORY_20260715_20260811.md)
- [archive/PLUGIN_PLAN_V3_LEGACY.md](archive/PLUGIN_PLAN_V3_LEGACY.md)

历史文档不继续追加，也不能自动代表当前提交已通过相同验收。

## 更新规则

本文件只在“已验证能力、部署状态或阻塞发生实质变化”时更新。普通代码修改、
单次测试和会话交接不追加到这里；详细命令和日志由 commit、CI、Run Artifact
或历史归档承担。
