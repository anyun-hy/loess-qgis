# 当前实施状态

## v1.0 冻结基线

`v1.0` 固化 Ubuntu 升级 QGIS 4.2 之前的稳定状态：

| 平台 | QGIS 主机运行时 | 独立推理运行时 |
| --- | --- | --- |
| Ubuntu 24.04 | QGIS 3.44.x、Qt5、PyQt5 | Python 3.12、PyTorch 2.6.0 cu124、CUDA、RTX 3090 |
| macOS | QGIS 4.2.x、Qt6、PyQt6 | Python 3.12.13、PyTorch 2.7.1、MPS |

后续 Ubuntu QGIS 4.2/Qt6 迁移必须以此标签为回滚边界，不得修改
`v1.0` 标签指向。

## 已实现

- macOS QGIS 4.2/Qt6/MPS 与 Ubuntu QGIS 3.44/Qt5/CUDA 共用一套源码；
- PostgreSQL-only Run 控制面，包含事务租约、恢复、失败包重置和 Artifact 引用；
- 三个模型独立结果流与 approved Fusion 结果流；
- Work Package、Partition、V3 基线、V3.3 权威 Core 和四流并行组装；
- GeoParquet 中间数据、GeoPackage 最终数据和严格 coverage 验收；
- 公共分界拟合、人工分类工作区、SAM3 边界候选和 accepted labels；
- 可恢复的磁盘清理、最终制品大小观察报告和分阶段耗时。

## 当前生产合同

- Fragmentation V3.3 是权威碎片治理方案，V3 是冻结基线和新 Run 的显式回滚项；
- `gap=0`、`overlap=0`、`outside=0` 是最终硬门；
- 模型、Fusion profile、输入和关键输出必须记录 SHA-256；
- 权重、输入、PostgreSQL 数据、QGIS 工程和 Run 输出不进入源码仓库；
- 历史文件状态库不恢复，必须用当前部署创建新 Run。

## 自动验证

本地完整测试应在项目 `qgis` Conda 环境运行：

```bash
conda run -n qgis pytest -q
```

GitHub `quality` 只覆盖无 GPU/QGIS 依赖的源码、Shell、文档和静态合同。它不代表
CUDA、MPS、QGIS GUI 或真实模型资产已经验收。

## 尚需独立验证

- 每个公开提交对应的 macOS 与 Ubuntu 磁盘部署清单；
- QGIS 重启后实际加载的插件路径和版本；
- 使用者自己的完整模型资产和真实输入，从新 Run 到最终矢量的全链验收。
