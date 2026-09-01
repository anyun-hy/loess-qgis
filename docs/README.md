# Loess QGIS 文档入口

本目录只保留当前有效的项目说明、架构、状态、操作方法和长期技术决策。

## 阅读顺序

1. [PROJECT_IDEA.md](PROJECT_IDEA.md)：项目价值、范围和成功标准；
2. [ARCHITECTURE.md](ARCHITECTURE.md)：当前软件、数据和恢复合同；
3. [交互架构图](../visualizations/README.md)：高层运行时和数据库连接；
4. [PRODUCTION_CALL_GRAPH.md](diagrams/PRODUCTION_CALL_GRAPH.md)：生产调用关系；
5. [CURRENT_STATUS.md](CURRENT_STATUS.md)：已验证能力和仍需实机验证的边界；
6. [FRAGMENTATION_V3.md](operations/FRAGMENTATION_V3.md)：碎片治理方法；
7. [FRAGMENTATION_V33_SELECTION_20260826.md](decisions/FRAGMENTATION_V33_SELECTION_20260826.md)：V3.3 选择依据；
8. [RUNTIME_OPTIMIZATION_DEFERRED_20260901.md](decisions/RUNTIME_OPTIMIZATION_DEFERRED_20260901.md)：暂缓优化和重评条件。

## 文档职责

| 文档区域 | 内容 |
|---|---|
| `PROJECT_IDEA.md` | WHY、价值和成功标准 |
| `ARCHITECTURE.md` | 当前有效合同、数据流和边界 |
| `CURRENT_STATUS.md` | 当前证据和未完成验收 |
| `operations/` | 正式运行方法 |
| `decisions/` | 长期有效的技术选择 |
| `diagrams/` | 当前机制图 |
| 根 `visualizations/` | 交互架构图及 Archify JSON 源规格 |

一次性运行日志、用户路径、部署主机细节、会话交接和实验输出不进入公开文档。
测试通过、磁盘部署、QGIS 重启和真实输入验收必须分别表述。
