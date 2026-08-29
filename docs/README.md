# Loess QGIS 文档入口

本目录只保留当前有效的项目说明、架构、状态、运行手册和长期决策。历史过程
统一放入 `archive/`，研究候选及实验细节放入 Git 忽略的 `scratch/archive/`。

## 阅读顺序

1. [PROJECT_IDEA.md](PROJECT_IDEA.md)：项目为什么存在、范围和成功标准；
2. [ARCHITECTURE.md](ARCHITECTURE.md)：当前有效的软件与数据合同；
3. [diagrams/PRODUCTION_CALL_GRAPH.md](diagrams/PRODUCTION_CALL_GRAPH.md)：当前
   QGIS 自动推理、V3.3、四流组装和人工整理的完整调用关系；
4. [CURRENT_STATUS.md](CURRENT_STATUS.md)：当前已验证能力、未完成项和证据；
5. [operations/FRAGMENTATION_V3.md](operations/FRAGMENTATION_V3.md)：当前生产
   碎片治理方法；
6. [decisions/POSTPROCESSING_METHOD_20260823.md](decisions/POSTPROCESSING_METHOD_20260823.md)：
   V3、Generate、失败 RAG 和 V3.1 启动前的方法决策；
7. [decisions/FRAGMENTATION_V33_SELECTION_20260826.md](decisions/FRAGMENTATION_V33_SELECTION_20260826.md)：
   V3.1—V3.4 全量比较与 V3.3 选择决定；
8. [handoffs/V31_START_20260824.md](handoffs/V31_START_20260824.md)：用户明确
   请求的新对话 V3.1 启动交接。

## 文档职责

| 文档区域 | 允许内容 | 不允许内容 |
|---|---|---|
| `PROJECT_IDEA.md` | WHY、价值、范围 | 实现状态、测试流水账 |
| `ARCHITECTURE.md` | 当前有效合同、数据流、边界 | 日期日志、一次性实验结果 |
| `CURRENT_STATUS.md` | 当前事实、阻塞、最近验证 | 完整历史、未来设计草案 |
| `operations/` | 正式运行方法和操作合同 | 未验收研究候选 |
| `decisions/` | 长期有效的方法选择及理由 | 每次代码修改记录 |
| `diagrams/` | 当前机制图 | 无来源截图和临时导出 |
| `handoffs/` | 用户明确请求的跨会话任务交接 | 普通代码修改和测试流水账 |
| `archive/` | 只读历史证据 | 当前指导性结论 |

## 更新规则

1. 普通代码修改不创建 handoff 文档，也不追加日期流水账；
2. 架构或数据合同变化才修改 `ARCHITECTURE.md`；
3. 已验证能力、部署状态或阻塞实质变化才修改 `CURRENT_STATUS.md`；
4. 方法选择发生变化才新增或更新 `decisions/`；
5. 历史文档只归档，不继续追加；
6. 测试通过、fixture 通过、远程部署和 QGIS 实机验收必须分别表述；
7. 根 `README.md` 只链接本文，避免多个入口互相漂移。

## 历史与研究材料

- [archive/README.md](archive/README.md) 说明 tracked 历史文档；
- 失败 RAG 代码与详细证据位于
  `scratch/archive/rag_failed_20260823/`，不会进入 Git；
- Generate 研究说明位于
  `scratch/archive/generate_reference_20260823/`，不是生产文档。
- V3.1—V3.4 实验源码、测试和运行脚本保存在
  `research/fragmentation-v31-v34` 的 Git 历史，主干当前文件只保留 V3.3。
