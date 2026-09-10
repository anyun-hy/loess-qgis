# 当前实施状态

## v1.0 冻结回滚点

`v1.0` 固化 Ubuntu 升级 QGIS 4.2 之前的稳定状态：

| 平台 | QGIS 主机运行时 | 独立推理运行时 |
| --- | --- | --- |
| Ubuntu 24.04 | QGIS 3.44.x、Qt5、PyQt5 | Python 3.12、PyTorch 2.6.0 cu124、CUDA、RTX 3090 |
| macOS | QGIS 4.2.x、Qt6、PyQt6 | Python 3.12.13、PyTorch 2.7.1、MPS |

该标签只保留为迁移前的历史回滚边界，不得修改标签指向；当前分支已经
停止交付 QGIS 3/Qt5 运行时。

## 当前开发运行时目标

| 平台 | QGIS 主机运行时 | 独立推理运行时 |
| --- | --- | --- |
| 腾讯 Ubuntu 26.04.1 | QGIS 4.2.2、PyQt6 6.10.2、Qt6 6.10.2、原生 Wayland QPA、Python 3.14.4 | Python 3.12、PyTorch 2.6.0 cu124、CUDA、RTX 3090 |
| macOS | QGIS 4.2.x、Qt6、PyQt6 | Python 3.12.13、PyTorch 2.7.1、MPS |

这是源码、插件元数据、安装路径与环境检查的共同合同；它不替代腾讯端的
QGIS GUI 实机验收。

## 已实现

- macOS 与 Ubuntu QGIS 4.2/PyQt6/Qt6 共用一套源码，分别使用 MPS 与 CUDA；
- Ubuntu 插件拒绝非 Wayland QPA；日志筛选采用有界缓存和分帧重绘；
- QGIS GUI、runtime 调度/QProcess、PostgreSQL 监控和 pipeline 日志写入已分离到
  明确线程；高频日志按 100ms 批量传输、进度按 key 合并，Job 心跳按 5 秒集中
  更新，Artifact 清理的 SHA/删除不再占用 Qt 线程；
- runtime、监控和 SAM3 界面关闭采用非阻塞生命周期；输入冻结与 accepted 审计、
  人工最终组装与拓扑检查已任务化，最终组装按批写入；
- 普通类别编辑按变化 FID 追踪，保存仅更新受影响类别的哈希；完整文件身份校验
  仍保留在恢复和最终组装边界，纯会话状态保存不重算类别文件；
- PostgreSQL-only Run 控制面，包含事务租约、恢复、失败包重置和 Artifact 引用；
- 原生 Qt 推理监控已分为总览、详细进度、结果与验收、事件与日志四页，采用
  可记忆的深蓝/浅色局部主题，并按实际 Run 显示 CUDA、MPS 或 CPU；停止仍在
  监控与主界面，恢复和重做失败包仍只在主界面；
- 新 Run 的开始、恢复、重做、Job 尝试、包内模型和十步组装阶段写入 PostgreSQL
  监控历史；历史尝试编号不复用重试预算，后续恢复保留旧失败并建立关联；
- 三个模型独立结果流与 approved Fusion 结果流；
- Work Package、Partition、V3 基线、V3.3 权威 Core 和四流并行组装；
- GeoParquet 中间数据、GeoPackage 最终数据和严格 coverage 验收；
- 公共分界拟合、人工分类工作区、SAM3 边界候选和 accepted labels；
- 可恢复的磁盘清理、最终制品大小观察报告和分阶段耗时。
- 几何 Job 使用 PSI/可用内存/Swap 驱动的动态并发与非阻塞降载；模型空间单元
  概率数组原位处理，并在矢量化前释放；

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

- 当前性能改造只有自动回归和本机原生 Qt/QGIS 临时数据验证，不等于实机大范围
  交互验收；调度数据库/最终哈希、跨线程日志背压、accepted 入库和大型单次 GEOS
  运算仍需单独优化或测量；
- 推理监控四页目前只有 macOS 原生 QGIS/Qt6 离屏样例状态截图和自动回归；真实
  MPS Run、腾讯 Wayland/CUDA Run、历史写入相对推理耗时不超过 5% 的目标仍需
  使用同一输入做实机测量；
- 每个公开提交对应的 macOS 与 Ubuntu 磁盘部署清单；
- 腾讯 Ubuntu QGIS 4.2 重启后从 `QGIS4` profile 实际加载插件、弹窗和新 Run；
- 腾讯原生 Wayland 同规模 Run 下验证推理监控显示/隐藏、日志切换、详情搜索和
  分页均无多秒冻结，并记录 GUI event-loop 延迟；
- 腾讯 Ubuntu 部署后以真实 `0.25m` Run 验证动态内存并发不会再次触发
  `systemd-oomd`，并核对日志中的 `[memory-admission]` 增容/降载轨迹；
- 使用者自己的完整模型资产和真实输入，从新 Run 到最终矢量的全链验收。
