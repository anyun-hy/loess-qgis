# 源码实验与手动验证工具

以下入口位于源码仓库 `tools/experiments/`，不属于正式推理调用链，也不随
`scripts/deploy/init_project.sh` 或 `scripts/deploy/install_plugin.sh` 安装。旧 `inference_scripts/`
同名入口不保留跳转副本；外部手工命令需要更新脚本路径，参数不变。

## 运行方式

在仓库根目录使用 Conda `qgis` 环境查看参数；以下命令只显示帮助，不运行实验：

```bash
conda run -n qgis python tools/experiments/fragmentation_ab_experiment.py --help
conda run -n qgis python tools/experiments/subpixel_vectorize_experiment.py --help
conda run -n qgis python tools/experiments/evaluate_fragmentation_v33_replay.py --help
```

从其他目录运行时使用脚本的绝对路径，无需额外设置 `PYTHONPATH`。保留完整源码
仓库：工具引用其中的 `src/loess_runtime/`，回放评估另引用
`src/labeling_tool/shared/` 的共享合同。不要将单个脚本复制到部署项目并假设其依赖齐全。

## 输入与输出

| 工具 | 输入 | 输出及使用边界 |
| --- | --- | --- |
| 碎片治理 A/B | `--run-dir`、`--fusion-id`、`--source-raster`、`--manual-labels` | 保留既有 `fusion/<fusion-id>/raster_parts` 输入结构，使用 `--output-dir` 指定隔离输出 |
| 亚像素矢量实验 | `--scores`、`--raster`，可选 `--reference` | 概率数组为 `(14,H,W)`；`--output-dir` 保存 GPKG 与报告 |
| V3.3 回放评估 | `--run-spec`、`--v3-manifest`、`--historical-v33-manifest` | 校验 manifest、输入哈希与 140 分区合同，`--output` 指定报告路径 |

工具可读取部署项目的 Run 输入，但执行代码与依赖始终来自源码仓库。迁移不补齐
旧 Run 缺失的文件，也不保证任意新 Run 都满足历史实验输入结构。

显式输入路径沿用命令行相对当前工作目录的解释；manifest 内路径仍按原合同
解析。回放评估未指定 `--output` 时，按原有标记写入 Run 下的
`fusion/approved_replay/replay_evaluation.json` 或
`candidates/fragmentation_v33/replay_evaluation.json`。

实际执行会写文件，部分同名输出可能被覆盖；使用独立实验目录及显式报告路径，
不要将输出指向正式成果或人工确认数据。本次路径迁移没有增加备份或清理器。

## 保留边界

`fragmentation_postprocess.py` 位于 `src/loess_runtime/geometry/`，源码 shell 入口
位于 `scripts/runtime/`；生成项目仍提供原命名入口，历史成果读取合同保留。
V3/V3.3 生产算法归入几何模块。`boundary_ab_validate` 仍由亚像素实验的基础统计和
可选参考评估共同调用，不因迁移工具入口而删除。

部署源码指纹仅反映生产部署源，不涵盖这些实验入口；可用 Git 版本及本地变更
识别实验代码版本，不能用生产指纹证明实验脚本未变更。

## 手动验证入口

手动创建验证 Run、烟雾检查和压力工具位于 [tools/validation/](../../tools/validation/)，
不由 pytest 自动收集，也不随生产部署安装。原生 QGIS 回归探针位于
[tests/support/](../../tests/support/)，由对应 pytest 用例启动。

使用部署资产的三个入口可以先查看帮助：

```bash
conda run -n qgis python tools/validation/create_e9_formal_run.py --help
conda run -n qgis python tools/validation/create_e9_formal_sam3_job.py --help
conda run -n qgis python tools/validation/prepare_v5_l0_run.py --help
```

这三个入口要求显式传入 `--scripts-dir /path/to/loess-project/inference_scripts`，
以区分源码包位置和实际部署资产。配置指纹来自该项目的 manifest、identity 与
launcher，不再维护一份旧平铺 Python 文件清单。实际创建 Run 或执行验证会写入
输出，应按命令参数指定验证输入和输出位置；帮助命令不执行这些操作。

### 当前 V5 小范围真实模型验证

`prepare_v5_real_run.py` 使用已部署的插件与推理代码，从真实 GeoTIFF 中心选取
`832×832` 源像素，按 `512` 像素切片、`192` 像素重叠建立四片 Schema 2 Run。
它要求实际环境检查报告、CUDA、三个模型及 approved Fusion 资产，并核对部署
源码指纹。`run_v5_qt.py` 再调用已安装插件的线程化 runner，执行推理、组装和
终态发布；退出前等待 runner 关闭。两个入口不兼容旧 Schema 1 验证 Run。

准备工具使用 Conda `qgis` 环境；执行工具使用能导入 `qgis.PyQt` 的 QGIS 宿主
Python，模型子进程仍由部署项目的启动脚本进入 Conda 环境。先用 `--help` 查看
参数。必须显式指定插件目录、部署脚本目录、预期源码 SHA-256、隔离 PostgreSQL
DSN/schema，以及独立输出目录；禁止使用默认生产 schema。输出已有
`accepted_labels.gpkg` 时准备工具会拒绝创建。

实际准备会创建 Run 目录与数据库记录，执行会产生模型结果和验收文件。环境报告
必须对应当前部署，重新部署后应重新检查环境。可设置 `--timeout-seconds` 和
`--stop-grace-seconds` 限制测试时间；超时不算成功，仍需核对 Run 终态及产物。
这是小范围功能验证，不能证明整幅影像性能、长期稳定性或分类精度，也不会完成人工
确认和长期标签入库。验证工具本身不在生产源码指纹范围，应另外记录其版本或哈希。

执行工具的 `--action start|resume|retry-failed` 分别调用正式的新建执行、恢复、
重做失败包入口，默认 `start`。`--stop-after-work-package-ready` 会在控制面首次
出现已完成推理包后请求停止，并保存触发时状态和停止耗时。主动停止仍按未完成
运行返回退出码 `2`；核对报告中的 `pipeline_result.status=stopped`、终态发布、
关闭完成和 `stop_observation`，不能仅凭退出码断定停止测试成败。恢复时省略这个
停止选项，核对原 Run 的完成包、执行历史及产物身份，再判断是否真正恢复成功。
