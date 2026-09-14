# 源码实验工具

以下入口位于源码仓库 `tools/experiments/`，不属于正式推理调用链，也不随
`bash/init_project.sh` 或 `bash/install_plugin.sh` 安装。旧 `inference_scripts/`
同名入口不保留跳转副本；外部手工命令需要更新脚本路径，参数不变。

## 运行方式

在仓库根目录使用 Conda `qgis` 环境查看参数；以下命令只显示帮助，不运行实验：

```bash
conda run -n qgis python tools/experiments/fragmentation_ab_experiment.py --help
conda run -n qgis python tools/experiments/subpixel_vectorize_experiment.py --help
conda run -n qgis python tools/experiments/evaluate_fragmentation_v33_replay.py --help
```

从其他目录运行时使用脚本的绝对路径，无需额外设置 `PYTHONPATH`。保留完整源码
仓库：工具引用其中的 `inference_scripts/`，回放评估另引用 `qgis_plugins/` 的
共享合同。不要将单个脚本复制到部署项目并假设其依赖齐全。

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

`fragmentation_postprocess.py` 及其 shell 入口仍在原处，历史成果读取兼容也保留。
V3/V3.3 生产算法未移动。`boundary_ab_validate` 仍由亚像素实验的基础统计和
可选参考评估共同调用，不因迁移工具入口而删除。

部署源码指纹仅反映生产部署源，不涵盖这些实验入口；可用 Git 版本及本地变更
识别实验代码版本，不能用生产指纹证明实验脚本未变更。
