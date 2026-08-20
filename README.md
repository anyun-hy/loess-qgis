# Loess QGIS 标注与推理

此目录是 Loess 的唯一独立源码仓库，同时支持 Ubuntu QGIS 3.44/Qt5/CUDA
与 macOS QGIS 4/Qt6/MPS。插件、推理运行时和部署脚本共同使用这一份源码，
禁止建立完整的 macOS/Linux 双份实现。

`/Users/example/Desktop/loess-data` 是合并前的历史/原始数据工作目录。除非用户
在当前任务中明确点名并授权，不得搜索、读取、比较、复制或修改该目录，也不得
用其中的历史脚本或状态记录覆盖本仓库的当前事实。

正式源码直接位于仓库根目录：

- `qgis_plugins/labeling_tool/`：QGIS 插件、界面和运行编排。
- `inference_scripts/`：推理、组装、边界拟合及运行入口。
- `bash/`：项目初始化和插件安装脚本。
- `tests/`：自动化契约、故障注入和流水线测试。
- `docs/`：项目意图、唯一实施计划和验证证据。

插件安装与项目初始化是两个独立操作：

```sh
bash/install_plugin.sh --help
bash/init_project.sh --help
```

Tencent Ubuntu 远程命令统一通过项目入口执行：

```sh
bash/ssh_tencent.sh 'uname -a'
```

首次调用会建立 SSH 连接，后续调用复用该连接，默认闲置 15 分钟后自动关闭；
无需预先启动交互式 `ssh Tencent`，也不安装永久后台服务。

`init_project.sh` 在使用者选择的项目根目录中创建 `inference_scripts/`、
`runtime/`、`weights/`、`input/`、`qgis/` 和 `output/`。权重、用户输入、
运行输出和 QGIS 工程不进入本仓库，也不会由重复部署覆盖。

设计、实施和验收以 `docs/plugin_plan_v3.md` 为唯一准则；实际验证结果记录在
`docs/IMPLEMENTATION_STATUS.md`。初始 Tencent Ubuntu 拉取基线永久保存在
Git 标签 `ubuntu-baseline-2026-07-29` 中。
