# Loess QGIS 标注与推理

跨平台 QGIS 半自动土地覆盖标注插件与 PostgreSQL 推理运行时。

## 当前开发运行时目标

当前源码以 macOS 与腾讯 Ubuntu 的 QGIS 4.2 / PyQt6 / Qt6 为共同主机运行时：

| 平台 | QGIS | Qt / PyQt | 推理后端 | 推理环境 |
| --- | --- | --- | --- | --- |
| 腾讯 Ubuntu 26.04.1 | QGIS 4.2.2 | Qt6 / PyQt6 / Wayland | CUDA / RTX 3090 | 主机 Python 3.14.4；推理 Python 3.12、PyTorch 2.6.0 cu124 |
| macOS | QGIS 4.2.x | Qt6 / PyQt6 | MPS | Python 3.12.13、PyTorch 2.7.1 |

两个平台共用同一套插件、PostgreSQL 控制面和推理协议；QGIS 自带的
Python/Qt 运行时与独立的 `qgis` Conda 推理环境保持隔离。
Ubuntu 插件只支持原生 Qt6 Wayland QPA，不提供 X11/xcb 兼容入口。

`v1.0` 标签仍是 Ubuntu QGIS 3.44 / Qt5 的冻结回滚点；当前分支不再以
QGIS 3 或 Qt5 作为可部署目标。

> **English summary:** The current source targets QGIS 4.2 / PyQt6 / Qt6 on
> both macOS and Ubuntu, while retaining separate QGIS-host and Conda-inference
> runtimes. The v1.0 tag remains the frozen Ubuntu QGIS 3.44 / Qt5 rollback
> baseline.

## 代码组成

- `src/labeling_tool/`：按主面板、Run、监控、修整、QGIS 支持和共享合同分组的插件；
- `src/loess_runtime/`：按推理、几何、组装、SAM 和系统支持分组的独立运行时；
- `scripts/`：插件安装、运行项目初始化、推理启动和可选 SSH 辅助脚本；
- `configs/`：`defaults/` 保存默认配置和类别映射，`environments/` 保存两平台环境版本文件；
- `tests/`：契约、故障注入、恢复和跨平台回归；
- `tools/`：源码实验和手动验证工具，不进入生产部署；
- `visualizations/`：可交互的运行时与数据库架构图。

完整文档从 [`docs/README.md`](docs/README.md) 进入。

## 模型边界

仓库**不包含**语义模型 TorchScript、SAM3 checkpoint、Fusion profile、输入影像、
范围数据、QGIS 工程、PostgreSQL 数据或 Run 输出。`configs/defaults/config.yaml`
只登记正式权重的文件名与可信 SHA-256；使用者必须自行取得有权使用的资产并放入
部署项目的 `weights/`。

## 快速开始

首次使用的 PostgreSQL、profile 选择、连接诊断和安装后核对见
[`docs/operations/FIRST_INSTALL.md`](docs/operations/FIRST_INSTALL.md)。

1. 安装 Miniconda/Anaconda、PostgreSQL 与目标版本 QGIS；
2. 克隆仓库并初始化运行项目：

   ```bash
   git clone https://github.com/anyun-hy/loess-qgis.git
   cd loess-qgis
   scripts/deploy/init_project.sh --project-root "$HOME/Desktop/loess-project" --platform auto --create-env
   ```

3. 将有权使用的模型资产放入 `loess-project/weights/`，然后校验：

   ```bash
   scripts/deploy/init_project.sh --project-root "$HOME/Desktop/loess-project" --platform auto --check-only --check-assets
   ```

4. 确认目标 QGIS 当前使用的 profile 名称后安装插件。下面以 `QGIS4` 为示例；若你的当前 profile 不是这个名称，替换为实际名称：

   ```bash
   scripts/deploy/install_plugin.sh --platform auto --profile QGIS4
   ```

5. 重启 QGIS，在插件中选择影像、研究范围和输出位置后创建新 Run。

PostgreSQL 默认使用当前系统用户名作为数据库名和角色名，并通过本机 Unix socket
连接；这要求管理员已经准备好同名角色和数据库。连接检查、缺少角色/数据库时的处理、
替代 DSN 和安装后版本核对见 [首次安装说明](docs/operations/FIRST_INSTALL.md)。

## 开发

```bash
conda run -n qgis pytest -q
conda run -n qgis python -m compileall -q src scripts tools tests
```

贡献必须从功能分支提交 Pull Request；详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
安全问题请遵循 [SECURITY.md](SECURITY.md)，不要提交公开 Issue。

## 许可证

项目使用 [GNU GPL v3 或更高版本](LICENSE)。第三方资产和声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型与数据的许可不由本仓库授予。
