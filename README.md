# Loess QGIS 标注与推理

跨平台 QGIS 半自动土地覆盖标注插件与 PostgreSQL 推理运行时，同时支持：

- Ubuntu：QGIS 3.44、Qt5/PyQt5、CUDA；
- macOS：QGIS 4.2、Qt6/PyQt6、MPS。

> **English summary:** A cross-platform QGIS plugin and PostgreSQL-backed
> inference runtime for semi-automatic land-cover labeling. The same source
> tree supports QGIS 3.44 on Ubuntu/CUDA and QGIS 4.2 on macOS/MPS.

## 代码组成

- `qgis_plugins/labeling_tool/`：QGIS UI、运行编排、监控和人工修整；
- `inference_scripts/`：Tile 推理、Fusion、V3/V3.3、矢量化和验收；
- `bash/`：插件安装、运行项目初始化和可选 SSH 辅助脚本；
- `tests/`：契约、故障注入、恢复和跨平台回归；
- `visualizations/`：可交互的运行时与数据库架构图。

完整文档从 [`docs/README.md`](docs/README.md) 进入。

## 模型边界

仓库**不包含**语义模型 TorchScript、SAM3 checkpoint、Fusion profile、输入影像、
范围数据、QGIS 工程、PostgreSQL 数据或 Run 输出。`inference_scripts/config.yaml`
只登记正式权重的文件名与可信 SHA-256；使用者必须自行取得有权使用的资产并放入
部署项目的 `weights/`。

## 快速开始

1. 安装 Miniconda/Anaconda、PostgreSQL 与目标版本 QGIS；
2. 克隆仓库并初始化运行项目：

   ```bash
   git clone https://github.com/anyun-hy/loess-qgis.git
   cd loess-qgis
   bash/init_project.sh --project-root "$HOME/Desktop/loess-project" --platform auto --create-env
   ```

3. 将有权使用的模型资产放入 `loess-project/weights/`，然后校验：

   ```bash
   bash/init_project.sh --project-root "$HOME/Desktop/loess-project" --platform auto --check-only --check-assets
   ```

4. 安装插件到 QGIS `default` profile：

   ```bash
   bash/install_plugin.sh --platform auto --profile default
   ```

5. 重启 QGIS，在插件中选择影像、研究范围和输出位置后创建新 Run。

PostgreSQL 默认使用当前系统用户名作为数据库名和角色名，并通过本机 Unix socket
连接。其他配置通过 `LOESS_STATE_DB_DSN` 和 `LOESS_STATE_DB_SCHEMA` 覆盖。

## 开发

```bash
conda run -n qgis pytest -q
conda run -n qgis python -m compileall -q qgis_plugins inference_scripts tests
```

贡献必须从功能分支提交 Pull Request；详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
安全问题请遵循 [SECURITY.md](SECURITY.md)，不要提交公开 Issue。

## 许可证

项目使用 [GNU GPL v3 或更高版本](LICENSE)。第三方资产和声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型与数据的许可不由本仓库授予。
