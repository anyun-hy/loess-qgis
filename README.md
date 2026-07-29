# Ubuntu/macOS 合并隔离仓库

此目录是独立的嵌套 Git 仓库，仅用于 Ubuntu 代码审计以及后续 macOS/Ubuntu 合并。
它不继承 `/Users/example/Desktop/loess-data` 外层仓库的历史，也不会被加入外层仓库。

当前基线内容：

- `repository/`：Tencent Ubuntu `/home/example/Desktop/loess` 的源码快照。
- `installed_plugin/`：Ubuntu QGIS 实际安装的 `labeling_tool` 插件快照。
- `comparison/`：远端清单、SHA256、插件差异、`output/` 代码清单和验证结果。

初始 `main` 提交只用于固定拉取基线。后续合并修改应从该基线建立独立分支，
避免把外层项目的其他实验、测试或未提交改动混入合并历史。
